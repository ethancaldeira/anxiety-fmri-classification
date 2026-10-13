"""
data_loader.py — load precomputed connectivity features (local cache, or
S3 for files not cached) and split by subject.

Flow:
    1. Adapter loads subjects + binary labels (`adapter.load_subjects()`).
    2. For each subject, read the .npy feature from the local cache
       (data/cache/<dataset>/<voxel|roi_roi>/<atlas>/); files that are not
       cached are downloaded from s3://<FMRI_S3_BUCKET>/ first.
    3. If feature shapes vary (some atlases drop ROIs when out-of-FOV),
       pad to the max shape so np.stack works.
    4. Split by SUBJECT: a fixed split file if configured, otherwise a
       stratified 65/15/20 train/val/test split with seed 42.

Connectivity-type handling:
    voxel    -> raw .npy is (31, 37, 31, N_ROIs); fed straight to 3D CNN.
    roi_roi  -> .npy is (N_ROIs, N_ROIs) Fisher-z matrix; we flatten the
                upper triangle (excluding diagonal) into a 1-D vector for MLP.

The S3 bucket and per-subject keys come entirely from the dataset adapter,
so this module is dataset-agnostic.
"""

from __future__ import annotations

import json
import os

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from sklearn.model_selection import train_test_split

from datasets.base import S3AccessError, is_missing_key


# ---------------------------------------------------------------------------
# S3 download with local cache
# ---------------------------------------------------------------------------

def _download_one(s3_client, adapter, key: str, local_path: str) -> bool:
    """Make sure local_path exists, downloading s3://<bucket>/key if needed.

    Returns True if the file is (now) cached and False if the object does
    not exist on S3. Raises S3AccessError if the file is not cached and S3
    is not configured, or if S3 refuses the request (credentials, bucket).
    """
    if os.path.exists(local_path):
        return True
    bucket = adapter.require_bucket(f"Feature file {local_path}", s3_client)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        s3_client.download_file(bucket, key, local_path)
        return True
    except ClientError as e:
        if is_missing_key(e):
            return False
        raise S3AccessError(
            f"Cannot read s3://{bucket}/{key}: {e}. "
            f"Check AWS credentials and FMRI_S3_BUCKET."
        ) from e


def _fetch_features(adapter, subjects_df: pd.DataFrame,
                    atlas: str, connectivity_type: str,
                    verbose_every: int = 100):
    """Return (X_array, valid_subjects_df). Drops subjects whose feature
    file is neither cached nor on S3."""
    s3_client = boto3.client('s3')
    cache_dir = adapter.local_cache_dir(atlas, connectivity_type)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"  Cache: {cache_dir}")

    arrays = []
    valid_rows = []
    n = len(subjects_df)

    for i, row in subjects_df.reset_index(drop=True).iterrows():
        sid = str(row['subject_id'])
        if connectivity_type == 'voxel':
            key = adapter.s3_voxel_key(sid, atlas)
        else:
            key = adapter.s3_roi_roi_key(sid, atlas)

        local_path = os.path.join(cache_dir, f"{sid}.npy")
        ok = _download_one(s3_client, adapter, key, local_path)
        if not ok:
            if (i + 1) % verbose_every == 0 or i < 5:
                print(f"  [{i+1}/{n}] {sid} — not on S3, skipping")
            continue

        try:
            arr = np.load(local_path)
        except Exception as e:
            print(f"  [{i+1}/{n}] {sid} — load error: {e}")
            continue

        # Some adapters' .npy files have leading singleton dim(s) (e.g.
        # ABIDE voxel files saved as (1, 31, 37, 31, 200)). Squeeze them
        # off so the per-subject feature is the expected rank
        # (voxel -> 4D, roi_roi -> 2D).
        expected_ndim = 4 if connectivity_type == 'voxel' else 2
        while arr.ndim > expected_ndim and arr.shape[0] == 1:
            arr = arr[0]

        arrays.append(arr)
        valid_rows.append(row)
        if (i + 1) % verbose_every == 0 or i < 3:
            print(f"  [{i+1}/{n}] {sid}  shape={arr.shape}")

    if not arrays:
        return None, subjects_df.iloc[:0].copy()

    # Pad to common shape if needed.
    max_shape = tuple(max(a.shape[d] for a in arrays) for d in range(arrays[0].ndim))
    if any(a.shape != max_shape for a in arrays):
        shapes = set(a.shape for a in arrays)
        print(f"  Padding feature arrays to {max_shape}  (found {shapes})")
        padded = []
        for a in arrays:
            if a.shape != max_shape:
                pad_width = [(0, max_shape[d] - a.shape[d]) for d in range(a.ndim)]
                a = np.pad(a, pad_width, mode='constant', constant_values=0)
            padded.append(a)
        arrays = padded

    X = np.stack(arrays, axis=0)
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    print(f"  Loaded {len(valid_df)}/{n} subjects.  X shape: {X.shape}")
    return X, valid_df


# ---------------------------------------------------------------------------
# ROI-ROI flattening (upper triangle, no diagonal)
# ---------------------------------------------------------------------------

def _flatten_roi_roi(X: np.ndarray) -> np.ndarray:
    """(N, R, R) -> (N, R*(R-1)/2): upper triangle (excl. diagonal)."""
    if X.ndim != 3 or X.shape[1] != X.shape[2]:
        raise ValueError(f"Expected square ROI-ROI matrices; got X.shape={X.shape}")
    r = X.shape[1]
    iu = np.triu_indices(r, k=1)
    return X[:, iu[0], iu[1]]


# ---------------------------------------------------------------------------
# subject-level stratified split
# ---------------------------------------------------------------------------

def split_subjects(subjects_df: pd.DataFrame, config):
    """Train/val/test split BY SUBJECT.

      1. If `config.SPLIT_FILE` is set, use the fixed split it holds
         (JSON with `train`, `val` and `test` subject-ID lists). GSP uses
         data/gsp_split.json; the Ch7 ABIDE runs use
         data/abide/abide_split_seed42.json.
      2. Otherwise (ABIDE in Ch5/Ch6), draw a stratified split with
         config.{TRAIN,VAL,TEST}_SPLIT and config.RANDOM_SEED.

    Returns three (subject_ids, labels) tuples.
    """
    sid_all = subjects_df['subject_id'].astype(str).values
    y_all   = subjects_df['label'].values
    sid_to_label = dict(zip(sid_all, y_all))
    have = set(sid_all)
    seed = config.RANDOM_SEED

    split_file = getattr(config, 'SPLIT_FILE', None)
    if split_file:
        if not os.path.isfile(split_file):
            hint = (" Create it with scripts/make_gsp_inputs.py."
                    if os.path.basename(split_file) == 'gsp_split.json' else '')
            raise FileNotFoundError(f"SPLIT_FILE not found: {split_file}.{hint}")
        with open(split_file) as f:
            sp = json.load(f)

        def _filter(ids):
            kept = [str(s) for s in ids if str(s) in have]
            dropped = len(ids) - len(kept)
            return kept, dropped

        train_ids, n_drop_tr = _filter(sp.get('train', []))
        test_ids,  n_drop_te = _filter(sp.get('test',  []))
        val_ids,   n_drop_v  = _filter(sp.get('val',   []))

        if not train_ids or not val_ids or not test_ids:
            raise RuntimeError(
                f"SPLIT_FILE {split_file} has no usable train/val/test after "
                f"intersecting with loaded subjects."
            )
        print(f"  Using fixed split from {split_file}")

        if n_drop_tr or n_drop_te or n_drop_v:
            print(f"  [warn] split-file IDs missing from current subjects df: "
                  f"train -{n_drop_tr}  val -{n_drop_v}  test -{n_drop_te}")

        sid_tr  = np.array(train_ids); y_tr  = np.array([sid_to_label[s] for s in train_ids])
        sid_val = np.array(val_ids);   y_val = np.array([sid_to_label[s] for s in val_ids])
        sid_te  = np.array(test_ids);  y_te  = np.array([sid_to_label[s] for s in test_ids])
    else:
        test_size = config.TEST_SPLIT
        val_size  = config.VAL_SPLIT / (1.0 - config.TEST_SPLIT)
        sid_trval, sid_te, y_trval, y_te = train_test_split(
            sid_all, y_all, test_size=test_size, stratify=y_all, random_state=seed,
        )
        sid_tr, sid_val, y_tr, y_val = train_test_split(
            sid_trval, y_trval, test_size=val_size, stratify=y_trval, random_state=seed,
        )
        print(f"  Random stratified split  (seed {seed})")

    print(f"  Split: train={len(sid_tr)}  val={len(sid_val)}  test={len(sid_te)}")
    print(f"    class balance — train: {dict(zip(*np.unique(y_tr,  return_counts=True)))}"
          f"   val: {dict(zip(*np.unique(y_val, return_counts=True)))}"
          f"   test: {dict(zip(*np.unique(y_te,  return_counts=True)))}")
    return (sid_tr, y_tr), (sid_val, y_val), (sid_te, y_te)


# ---------------------------------------------------------------------------
# top-level convenience
# ---------------------------------------------------------------------------

def load_and_prepare(config, adapter):
    """End-to-end: load subjects + features + split.

    Returns dict with keys:
        X_train, y_train, sid_train
        X_val,   y_val,   sid_val
        X_test,  y_test,  sid_test
    """
    print(f"\n--- Loading subjects ({adapter.name}) ---")
    subjects_df = adapter.load_subjects()

    print(f"\n--- Fetching {config.CONNECTIVITY_TYPE} features for atlas '{config.ATLAS}' ---")
    X_all, valid_df = _fetch_features(adapter, subjects_df,
                                      config.ATLAS, config.CONNECTIVITY_TYPE)
    if X_all is None:
        raise RuntimeError(
            f"No features available for {adapter.name} / {config.ATLAS} "
            f"/ {config.CONNECTIVITY_TYPE}. Run the appropriate compute step first."
        )

    if config.CONNECTIVITY_TYPE == 'roi_roi':
        X_all = _flatten_roi_roi(X_all)
        print(f"  Flattened ROI-ROI upper triangle: X shape now {X_all.shape}")

    print(f"\n--- Stratified subject split ---")
    (sid_tr, y_tr), (sid_val, y_val), (sid_te, y_te) = split_subjects(valid_df, config)

    sid_to_idx = {s: i for i, s in enumerate(valid_df['subject_id'].values)}
    def _gather(sids):
        return X_all[[sid_to_idx[s] for s in sids]]

    X_tr  = _gather(sid_tr)
    X_val = _gather(sid_val)
    X_te  = _gather(sid_te)

    return {
        'X_train': X_tr,   'y_train': y_tr,   'sid_train': sid_tr,
        'X_val':   X_val,  'y_val':   y_val,  'sid_val':   sid_val,
        'X_test':  X_te,   'y_test':  y_te,   'sid_test':  sid_te,
    }
