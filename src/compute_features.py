"""
compute_features.py — compute voxel-wise fingerprints and/or ROI-ROI
connectivity matrices for a dataset × atlas set.

CPU and I/O bound (no GPU needed). Works per subject: each preprocessed
BOLD is read once and every requested (atlas, feature) pair is computed
from it. Outputs go to the local feature cache,
data/cache/<dataset>/<voxel|roi_roi>/<atlas>/<subject>.npy, which is where
src/train.py reads them. Existing outputs are skipped, so an interrupted
run can be resumed.

Preprocessed BOLDs are read from data/cache/<dataset>/fmri/ if present,
otherwise downloaded from s3://<FMRI_S3_BUCKET>/ (see README). With
--upload, every output is also written to the bucket, outputs already in
the bucket are skipped, and each local BOLD is deleted once all of its
outputs are uploaded.

Usage:
    # the nine standard atlases, both feature types
    python src/compute_features.py --dataset abide --features voxel roi_roi --workers 8

    # one subject, one atlas
    python src/compute_features.py --dataset abide --atlases cc200_roi_atlas \
        --features roi_roi --limit 1

Voxel fingerprints: Pearson correlation of each grey-matter voxel with each
ROI's mean timeseries (voxels outside the grey-matter mask are zero), then
2× average pooling (Khosla et al. recipe). ROI-ROI: Fisher-z of the
ROI-mean correlation matrix.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Optional

import boto3
import nibabel as nib
import numpy as np
from botocore.exceptions import ClientError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, STANDARD_ATLASES                   # noqa: E402
from datasets import get_adapter                              # noqa: E402
from datasets.base import S3AccessError, is_missing_key       # noqa: E402


MNI_DIMS = (61, 73, 61)
SUPERVOXEL_DIMS = (31, 37, 31)


# ---------------------------------------------------------------------------
# core math
# ---------------------------------------------------------------------------

def corr2_coeff(A, B):
    """Row-wise Pearson correlation between (n_voxels, T) and (n_rois, T)
    matrices. Returns (n_voxels, n_rois)."""
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def avg_pool_3d(arr, pool_size=2):
    """AveragePooling3D(pool_size=2, padding='same') in pure numpy.
    Reduces (61, 73, 61, C) → (31, 37, 31, C)."""
    x, y, z, c = arr.shape
    px = (pool_size - x % pool_size) % pool_size
    py = (pool_size - y % pool_size) % pool_size
    pz = (pool_size - z % pool_size) % pool_size
    if px or py or pz:
        arr = np.pad(arr, ((0, px), (0, py), (0, pz), (0, 0)))
    nx, ny, nz, _ = arr.shape
    return arr.reshape(
        nx // pool_size, pool_size,
        ny // pool_size, pool_size,
        nz // pool_size, pool_size, c
    ).mean(axis=(1, 3, 5)).astype(np.float32)


def compute_voxel_fingerprint(img_data, atlas_data, mask_data, nodes, n_rois):
    """(31, 37, 31, n_rois) supervoxel fingerprint for one subject."""
    roi_means = []
    for n in nodes:
        if n > 0:
            roi_means.append(np.mean(img_data[atlas_data == n], axis=0))
    roi_means = np.asarray(roi_means)                              # (n_rois, T)
    masked = img_data[mask_data > 0].reshape(-1, img_data.shape[-1])  # (n_vox, T)

    n_voxels = MNI_DIMS[0] * MNI_DIMS[1] * MNI_DIMS[2]
    fp = np.zeros((n_voxels, n_rois), dtype=np.float32)
    fp[(mask_data > 0).reshape(-1)] = corr2_coeff(masked, roi_means).astype(np.float32)
    fp[np.isnan(fp)] = 0
    fp = fp.reshape(*MNI_DIMS, n_rois)
    return avg_pool_3d(fp)


def compute_roi_roi_connectivity(img_data, atlas_data, nodes, n_rois, fisher_z=True):
    """(n_rois, n_rois) Fisher-z ROI-ROI connectivity matrix."""
    roi_ts = []
    for n in nodes:
        if n > 0:
            voxels = img_data[atlas_data == n]
            roi_ts.append(np.mean(voxels, axis=0) if voxels.shape[0] > 0
                          else np.zeros(img_data.shape[-1]))
    roi_ts = np.array(roi_ts)                                       # (n_rois, T)
    conn = np.corrcoef(roi_ts)
    conn[np.isnan(conn)] = 0.0
    np.clip(conn, -1.0, 1.0, out=conn)
    if fisher_z:
        np.clip(conn, -0.999999, 0.999999, out=conn)
        conn = np.arctanh(conn)
        np.fill_diagonal(conn, 0.0)
    return conn.astype(np.float32)


# ---------------------------------------------------------------------------
# atlas loading
# ---------------------------------------------------------------------------

ATLAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'atlases')


def load_atlas(atlas_name):
    """Returns (atlas_data, nodes, n_rois)."""
    path = os.path.join(ATLAS_DIR, f"{atlas_name}.nii.gz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Atlas not found: {path}")
    atlas_data = nib.load(path).get_fdata()
    nodes = sorted(int(x) for x in np.unique(atlas_data) if x > 0)
    return atlas_data, nodes, len(nodes)


def load_mask(mask_path):
    return nib.load(mask_path).get_fdata()


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_error(bucket, key, err) -> S3AccessError:
    return S3AccessError(f"S3 request for s3://{bucket}/{key} failed: {err}. "
                         f"Check AWS credentials and FMRI_S3_BUCKET.")


def s3_key_exists(s3_client, bucket, key) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if is_missing_key(e):
            return False
        raise _s3_error(bucket, key, e) from e


def download_fmri(adapter, subject_id, cache_dir, s3_client) -> Optional[str]:
    """Return the local path of the subject's preprocessed BOLD, downloading
    it from S3 if it is not in cache_dir. Returns None if the subject has no
    BOLD on S3."""
    local = adapter.local_fmri_path(subject_id, cache_dir)
    if local:
        return local
    bucket = adapter.require_bucket(f"Preprocessed BOLD for {subject_id}", s3_client)
    os.makedirs(cache_dir, exist_ok=True)
    key = adapter.find_raw_fmri_key(subject_id, s3_client)
    if not key:
        return None
    local_path = os.path.join(cache_dir, os.path.basename(key))
    try:
        s3_client.download_file(bucket, key, local_path)
        return local_path
    except ClientError as e:
        if is_missing_key(e):
            return None
        raise _s3_error(bucket, key, e) from e


# ---------------------------------------------------------------------------
# per-subject worker
# ---------------------------------------------------------------------------

def _process_subject(args_tuple):
    """Worker: download BOLD once, compute all (atlas, feature) tuples,
    write locally and (optionally) upload to S3.

    args_tuple is pickled across the ProcessPoolExecutor boundary.
    """
    (dataset_name, subject_id, atlases, features, upload,
     mask_path, cache_root, dry_run) = args_tuple

    # Re-init per worker (S3 client, Config, adapter).
    s3 = boto3.client('s3')
    config = Config
    adapter = get_adapter(dataset_name, config)

    fmri_cache = os.path.join(cache_root, dataset_name, 'fmri')
    mask_data = load_mask(mask_path)

    results = []   # list of (subject_id, atlas, feature, status)

    # Build the per-subject TODO list (skip what already exists).
    todo = []
    for atlas in atlases:
        for feature in features:
            local_dir = os.path.join(
                cache_root, dataset_name, feature, atlas,
            )
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, f"{subject_id}.npy")
            s3_key = (adapter.s3_voxel_key(subject_id, atlas) if feature == 'voxel'
                      else adapter.s3_roi_roi_key(subject_id, atlas))

            if upload and s3_key_exists(s3, adapter.S3_BUCKET, s3_key):
                results.append((subject_id, atlas, feature, 'skip-s3-exists'))
                continue
            if os.path.exists(local_path):
                if upload:
                    try:
                        s3.upload_file(local_path, adapter.S3_BUCKET, s3_key)
                        results.append((subject_id, atlas, feature, 'uploaded-from-cache'))
                    except ClientError as e:
                        results.append((subject_id, atlas, feature, f'upload-failed:{e}'))
                else:
                    results.append((subject_id, atlas, feature, 'skip-local-exists'))
                continue

            todo.append((atlas, feature, local_path, s3_key))

    if not todo:
        return results

    if dry_run:
        for atlas, feature, _, _ in todo:
            results.append((subject_id, atlas, feature, 'todo-dry-run'))
        return results

    # We need to actually compute. Download the BOLD now.
    local_fmri = download_fmri(adapter, subject_id, fmri_cache, s3)
    if local_fmri is None:
        for atlas, feature, _, _ in todo:
            results.append((subject_id, atlas, feature, 'fmri-not-found'))
        return results

    try:
        img_data = nib.load(local_fmri).get_fdata(dtype=np.float32)
    except Exception as e:
        for atlas, feature, _, _ in todo:
            results.append((subject_id, atlas, feature, f'fmri-load-failed:{e}'))
        return results

    # Group TODO by atlas so we load each atlas + compute its ROI means once.
    atlases_needed = sorted({atlas for atlas, _, _, _ in todo})
    for atlas in atlases_needed:
        try:
            atlas_data, nodes, n_rois = load_atlas(atlas)
        except Exception as e:
            for a, feature, _, _ in todo:
                if a == atlas:
                    results.append((subject_id, atlas, feature, f'atlas-load-failed:{e}'))
            continue

        for a, feature, local_path, s3_key in todo:
            if a != atlas:
                continue
            try:
                if feature == 'voxel':
                    arr = compute_voxel_fingerprint(img_data, atlas_data, mask_data, nodes, n_rois)
                else:
                    arr = compute_roi_roi_connectivity(img_data, atlas_data, nodes, n_rois)
                np.save(local_path, arr)
                status = 'computed'
                if upload:
                    try:
                        s3.upload_file(local_path, adapter.S3_BUCKET, s3_key)
                        status = 'computed+uploaded'
                    except ClientError as e:
                        status = f'computed-upload-failed:{e}'
                results.append((subject_id, atlas, feature, status))
            except Exception as e:
                results.append((subject_id, atlas, feature, f'compute-failed:{e}'))

    # Free memory
    del img_data
    gc.collect()

    # When uploading, the local BOLD is now redundant — delete it so
    # peak disk stays bounded at ~workers × per-BOLD size (~8 × 100 MB)
    # rather than accumulating ~150 GB across a full GSP+ABIDE pass.
    # We only delete when upload=True AND nothing in this subject's todo
    # failed to upload — anything left in `todo` that wasn't uploaded
    # still needs the BOLD if the user retries.
    if upload and local_fmri and os.path.exists(local_fmri):
        all_uploaded = all(
            r[3] in ('computed+uploaded', 'uploaded-from-cache', 'skip-s3-exists')
            for r in results
            if r[0] == subject_id
        )
        if all_uploaded:
            try:
                os.remove(local_fmri)
            except OSError:
                pass

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['gsp', 'abide'])
    ap.add_argument('--atlases', nargs='+', default=None,
                    help=f'Atlas names (files in src/templates/atlases/). '
                         f'Default: the nine standard atlases {STANDARD_ATLASES}')
    ap.add_argument('--features', nargs='+', default=['voxel', 'roi_roi'],
                    choices=['voxel', 'roi_roi'])
    ap.add_argument('--limit', type=int, default=None,
                    help='Process only the first N subjects (for testing)')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1),
                    help='Parallel subject workers (default: all but 1 CPU)')
    ap.add_argument('--mask-file', default=os.path.join(
        os.path.dirname(ATLAS_DIR), 'masks', 'grey_matter_mask.nii.gz'))
    ap.add_argument('--cache-root', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cache'))
    ap.add_argument('--upload', action='store_true',
                    help='Also upload each feature to s3://<FMRI_S3_BUCKET>/ '
                         '(skips features already there) and delete each '
                         "subject's local BOLD once all of its features "
                         'are uploaded.')
    ap.add_argument('--dry-run', action='store_true',
                    help='List what would be computed without doing it.')
    args = ap.parse_args()

    atlases  = args.atlases or STANDARD_ATLASES
    features = args.features

    print('=' * 70)
    print(f'compute_features.py')
    print(f'  dataset:  {args.dataset}')
    print(f'  atlases:  {atlases}')
    print(f'  features: {features}')
    print(f'  workers:  {args.workers}')
    print(f'  upload:   {args.upload}')
    print(f'  dry-run:  {args.dry_run}')
    print('=' * 70)

    # Enumerate subjects from the adapter.
    adapter = get_adapter(args.dataset, Config)
    if args.upload:
        if not adapter.S3_BUCKET:
            sys.exit('ERROR: --upload needs FMRI_S3_BUCKET (the target bucket).')
        adapter.require_bucket('Upload target', boto3.client('s3'))
    subj_df = adapter.load_subjects()
    subject_ids = [str(s) for s in subj_df['subject_id'].tolist()]
    if args.limit:
        subject_ids = subject_ids[:args.limit]
    print(f'  subjects: {len(subject_ids)}\n')

    # Build per-subject work tuples.
    work = [(args.dataset, sid, atlases, features, args.upload,
             args.mask_file, args.cache_root, args.dry_run)
            for sid in subject_ids]

    # Run.
    t0 = time.time()
    done = 0
    summary: dict[str, int] = {}
    if args.workers == 1:
        for w in work:
            for sid, atlas, feature, status in _process_subject(w):
                summary[status] = summary.get(status, 0) + 1
            done += 1
            if done % 25 == 0:
                _print_progress(done, len(work), summary, t0)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futs = [exe.submit(_process_subject, w) for w in work]
            for f in as_completed(futs):
                try:
                    for sid, atlas, feature, status in f.result():
                        summary[status] = summary.get(status, 0) + 1
                except S3AccessError:
                    exe.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as e:
                    summary[f'worker-error:{e}'] = summary.get(f'worker-error:{e}', 0) + 1
                done += 1
                if done % 25 == 0 or done == len(work):
                    _print_progress(done, len(work), summary, t0)

    print('\n' + '=' * 70)
    print(f'Total: {done} subjects in {(time.time()-t0)/60:.1f} min')
    for status, n in sorted(summary.items(), key=lambda kv: -kv[1]):
        print(f'  {status:40s} {n}')


def _print_progress(done, total, summary, t0):
    elapsed = (time.time() - t0) / 60
    rate = done / max(elapsed, 1e-6)
    eta = (total - done) / max(rate, 1e-6)
    print(f'  [{done}/{total}]  {elapsed:.1f} min elapsed  '
          f'{rate:.1f} subj/min  ETA {eta:.1f} min', flush=True)


if __name__ == '__main__':
    main()
