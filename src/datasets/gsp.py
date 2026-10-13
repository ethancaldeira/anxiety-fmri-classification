"""
GSPAdapter — Brain Genomics Superstruct Project (anxiety).

- Subjects come from data/subjects.txt (877 subjects with STAI_tAnxiety
  scores), written by scripts/make_gsp_inputs.py.
- Binary label: STAI_tAnxiety >= Config.STAI_CUTOFF (default 36 = cohort median).
- S3 layout (bucket from FMRI_S3_BUCKET):
    voxel-wise fingerprints  : s3://<FMRI_S3_BUCKET>/GSP_PREPROC/fingerprints/{atlas}/{subject}.npy
    ROI-ROI connectivity     : s3://<FMRI_S3_BUCKET>/GSP_PREPROC/connectivity/{atlas}/{subject}.npy
    preprocessed BOLD        : s3://<FMRI_S3_BUCKET>/GSP_PREPROC/output/pipeline_cpac_preproc/sub-NNNN/ses-1/func/
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

from .base import DatasetAdapter, S3AccessError


class GSPAdapter(DatasetAdapter):
    name = 'gsp'

    SUBJECTS_FILE = 'data/subjects.txt'

    S3_VOXEL_PREFIX  = 'GSP_PREPROC/fingerprints'
    S3_ROIROI_PREFIX = 'GSP_PREPROC/connectivity'
    S3_FMRI_PREFIX   = 'GSP_PREPROC/output/pipeline_cpac_preproc'

    # Friendly atlas names → actual S3 directory names. Used only when
    # building the S3 key so the rest of the pipeline (commands, results
    # dirs, local cache, plot titles) shows the short label.
    ATLAS_ALIASES = {
        'individual': 'individual_ROI_200',
    }

    @classmethod
    def _resolve_atlas(cls, atlas: str) -> str:
        return cls.ATLAS_ALIASES.get(atlas, atlas)

    # ------------------------------------------------------------------
    # subjects + labels
    # ------------------------------------------------------------------

    def load_subjects(self) -> pd.DataFrame:
        cfg = self.config
        path = os.path.join(cfg.BASE_DIR, self.SUBJECTS_FILE)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"GSP subjects file not found: {path}. "
                f"Create it with scripts/make_gsp_inputs.py (see README)."
            )

        df = pd.read_csv(path)
        if not {'participant_id', 'STAI_tAnxiety'}.issubset(df.columns):
            raise ValueError(
                f"GSP subjects.txt must have columns "
                f"'participant_id, STAI_tAnxiety'; got {list(df.columns)}"
            )

        # Drop subjects with missing STAI-T.
        df = df.dropna(subset=['STAI_tAnxiety']).copy()

        # Binary label from continuous STAI-T at the configured cutoff.
        cutoff = getattr(cfg, 'STAI_CUTOFF', 36)
        df['label'] = (df['STAI_tAnxiety'] >= cutoff).astype(int)
        df = df.rename(columns={'participant_id': 'subject_id'})

        # Optional negative-control: subject-level label shuffle.
        if getattr(cfg, 'LABEL_PERMUTE', False):
            seed = getattr(cfg, 'PERMUTE_SEED', 1234)
            rng = np.random.default_rng(seed)
            shuffled = df['label'].values.copy()
            rng.shuffle(shuffled)
            df['label'] = shuffled
            print(f"  [sanity] LABEL_PERMUTE ON: shuffled GSP labels "
                  f"({len(df)} subjects, seed={seed})")

        n_pos = int((df.label == 1).sum())
        n_neg = int((df.label == 0).sum())
        print(f"  GSP: {len(df)} subjects loaded "
              f"(STAI-T >= {cutoff}: {n_pos} positive / {n_neg} negative, "
              f"{100.0 * n_pos / max(1, len(df)):.1f}% positive)")

        return df[['subject_id', 'STAI_tAnxiety', 'label']].reset_index(drop=True)

    # ------------------------------------------------------------------
    # S3 paths
    # ------------------------------------------------------------------

    def s3_voxel_key(self, subject_id: str, atlas: str) -> str:
        return f"{self.S3_VOXEL_PREFIX}/{self._resolve_atlas(atlas)}/{subject_id}.npy"

    def s3_roi_roi_key(self, subject_id: str, atlas: str) -> str:
        return f"{self.S3_ROIROI_PREFIX}/{self._resolve_atlas(atlas)}/{subject_id}.npy"

    # ------------------------------------------------------------------
    # preprocessed BOLD lookup — BOLDs live under per-subject C-PAC output
    # directories, and the file is picked by its space/derivative tags.
    # ------------------------------------------------------------------

    BOLD_TAGS = ('space-MNI152NLin6ASym', 'reg-1', 'desc-preproc_bold')

    @staticmethod
    def _to_bids_id(raw_id: str) -> str:
        """SubNNNN_S1 -> sub-NNNN (matches the C-PAC output layout)."""
        m = re.match(r"Sub(\d+)_S\d+", raw_id)
        return f"sub-{int(m.group(1)):04d}" if m else raw_id

    def _is_bold(self, name: str) -> bool:
        return name.endswith('.nii.gz') and all(t in name for t in self.BOLD_TAGS)

    def local_fmri_path(self, subject_id: str, fmri_dir: str) -> Optional[str]:
        prefix = f"{self._to_bids_id(subject_id)}_"
        if not os.path.isdir(fmri_dir):
            return None
        hits = sorted(f for f in os.listdir(fmri_dir)
                      if f.startswith(prefix) and self._is_bold(f))
        return os.path.join(fmri_dir, hits[0]) if hits else None

    def find_raw_fmri_key(self, subject_id: str, s3_client) -> Optional[str]:
        bids_id = self._to_bids_id(subject_id)
        func_prefix = f"{self.S3_FMRI_PREFIX}/{bids_id}/ses-1/func/"
        bucket = self.require_bucket(f"Preprocessed BOLD for {subject_id}", s3_client)
        try:
            resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=func_prefix)
        except ClientError as e:
            raise S3AccessError(
                f"Cannot list s3://{bucket}/{func_prefix}: {e}. "
                f"Check AWS credentials and FMRI_S3_BUCKET."
            ) from e
        for obj in resp.get('Contents', []):
            key = obj['Key']
            if self._is_bold(os.path.basename(key)):
                return key
        return None
