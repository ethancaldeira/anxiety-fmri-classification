"""
ABIDEAdapter — ABIDE I (autism).

- Subjects come from data/abide/subjects_ABIDEI.txt + the phenotype CSV
  data/abide/Phenotypic_V1_0b_preprocessed1.csv (the latter has the
  DX_GROUP diagnosis column we need).
- Binary label: DX_GROUP == 1  ->  1 (ASD), DX_GROUP == 2  ->  0 (Control).
- S3 layout (bucket from FMRI_S3_BUCKET):
    voxel-wise fingerprints  : s3://<FMRI_S3_BUCKET>/processed/{atlas}/{SUB_ID}.npy
    ROI-ROI connectivity     : s3://<FMRI_S3_BUCKET>/connectivity/{atlas}/{SUB_ID}.npy
    preprocessed BOLD        : s3://<FMRI_S3_BUCKET>/raw/{FILE_ID}_func_preproc.nii.gz
  The preprocessed BOLDs are the Preprocessed Connectomes Project C-PAC
  nofilt_noglobal func_preproc files; they can also be placed directly in
  data/cache/abide/fmri/ under the same file names (see README).

Features are computed from the preprocessed BOLD by src/compute_features.py.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from .base import DatasetAdapter


class ABIDEAdapter(DatasetAdapter):
    name = 'abide'

    PHENOTYPE_REL  = 'data/abide/Phenotypic_V1_0b_preprocessed1.csv'
    SUBJECTS_REL   = 'data/abide/subjects_ABIDEI.txt'

    S3_VOXEL_PREFIX  = 'processed'     # processed/<atlas>/<SUB_ID>.npy
    S3_ROIROI_PREFIX = 'connectivity'  # connectivity/<atlas>/<SUB_ID>.npy
    S3_FMRI_PREFIX   = 'raw'           # raw/<FILE_ID>_func_preproc.nii.gz

    # ABIDE phenotype columns we read.
    SUB_COL  = 'SUB_ID'
    DX_COL   = 'DX_GROUP'           # 1 = ASD, 2 = Control (TDC)
    FILE_COL = 'FILE_ID'            # site-prefixed ID used in S3 filename

    # ------------------------------------------------------------------
    # subjects + labels
    # ------------------------------------------------------------------

    def load_subjects(self) -> pd.DataFrame:
        cfg = self.config
        pheno_path    = os.path.join(cfg.BASE_DIR, self.PHENOTYPE_REL)
        subjects_path = os.path.join(cfg.BASE_DIR, self.SUBJECTS_REL)
        if not os.path.isfile(pheno_path):
            raise FileNotFoundError(f"ABIDE phenotype CSV not found: {pheno_path}")
        if not os.path.isfile(subjects_path):
            raise FileNotFoundError(f"ABIDE subjects list not found: {subjects_path}")

        pheno = pd.read_csv(pheno_path)
        with open(subjects_path) as f:
            keep = {line.strip() for line in f if line.strip()}

        # SUB_ID is numeric in the CSV; compare both sides as stripped
        # strings, which is also the subject_id used for feature file names
        # (<SUB_ID>.npy).
        pheno[self.SUB_COL] = pheno[self.SUB_COL].astype(str).str.strip()
        keep = {str(s).strip() for s in keep}

        df = pheno[pheno[self.SUB_COL].isin(keep)].copy()
        df = df.dropna(subset=[self.DX_COL]).copy()

        # ABIDE DX_GROUP: 1 = ASD, 2 = Control. Map to 1 / 0.
        df['label'] = (df[self.DX_COL].astype(int) == 1).astype(int)
        df = df.rename(columns={self.SUB_COL: 'subject_id'})

        if getattr(cfg, 'LABEL_PERMUTE', False):
            seed = getattr(cfg, 'PERMUTE_SEED', 1234)
            rng = np.random.default_rng(seed)
            shuffled = df['label'].values.copy()
            rng.shuffle(shuffled)
            df['label'] = shuffled
            print(f"  [sanity] LABEL_PERMUTE ON: shuffled ABIDE labels "
                  f"({len(df)} subjects, seed={seed})")

        n_pos = int((df.label == 1).sum())
        n_neg = int((df.label == 0).sum())
        print(f"  ABIDE: {len(df)} subjects loaded "
              f"(DX_GROUP: {n_pos} ASD / {n_neg} Control, "
              f"{100.0 * n_pos / max(1, len(df)):.1f}% ASD)")

        return df[['subject_id', 'label']].reset_index(drop=True)

    # ------------------------------------------------------------------
    # S3 paths
    # ------------------------------------------------------------------

    def s3_voxel_key(self, subject_id: str, atlas: str) -> str:
        return f"{self.S3_VOXEL_PREFIX}/{atlas}/{subject_id}.npy"

    def s3_roi_roi_key(self, subject_id: str, atlas: str) -> str:
        return f"{self.S3_ROIROI_PREFIX}/{atlas}/{subject_id}.npy"

    # ------------------------------------------------------------------
    # preprocessed BOLD lookup — ABIDE has a flat layout keyed by FILE_ID,
    # which we read from the phenotype CSV. Cached after first lookup.
    # ------------------------------------------------------------------

    _file_id_cache: dict | None = None

    def _ensure_file_id_cache(self):
        if self._file_id_cache is not None:
            return
        pheno_path = os.path.join(self.config.BASE_DIR, self.PHENOTYPE_REL)
        pheno = pd.read_csv(pheno_path)
        pheno[self.SUB_COL]  = pheno[self.SUB_COL].astype(str).str.strip()
        pheno[self.FILE_COL] = pheno[self.FILE_COL].astype(str).str.strip()
        self._file_id_cache = dict(zip(pheno[self.SUB_COL], pheno[self.FILE_COL]))

    def raw_fmri_s3_key(self, subject_id: str) -> Optional[str]:
        self._ensure_file_id_cache()
        fid = self._file_id_cache.get(str(subject_id).strip())
        if not fid or fid.lower() == 'nan':
            return None
        return f"{self.S3_FMRI_PREFIX}/{fid}_func_preproc.nii.gz"
