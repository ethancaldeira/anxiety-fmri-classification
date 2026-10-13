"""
DatasetAdapter ABC.

Each adapter knows the dataset-specific bits:
  - where the subject IDs come from
  - what column the binary label is built from
  - where the preprocessed BOLDs and computed features live on S3
    (voxel-wise fingerprints and ROI-ROI matrices)

Everything else — splits, training, evaluation, results writing — is
dataset-agnostic and lives in data_loader / model / train / evaluate.

S3 access is optional. Features and BOLDs are looked up in the local
cache (data/cache/<dataset>/) first; S3 is only used for files that are
not there, and then needs the bucket name in FMRI_S3_BUCKET.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from botocore.exceptions import ClientError

# Buckets already confirmed reachable in this process.
_VERIFIED_BUCKETS: set[str] = set()


class S3AccessError(RuntimeError):
    """A file had to come from S3 but S3 is not configured or not accessible."""


def is_missing_key(err) -> bool:
    """True if a botocore ClientError means 'object does not exist'.

    Any other error code (AccessDenied, NoSuchBucket, expired or missing
    credentials, ...) is a configuration problem and must not be treated
    as a missing file.
    """
    code = str(err.response.get('Error', {}).get('Code', ''))
    return code in ('404', 'NoSuchKey', 'NotFound')


class DatasetAdapter(ABC):
    """Per-dataset adapter: subjects, labels, S3 paths."""

    #: short name used in EXPERIMENT_NAME, log lines, etc.
    name: str = ''

    #: S3 bucket holding preprocessed BOLDs and computed features
    #: (environment variable FMRI_S3_BUCKET; None = local files only)
    S3_BUCKET: Optional[str] = os.environ.get('FMRI_S3_BUCKET') or None

    def __init__(self, config):
        self.config = config

    def require_bucket(self, what: str, s3_client=None) -> str:
        """Return the S3 bucket name, or raise S3AccessError explaining that
        `what` is not available locally and no bucket is configured.

        With `s3_client`, also check once per process that the bucket is
        reachable (head_bucket), so that a wrong bucket name or missing
        permissions are reported instead of looking like missing files.
        """
        bucket = self.S3_BUCKET
        if not bucket:
            raise S3AccessError(
                f"{what} is not available locally and FMRI_S3_BUCKET is not set. "
                f"Either place the file in the local cache "
                f"(data/cache/{self.name}/, see README) or set FMRI_S3_BUCKET "
                f"to a bucket with the layout described in the README."
            )
        if s3_client is not None and bucket not in _VERIFIED_BUCKETS:
            try:
                s3_client.head_bucket(Bucket=bucket)
            except ClientError as e:
                raise S3AccessError(
                    f"Cannot access S3 bucket {bucket!r} (FMRI_S3_BUCKET): {e}. "
                    f"Check the bucket name and AWS credentials."
                ) from e
            _VERIFIED_BUCKETS.add(bucket)
        return bucket

    # ------------------------------------------------------------------
    # subjects + labels
    # ------------------------------------------------------------------

    @abstractmethod
    def load_subjects(self) -> pd.DataFrame:
        """Return a DataFrame with columns:

            subject_id : str           — used to build the S3 key
            label      : int (0 or 1)  — binary class label

        Subjects with missing labels MUST be filtered out before return.
        """

    # ------------------------------------------------------------------
    # S3 paths
    # ------------------------------------------------------------------

    @abstractmethod
    def s3_voxel_key(self, subject_id: str, atlas: str) -> str:
        """Return the S3 KEY (no bucket) for the voxel-wise fingerprint
        of this subject + atlas. Files are .npy with shape
        (31, 37, 31, N_ROIs) — drop-in for the 3D CNN."""

    @abstractmethod
    def s3_roi_roi_key(self, subject_id: str, atlas: str) -> str:
        """Return the S3 KEY (no bucket) for the (N_ROIs, N_ROIs) ROI-ROI
        matrix of this subject + atlas."""

    # ------------------------------------------------------------------
    # preprocessed BOLD access (used by src/compute_features.py)
    # ------------------------------------------------------------------

    def raw_fmri_s3_key(self, subject_id: str) -> Optional[str]:
        """Return the S3 KEY for this subject's preprocessed BOLD file
        (NOT bucket-qualified). Some datasets store BOLDs at a flat key
        (ABIDE) and some require a directory listing (GSP). For the
        listing case, override `find_raw_fmri_key` instead.

        Default: None — caller falls back to find_raw_fmri_key.
        """
        return None

    def find_raw_fmri_key(self, subject_id: str, s3_client) -> Optional[str]:
        """Locate this subject's BOLD on S3 — used when the key isn't
        derivable from the subject_id alone (e.g. GSP, where BOLDs live
        under a per-subject directory and we filter by filename pattern).

        Default: use the flat raw_fmri_s3_key if available.
        """
        return self.raw_fmri_s3_key(subject_id)

    def local_fmri_path(self, subject_id: str, fmri_dir: str) -> Optional[str]:
        """Return the path of this subject's preprocessed BOLD in the local
        cache directory `fmri_dir`, or None if it is not there.

        Default: the file name of raw_fmri_s3_key.
        """
        key = self.raw_fmri_s3_key(subject_id)
        if not key:
            return None
        path = os.path.join(fmri_dir, os.path.basename(key))
        return path if os.path.isfile(path) else None

    # ------------------------------------------------------------------
    # local cache layout (shared, dataset-aware)
    # ------------------------------------------------------------------

    def local_cache_dir(self, atlas: str, connectivity_type: str) -> str:
        """Where downloaded features get cached on disk.

        Layout: <BASE_DIR>/data/cache/<dataset>/<connectivity_type>/<atlas>/
        """
        sub = 'voxel' if connectivity_type == 'voxel' else 'roi_roi'
        return os.path.join(
            self.config.BASE_DIR, 'data', 'cache', self.name, sub, atlas,
        )
