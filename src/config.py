"""
Configuration for the rs-fMRI classification pipeline (Ch5 and Ch6).

Mirrors the Ch4 (anxiety-smri-classification) Config style: a single
class with class-level attributes, env-var overrides for the knobs
you'd actually want to vary per-run, and an EXPERIMENT_NAME pattern
that lets `run_experiments.py` dispatch many cells without colliding.

The same training pipeline runs on either GSP (anxiety) or ABIDE
(autism) by setting `DATASET`. The dataset adapter (src/datasets/)
handles the per-dataset differences (subject list, label column,
S3 paths).
"""

import os


# The nine standard atlases of Chapter 5 (both datasets). Files live in
# src/templates/atlases/<name>.nii.gz.
STANDARD_ATLASES = [
    'aal_roi_atlas', 'cc200_roi_atlas', 'cc400_roi_atlas',
    'dos160_roi_atlas', 'ez_roi_atlas', 'ho_roi_atlas', 'tt_roi_atlas',
    'schaefer_200_7n_yeo', 'schaefer_400_7n_yeo',
]


class Config:
    """Configuration for the rs-fMRI classification pipeline."""

    # ==========================================
    # BASE PATHS
    # ==========================================

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')      # local cache (gitignored)

    # ==========================================
    # DATASET / TASK / FEATURES
    # ==========================================

    # Which dataset adapter to use. Adapters live in src/datasets/.
    #   gsp   -> GSP (Brain Genomics Superstruct Project), STAI-T binary
    #   abide -> ABIDE I, DX_GROUP binary (1=ASD -> 1, 2=Control -> 0)
    DATASET = os.environ.get('DATASET', 'gsp')

    # Task. Only 'classification' is implemented (see src/model.py for what
    # another task would need). Part of every run name.
    TASK = os.environ.get('TASK', 'classification')

    # Connectivity feature type:
    #   voxel    -> voxel-wise fingerprint, shape (31, 37, 31, N_ROIs), 3D CNN input
    #   roi_roi  -> ROI-ROI Fisher-z connectivity, upper triangle, MLP input
    CONNECTIVITY_TYPE = os.environ.get('CONNECTIVITY_TYPE', 'voxel')

    # Atlas name; names the feature cache / S3 feature directory. The Ch5
    # standard set is STANDARD_ATLASES above.
    ATLAS = os.environ.get('ATLAS', 'cc200_roi_atlas')

    # ==========================================
    # GSP-SPECIFIC
    # ==========================================

    # Binary cutoff applied to STAI_tAnxiety. Default 36 = the median of the
    # 877-subject GSP cohort (454 high / 423 low). The conventional clinical
    # cutoff of 40 is contested (Knight 1983; Julian 2011).
    # GSP only; ABIDE uses DX_GROUP directly.
    STAI_CUTOFF = int(os.environ.get('STAI_CUTOFF', '36'))

    # ==========================================
    # SANITY-CHECK HOOKS (matches Ch4)
    # ==========================================

    # LABEL_PERMUTE=1: subject-level label shuffle (negative control).
    # PERMUTE_SEED:    RNG seed for that shuffle.
    LABEL_PERMUTE = bool(int(os.environ.get('LABEL_PERMUTE', '0')))
    PERMUTE_SEED  = int(os.environ.get('PERMUTE_SEED', '1234'))

    # ==========================================
    # SPLIT / CROSS-VALIDATION
    # ==========================================

    TRAIN_SPLIT = 0.65
    VAL_SPLIT   = 0.15
    TEST_SPLIT  = 0.20
    RANDOM_SEED = 42

    # KEEP_CACHE: when False, train.py deletes the per-atlas feature cache
    # dir at the end of the run, which bounds disk use in multi-atlas
    # batches (GSP voxel features take ~25 GB per 200-ROI atlas). Default TRUE so
    # a single-cell manual run keeps its cache; the orchestrator's
    # multi-cell subcommands (run_standard_atlases, run_sp_atlases,
    # run_custom_atlases, run_group_ensemble, compare, all) set KEEP_CACHE=0.
    KEEP_CACHE = bool(int(os.environ.get('KEEP_CACHE', '1')))

    # GSP uses the fixed split in data/gsp_split.json (595/106/176
    # train/val/test, written by scripts/make_gsp_inputs.py). Ch5, Ch6 and
    # Ch7 share this held-out test set, and the Ch6 group parcellations are
    # built from its train subjects only. ABIDE draws a stratified
    # TRAIN/VAL/TEST_SPLIT split with RANDOM_SEED unless SPLIT_FILE is set.
    SPLIT_FILE = os.environ.get(
        'SPLIT_FILE',
        os.path.join(BASE_DIR, 'data', 'gsp_split.json') if DATASET == 'gsp' else '',
    ) or None

    # ==========================================
    # TRAINING
    # ==========================================
    # Ch5 trains the 3D CNN from scratch (no pretrained backbone), so we
    # use a single training stage — unlike Ch4's two-stage scheme.

    # Defaults follow the Khosla et al. Ensemble3DCNN_connectomes training
    # recipe (github.com/mk2299/Ensemble3DCNN_connectomes,
    # train_and_evaluate_classification.py):
    #   - SGD, momentum 0.9, learning rate 1e-3 with inverse-time decay 1e-3
    #   - batch size 64
    #   - 50 epochs, no early stopping
    #   - no class weighting
    BATCH_SIZE              = int(os.environ.get('BATCH_SIZE', '64'))
    EPOCHS                  = int(os.environ.get('EPOCHS', '50'))
    LEARNING_RATE           = float(os.environ.get('LEARNING_RATE', '1e-3'))
    SGD_DECAY               = float(os.environ.get('SGD_DECAY', '1e-3'))
    # Patience >= EPOCHS means training always runs all EPOCHS (as Khosla);
    # the lowest-val_loss weights are still restored at the end.
    EARLY_STOPPING_PATIENCE = int(os.environ.get('EARLY_STOPPING_PATIENCE', '50'))

    # ==========================================
    # OUTPUT
    # ==========================================

    # Default run-dir name pattern: {dataset}_{task}_{connectivity}_{atlas}
    # e.g. "gsp_classification_voxel_cc200_roi_atlas"
    # Override via EXPERIMENT_NAME env var for one-off names (sanity runs etc).
    EXPERIMENT_NAME = os.environ.get(
        'EXPERIMENT_NAME',
        f"{DATASET}_{TASK}_{CONNECTIVITY_TYPE}_{ATLAS}",
    )

    # Results are nested by chapter so Ch5 (standard atlases) and Ch6
    # (custom and stochastic parcellations) don't mix. run_experiments.py
    # sets this from the subcommand; a direct `python src/train.py` run
    # defaults to Ch5.
    CHAPTER      = os.environ.get('CHAPTER', '5')
    # Optional extra nesting level inside the chapter dir, set by orchestrator
    # subcommands that produce a lot of cells (SP_classification/{dataset}/
    # for the stochastic parcellation set). Empty by default: the standard,
    # individual and group cells are written to results/ch{N}/<EXPERIMENT_NAME>/.
    RESULTS_SUBDIR = os.environ.get('RESULTS_SUBDIR', '').strip('/')
    RESULTS_ROOT = (os.path.join(BASE_DIR, 'results', f'ch{CHAPTER}', RESULTS_SUBDIR)
                    if RESULTS_SUBDIR
                    else os.path.join(BASE_DIR, 'results', f'ch{CHAPTER}'))
    RUN_DIR      = os.path.join(RESULTS_ROOT, EXPERIMENT_NAME)
    PLOTS_DIR    = os.path.join(RUN_DIR, 'plots')

    @classmethod
    def make_run_dirs(cls):
        """Create the per-run output directory tree."""
        os.makedirs(cls.RUN_DIR, exist_ok=True)
        os.makedirs(cls.PLOTS_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        print("=" * 60)
        print(f"EXPERIMENT: {cls.EXPERIMENT_NAME}")
        print("=" * 60)
        print(f"\nDATASET / TASK:")
        print(f"  dataset:           {cls.DATASET}")
        print(f"  task:              {cls.TASK}")
        print(f"  connectivity:      {cls.CONNECTIVITY_TYPE}")
        print(f"  atlas:             {cls.ATLAS}")
        if cls.DATASET == 'gsp':
            print(f"  STAI cutoff:       {cls.STAI_CUTOFF}")
        print(f"\nSPLIT:")
        if cls.SPLIT_FILE:
            print(f"  fixed split from: {cls.SPLIT_FILE}")
        else:
            print(f"  single split — train/val/test = "
                  f"{cls.TRAIN_SPLIT}/{cls.VAL_SPLIT}/{cls.TEST_SPLIT}  seed {cls.RANDOM_SEED}")
        print(f"\nTRAINING:")
        print(f"  batch size:        {cls.BATCH_SIZE}")
        print(f"  epochs:            {cls.EPOCHS}")
        print(f"  learning rate:     {cls.LEARNING_RATE}  (SGD, inverse-time decay {cls.SGD_DECAY})")
        print(f"  early stop patience: {cls.EARLY_STOPPING_PATIENCE}")
        if cls.LABEL_PERMUTE:
            print(f"\n  [SANITY] LABEL_PERMUTE ON (seed {cls.PERMUTE_SEED})")
        print(f"\nOUTPUT:")
        print(f"  run dir:           {cls.RUN_DIR}")
        print("=" * 60)


if __name__ == '__main__':
    Config.print_config()
