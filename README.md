# Trait Anxiety and Autism Classification from rs-fMRI

Binary classification from resting-state fMRI connectivity on two cohorts:
trait anxiety (STAI-T) in the GSP and autism in ABIDE I. Two feature types
are supported: voxel-wise connectivity fingerprints, classified with a 3D
CNN, and ROI-ROI connectivity matrices, classified with an MLP. The same
pipeline runs on both datasets; the dataset is a configuration setting.

This repository contains the code for **Chapter 5** (standard atlases), the
training code for **Chapter 6** (custom parcellations) and the fMRI arms of
**Chapter 7** of a thesis on anxiety prediction from MRI:

- Ch4: structural MRI, 2D slice CNNs ([anxiety-smri-classification](https://github.com/ethancaldeira/anxiety-smri-classification))
- **Ch5 (this repo)**: rs-fMRI with nine standard atlases, GSP and ABIDE
- **Ch6 (this repo)**: rs-fMRI with individual, group and stochastic
  parcellations; the individual and group parcellations are built in
  [anxiety-fmri-parcellations](https://github.com/ethancaldeira/anxiety-fmri-parcellations)
- Ch7: multimodal sMRI + fMRI late fusion and explainability
  ([anxiety-multimodal-fusion](https://github.com/ethancaldeira/anxiety-multimodal-fusion));
  its fMRI arms are trained here

---

## Repository layout

```
anxiety-fmri-classification/
├── run_experiments.py              # orchestrator for every Ch5 / Ch6 run
├── run_abide_sharedsplit.sh        # ABIDE voxel models on the shared sMRI/fMRI split (Ch7)
├── requirements.txt                # Python environment (TensorFlow)
├── environment-pds.yml             # conda environment for src/generate_sp_masks.py
├── LICENSE
├── scripts/
│   └── make_gsp_inputs.py          # builds data/subjects.txt and data/gsp_split.json
├── data/
│   └── abide/
│       ├── subjects_ABIDEI.txt     # 774 ABIDE I subject IDs
│       └── abide_split_seed42.json # split shared with the ABIDE structural arm (Ch7)
├── preprocessing/                  # C-PAC on AWS Batch for GSP (see preprocessing/README.md)
├── src/
│   ├── config.py                   # all settings; environment variables override
│   ├── datasets/                   # GSP and ABIDE adapters: subjects, labels, file locations
│   ├── compute_features.py         # BOLD → voxel fingerprints / ROI-ROI matrices
│   ├── data_loader.py              # feature loading and caching, subject split
│   ├── model.py                    # 3D CNN and MLP
│   ├── train.py                    # one training run
│   ├── evaluate.py                 # metrics, plots, cross-run tables
│   ├── results_io.py               # output-file helpers
│   ├── generate_sp_masks.py        # Ch6 stochastic parcellations
│   ├── aggregate_sp_metrics.py     # Ch6 per-scale summaries
│   ├── aggregate_ensemble.py       # Ch6 soft-vote ensembles (group, stochastic)
│   ├── plot_architectures.py       # model diagrams
│   └── templates/                  # atlases, grey-matter mask, Schaefer label tables
│                                   # (sources and licences: src/templates/atlases/README.md)
└── results/                        # not tracked; results/ch5/, results/ch6/, results/ch7/
```

---

## Installation

```bash
git clone https://github.com/ethancaldeira/anxiety-fmri-classification.git
cd anxiety-fmri-classification
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used for the thesis results
(Python 3.12, TensorFlow 2.20, Keras 3.12). Regenerating the stochastic
parcellations needs a separate conda environment (`environment-pds.yml`,
see Chapter 6 below). GSP preprocessing additionally needs Docker and AWS
access (see [`preprocessing/README.md`](preprocessing/README.md)).

### Hardware

| Stage | Hardware used |
|---|---|
| Training (`src/train.py`) | AWS g4dn.2xlarge: one NVIDIA T4 GPU (16 GB), 8 vCPUs, 32 GB RAM, plus a 120 GB swap file |
| Preprocessing, feature computation | AWS c5.4xlarge: 16 vCPUs, 32 GB RAM (CPU only) |

Memory is dominated by the GSP voxel features: at 200 ROIs one in-memory
copy of all 877 fingerprints is ~25 GB (877 × 31×37×31×200 float32), and
~50 GB at 400 ROIs. Feature loading holds about two copies at its peak
(stacking, then the train/val/test arrays), so the GSP voxel runs need
RAM plus swap of roughly 50 GB at 200 ROIs and 100 GB at 400 ROIs. Training
streams one batch at a time to the GPU, so GPU memory is not the limit
(the 400-ROI atlases use batch size 32). ROI-ROI runs and ABIDE runs need
far less. Disk: ~44 GB for the 877 preprocessed GSP BOLDs, ~91 GB for the
774 ABIDE BOLDs, and ~25 GB per 200-ROI GSP voxel feature cache.

---

## Data

| | GSP | ABIDE I |
|---|---|---|
| Target | STAI-T ≥ 36 (cohort median) | diagnosis: ASD = 1, control = 0 |
| Subjects | 877 | 774 |
| Preprocessed BOLD | C-PAC 1.8.7 `preproc` preconfig, run by `preprocessing/` | Preprocessed Connectomes Project, C-PAC `nofilt_noglobal` |
| Split (train / val / test) | fixed: `data/gsp_split.json` (595 / 106 / 176) | stratified 65 / 15 / 20 %, seed 42 (502 / 117 / 155); Ch7: `data/abide/abide_split_seed42.json` (503 / 116 / 155) |

### GSP

The Brain Genomics Superstruct Project Open Access release
(Holmes et al., 2015; 1,570 participants) is available from
[neuroinformatics.harvard.edu/gsp](http://neuroinformatics.harvard.edu/gsp/)
under the GSP Open Access Data Use Terms. The STAI-T scores belong to the
GSP restricted phenotypic data, available on application under the GSP
restricted data use terms. No GSP subject data are distributed with this
repository; the two GSP input files are built locally from a participants
TSV with the columns `participant_id` (`SubNNNN_S1`) and `STAI_tAnxiety`
for the 879 participants with a usable STAI-T score:

```bash
python scripts/make_gsp_inputs.py --participants /path/to/GSP_participants.tsv
```

- **`data/subjects.txt`**: CSV with `participant_id` (GSP subject ID,
  `SubNNNN_S1`) and `STAI_tAnxiety`, for the 877 subjects with a
  preprocessed BOLD. All 879 participants were preprocessed; for two of
  them (Sub0105_S1 and Sub1382_S1) C-PAC functional preprocessing stopped
  at nuisance regression and wrote no preprocessed BOLD, so they are not
  part of the functional cohort. The script drops these two by default, or
  keeps exactly the subjects listed in `--preprocessed <file>`.
- **`data/gsp_split.json`**: the train/val/test split used for every GSP
  run in Chapters 5–7. Recipe: subjects sorted by `participant_id`,
  stratified on STAI-T ≥ 36 with scikit-learn `train_test_split`
  (`random_state=42`): 20 % test, then 15 % of the remainder as
  validation, giving 595 / 106 / 176. The Ch6 group parcellations are
  built from the training subjects of this split.
  `python scripts/make_gsp_inputs.py --participants ... --check` compares
  existing files with the recipe.
- **fMRI:** the first resting-state run (BOLD1) of each subject,
  preprocessed with C-PAC (see `preprocessing/`). Features use the
  `space-MNI152NLin6ASym_reg-1_desc-preproc_bold` output (3 mm).

### ABIDE I

Preprocessed BOLD and phenotypes come from the
[Preprocessed Connectomes Project](http://preprocessed-connectomes-project.org/abide/)
release of ABIDE I (C-PAC pipeline, `nofilt_noglobal` strategy), in the
public `fcp-indi` bucket. `data/abide/subjects_ABIDEI.txt` lists the 774
subjects used: those with preprocessed functional data that passed all
three PCP functional quality ratings. To download the phenotype table and
the 774 BOLDs into the local cache:

```bash
aws s3 cp --no-sign-request \
    s3://fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv data/abide/

mkdir -p data/cache/abide/fmri
python - <<'EOF' | xargs -P 8 -I{} aws s3 cp --no-sign-request --only-show-errors \
    s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/nofilt_noglobal/func_preproc/{}_func_preproc.nii.gz \
    data/cache/abide/fmri/
import pandas as pd
keep = set(open('data/abide/subjects_ABIDEI.txt').read().split())
pheno = pd.read_csv('data/abide/Phenotypic_V1_0b_preprocessed1.csv')
print('\n'.join(pheno.loc[pheno.SUB_ID.astype(str).isin(keep), 'FILE_ID']))
EOF
```

### Local cache and S3

Every input is looked up in the local cache first:

| | GSP | ABIDE |
|---|---|---|
| Preprocessed BOLD | `data/cache/gsp/fmri/sub-NNNN_*space-MNI152NLin6ASym_reg-1_desc-preproc_bold.nii.gz` | `data/cache/abide/fmri/<FILE_ID>_func_preproc.nii.gz` |
| Features | `data/cache/gsp/<voxel\|roi_roi>/<atlas>/<subject>.npy` | `data/cache/abide/<voxel\|roi_roi>/<atlas>/<subject>.npy` |

`src/compute_features.py` writes features to the cache and `src/train.py`
reads them from there. Files that are not cached are fetched from an S3
bucket named by `FMRI_S3_BUCKET`, with this layout:

| | GSP | ABIDE |
|---|---|---|
| Preprocessed BOLD | `GSP_PREPROC/output/pipeline_cpac_preproc/sub-NNNN/ses-1/func/` | `raw/<FILE_ID>_func_preproc.nii.gz` |
| Voxel fingerprints | `GSP_PREPROC/fingerprints/<atlas>/<subject>.npy` | `processed/<atlas>/<subject>.npy` |
| ROI-ROI matrices | `GSP_PREPROC/connectivity/<atlas>/<subject>.npy` | `connectivity/<atlas>/<subject>.npy` |

If a file is neither cached nor available because `FMRI_S3_BUCKET` is
unset, the run stops with an error naming the file; S3 access errors
(credentials, permissions, bucket name) also stop the run. The
credentials need `s3:ListBucket` and `s3:GetObject` on the bucket
(`s3:PutObject` for `--upload`). Without a bucket, place the BOLDs in `data/cache/<dataset>/fmri/` as above and
compute the features locally.

---

## Features

`src/compute_features.py` builds both feature types from the preprocessed
BOLD, using the atlases in `src/templates/atlases/` and the grey-matter
mask in `src/templates/masks/`:

- **Voxel fingerprints:** the Pearson correlation of every grey-matter
  voxel with each ROI's mean timeseries (voxels outside the mask are
  zero), average-pooled 2× to 31 × 37 × 31 × N_ROIs.
- **ROI-ROI:** the Fisher-z correlation matrix of the ROI mean
  timeseries; the model uses its upper triangle.

```bash
# the nine standard atlases (the default --atlases), both feature types
python src/compute_features.py --dataset gsp --features voxel roi_roi --workers 8 [--upload]
```

`--upload` also writes each feature to `s3://<FMRI_S3_BUCKET>/` and
deletes each subject's local BOLD from `data/cache/<dataset>/fmri/` once
all of its features are uploaded.

---

## Models and training

- **3D CNN** (voxel): Conv3D(128) → ELU → Conv3D(32) → ELU → MaxPool3D →
  Dense(32, ELU, L2 0.005) → Dense(1, sigmoid); ~5.8 M parameters at 200 ROIs.
- **MLP** (ROI-ROI): Dense(256) → BatchNorm → Dropout(0.3) → Dense(64) →
  BatchNorm → Dropout(0.3) → Dense(32) → Dense(1, sigmoid); ELU
  activations, L2 1e-4 on the first two layers; ~5.1 M parameters at 200 ROIs.
- **Training:** trained from scratch with binary cross-entropy, SGD
  (momentum 0.9, learning rate 1e-3 with inverse-time decay 1e-3),
  batch size 64 (32 for CC400 and Schaefer-400), for 50 epochs. The
  weights from the epoch with the lowest validation loss are used for
  testing.

All settings are in `src/config.py`. The ones varied across runs can be
set through environment variables:

| Variable | Values | Default |
|---|---|---|
| `DATASET` | `gsp`, `abide` | `gsp` |
| `CONNECTIVITY_TYPE` | `voxel`, `roi_roi` | `voxel` |
| `ATLAS` | atlas name (file in `src/templates/atlases/`, or `individual`) | `cc200_roi_atlas` |
| `STAI_CUTOFF` | integer (GSP) | `36` |
| `SPLIT_FILE` | path to a split JSON (`train`, `val`, `test` ID lists) | `data/gsp_split.json` for GSP; unset (stratified random split) for ABIDE |
| `LABEL_PERMUTE` | `0`, `1`: shuffle labels across subjects (negative control) | `0` |
| `PERMUTE_SEED` | seed of that shuffle | `1234` |
| `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE` | | `64`, `50`, `1e-3` |
| `FMRI_S3_BUCKET` | S3 bucket with the layout above | unset (local files only) |
| `EXPERIMENT_NAME` | run directory name | `<dataset>_classification_<conn>_<atlas>` |
| `CHAPTER` | results subfolder `ch<N>` | `5`; set by `run_experiments.py` |
| `RESULTS_SUBDIR` | extra level under `results/ch<N>/` | unset; `SP_classification/<dataset>` for the stochastic set |
| `KEEP_CACHE` | `0`: delete the run's feature cache afterwards | `1`; multi-cell subcommands of `run_experiments.py` use `0` |
| `SP_SAMPLES_PER_K` | stochastic parcellations per scale used by `run_sp_atlases` (1–30) | `30` |

A single run:

```bash
DATASET=gsp CONNECTIVITY_TYPE=roi_roi ATLAS=cc200_roi_atlas python src/train.py
```

---

## Reproducing the experiments

`run_experiments.py` launches each run as a separate `src/train.py`
process. Ch5 subcommands write to `results/ch5/` and Ch6 subcommands to
`results/ch6/`. Runs whose `metrics_subject.json` already exists are
skipped. Multi-cell subcommands delete each cell's feature cache after
training (`KEEP_CACHE=0`).

### Chapter 5: standard atlases

Nine atlases: `aal_roi_atlas`, `cc200_roi_atlas`, `cc400_roi_atlas`,
`dos160_roi_atlas`, `ez_roi_atlas`, `ho_roi_atlas`, `tt_roi_atlas`,
`schaefer_200_7n_yeo`, `schaefer_400_7n_yeo`. Compute their features first
(see [Features](#features)) unless they are already in `FMRI_S3_BUCKET`.

```bash
for dataset in gsp abide; do
  for conn in voxel roi_roi; do
    python run_experiments.py run_standard_atlases --dataset $dataset --connectivity $conn
  done
done
```

This produces 36 runs, `results/ch5/<dataset>_classification_<conn>_<atlas>/`.
Other Ch5 subcommands:

```bash
python run_experiments.py classify --dataset gsp --atlas cc200_roi_atlas --connectivity voxel
python run_experiments.py compare --atlas cc200_roi_atlas      # same cell on both datasets
python run_experiments.py validate --dataset gsp               # label-permutation control
```

### Chapter 6: custom parcellations

**Individual parcellations (GSP).** One parcellation per subject:
Craddock temporal-correlation (tcorr) normalised-cut parcellation of the
subject's own BOLD, neighbouring voxels connected at r ≥ 0.5, K = 200.
These parcellations and their features are produced by
[anxiety-fmri-parcellations](https://github.com/ethancaldeira/anxiety-fmri-parcellations),
which writes the features to
`<FMRI_S3_BUCKET>/GSP_PREPROC/{fingerprints,connectivity}/individual_ROI_200/`
(locally: `data/cache/gsp/<voxel|roi_roi>/individual/`). `run_custom_atlases`
trains on them under the atlas name `individual`.

**Group parcellations (GSP).** The training subjects of `data/gsp_split.json`
are divided into STAI-T tertiles (cut-offs 32 and 40), and a group-mean
normalised-cut parcellation (K = 200) is built for each tertile in
anxiety-fmri-parcellations: `src/templates/atlases/group_gm_{low,mid,high}.nii.gz`.
`run_group_ensemble` computes features for every subject under all three
parcellations, trains one model per parcellation on the same training
subjects and target (STAI-T ≥ 36), and averages the three per-subject test
probabilities (soft vote, threshold 0.5), so no subject's label decides
which parcellation is applied to it.

**Stochastic parcellations (GSP and ABIDE, ROI-ROI).** 30 Poisson-disk
parcellations at each of 50, 100, 150 and 200 ROIs
(`src/templates/atlases/SP_v2_<scale>_<sample>.nii.gz`, 120 files). One MLP
is trained per parcellation, and the 30 per-subject test probabilities at
each scale are averaged (soft vote, threshold 0.5).

```bash
# Individual and group parcellations (GSP), both feature types
for conn in roi_roi voxel; do
  python run_experiments.py run_custom_atlases --connectivity $conn   # individual
  python run_experiments.py run_group_ensemble --connectivity $conn   # 3 group models + soft vote
done

# Stochastic parcellations, GSP and ABIDE
for dataset in gsp abide; do
  python src/compute_features.py --dataset $dataset --features roi_roi --workers 8 \
      --atlases $(cd src/templates/atlases && ls SP_v2_*.nii.gz | sed 's/.nii.gz//')
  python run_experiments.py run_sp_atlases --dataset $dataset --connectivity roi_roi
done
```

`run_group_ensemble` writes the three component runs and
`results/ch6/gsp_classification_<conn>_group_gm_ensemble/`
(`--upload` also writes the features to the bucket and deletes each
subject's local BOLD from `data/cache/gsp/fmri/` once all of its features
are uploaded). `run_sp_atlases`
writes the 120 runs to `results/ch6/SP_classification/<dataset>/`, per-scale
statistics over the samples to `_summary/`, and the per-scale soft-vote
ensembles to
`_ensemble/<dataset>_classification_roi_roi_SP_v2_<scale>_ensemble/`.

The stochastic parcellations are generated from the grey-matter mask by
`src/generate_sp_masks.py`, using the `pds` Poisson-disk sampling package
(Schirmer), which needs graph-tool from conda-forge. The conda environment
and the pds installation steps are in `environment-pds.yml`:

```bash
conda env create -f environment-pds.yml && conda activate pds
# then install pds as described in environment-pds.yml
python src/generate_sp_masks.py --scales 50 100 150 200 --n-samples 30 --out-dir /tmp/sp_masks
```

Each mask is seeded with its sample index; existing files in `--out-dir`
are skipped.

### Chapter 7: fMRI arms of the multimodal fusion

The Ch7 fusion in
[anxiety-multimodal-fusion](https://github.com/ethancaldeira/anxiety-multimodal-fusion)
uses the voxel 3D CNN with AAL and Schaefer-200. For GSP these are the Ch5
runs `results/ch5/gsp_classification_voxel_{aal_roi_atlas,schaefer_200_7n_yeo}/`
(same split as the GSP structural arm). For ABIDE the two models are trained
on the split shared with the ABIDE structural arm. The script reads the
ABIDE voxel features for these two atlases from
`data/cache/abide/voxel/` (or `FMRI_S3_BUCKET`); `run_standard_atlases`
removes its caches after each cell, so compute them first unless they are
in the bucket:

```bash
python src/compute_features.py --dataset abide --atlases aal_roi_atlas schaefer_200_7n_yeo --features voxel --workers 8
./run_abide_sharedsplit.sh
```

This trains the ABIDE voxel models for AAL and Schaefer-200 on
`data/abide/abide_split_seed42.json` (503 / 116 / 155), writing to
`results/ch7/abide_voxel_{aal,schaefer}_sharedsplit/`.

---

## Outputs

```
results/ch<N>/<dataset>_classification_<conn>_<atlas>/
├── config.json                  # every Config value used for the run
├── best.keras                   # lowest-validation-loss checkpoint
├── training_history.json
├── predictions_subject.csv      # subject_id, true, prob, pred
├── metrics_subject.json         # accuracy, precision, recall, ROC AUC
└── plots/                       # training curves, confusion matrix, ROC, PR,
                                 # probability histogram, calibration, borderline
```

All metrics and plots are derived from `predictions_subject.csv`:

```bash
python src/evaluate.py run results/ch5/gsp_classification_voxel_cc200_roi_atlas
python src/evaluate.py table 'results/ch5/*' --out results/ch5/_compare/table.csv
python src/evaluate.py compare 'results/ch5/*cc200*'           # ROC overlay
python src/plot_architectures.py                               # model diagrams → results/figures/
```

---

## Data use and acknowledgements

GSP data are used under the GSP Open Access Data Use Terms and, for the
STAI-T scores, the GSP restricted data use terms. Data were
provided by the Brain Genomics Superstruct Project of Harvard University and
the Massachusetts General Hospital (Principal Investigators: Randy Buckner,
Joshua Roffman, and Jordan Smoller), with support from the Center for Brain
Science Neuroinformatics Research Group, the Athinoula A. Martinos Center for
Biomedical Imaging, and the Center for Human Genetic Research. 20 individual
investigators at Harvard and MGH generously contributed data to the overall
project.

ABIDE I data are used under the ABIDE data usage terms, via the
Preprocessed Connectomes Project.

- Holmes, A. J. et al. (2015). Brain Genomics Superstruct Project initial
  data release with structural, functional, and behavioral measures.
  *Scientific Data*, 2, 150031.
- Di Martino, A. et al. (2014). The autism brain imaging data exchange:
  towards a large-scale evaluation of the intrinsic brain architecture in
  autism. *Molecular Psychiatry*, 19(6), 659–667.
- Craddock, C. et al. (2013). The Neuro Bureau Preprocessing Initiative:
  open sharing of preprocessed neuroimaging data and derivatives.
  *Frontiers in Neuroinformatics*.
- Khosla, M. et al. (2019). Ensemble learning with 3D convolutional neural
  networks for functional connectome-based prediction. *NeuroImage*, 199,
  651–662.

Atlases: AAL (Tzourio-Mazoyer et al., 2002); CC200 and CC400 (Craddock et
al., 2012); Dosenbach 160 (Dosenbach et al., 2010); Eickhoff-Zilles
(Eickhoff et al., 2005); Harvard-Oxford (FSL); Talairach-Tournoux
(Lancaster et al., 2000); Schaefer 200 and 400, 7 networks (Schaefer et al.,
2018). Sources and licences of every atlas and mask file are listed in
[`src/templates/atlases/README.md`](src/templates/atlases/README.md).

## License

Code is released under the GNU General Public License v3.0 or later (see
[`LICENSE`](LICENSE)).
`preprocessing/cpac_run_patched.py` is a modified C-PAC file and remains
under the GNU LGPL-3.0 (`preprocessing/COPYING.LESSER`,
`preprocessing/COPYING`). The atlas and mask files keep the terms of their
sources (`src/templates/atlases/README.md`). The ABIDE subject lists in
`data/abide/` are covered by the ABIDE data use terms.
