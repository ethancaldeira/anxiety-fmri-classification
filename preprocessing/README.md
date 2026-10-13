# GSP fMRI preprocessing

Resting-state fMRI preprocessing for the GSP with [C-PAC](https://fcp-indi.github.io/)
1.8.7, using its `preproc` preconfig, run on AWS Batch (one job per subject,
c5.4xlarge instances). Input is the GSP in BIDS format on S3 (first
resting-state run per subject); output is written back to S3.

The C-PAC image is the 1.8.7 release, pinned by digest in both
`Dockerfile.cpac-patched` and `run_single.sh`:

```
fcpindi/c-pac:release-v1.8.7@sha256:590200a9f6b87e4c67a7b19627f332d54fab94a54c0fc5ed709d6fa31017569f
```

(C-PAC writes its version as `1.8.7.dev1` into the pipeline config and
output headers.)

## Files

| File | Description |
|------|-------------|
| `Dockerfile.cpac-patched` | C-PAC image used on AWS Batch (two file replacements, see below) |
| `cpac_run_patched.py` | C-PAC `run.py` without the `awsCredentialsFile` assignment, which is not part of the 1.8.7 pipeline-config schema |
| `setup_batch.py` | One-time AWS Batch setup: IAM roles, compute environment, job queue, job definition |
| `submit_jobs.py` | Submit one Batch job per subject |
| `run_single.sh` | Run one subject locally with Docker |
| `COPYING`, `COPYING.LESSER` | GNU GPL-3.0 and LGPL-3.0 texts (licence of `cpac_run_patched.py`) |

## Configuration

All scripts read their S3 locations and AWS settings from environment
variables:

| Variable | Used by | Default |
|---|---|---|
| `S3_BUCKET` | `setup_batch.py` (IAM policy of the job role) | required |
| `S3_INPUT` | `submit_jobs.py`, `run_single.sh` | required, e.g. `s3://<bucket>/GSP_BIDS` |
| `S3_OUTPUT` | `submit_jobs.py`, `run_single.sh` | required, e.g. `s3://<bucket>/GSP_PREPROC` |
| `AWS_REGION` | `setup_batch.py`, `submit_jobs.py` | `eu-west-2` |
| `JOB_QUEUE` | `setup_batch.py`, `submit_jobs.py` | `cpac_job_queue` |
| `CPAC_IMAGE` | `run_single.sh` | the pinned image above |

Resource sizes (vCPUs, memory, ANTs threads) are constants at the top of
each script. To use the outputs in this repository, write them to
`<FMRI_S3_BUCKET>/GSP_PREPROC` (see the main README).

## GSP in BIDS

C-PAC reads the GSP from a BIDS directory on S3. Each subject's first
session is mapped from the GSP Open Access release as follows:

| GSP release | BIDS (`$S3_INPUT`) |
|---|---|
| `SubNNNN_Ses1/SubNNNN_Ses1_Scan_01_ANAT1.nii.gz` | `sub-NNNN/ses-1/anat/sub-NNNN_ses-1_T1w.nii.gz` |
| `SubNNNN_Ses1/SubNNNN_Ses1_Scan_02_BOLD1.nii.gz` | `sub-NNNN/ses-1/func/sub-NNNN_ses-1_task-rest_bold.nii.gz` |

plus a top-level `dataset_description.json` (`"BIDSVersion": "1.6.0"`,
`"DatasetType": "raw"`) and a top-level `task-rest_bold.json`:

```json
{
  "TaskName": "rest",
  "RepetitionTime": 3.0,
  "MagneticFieldStrength": 3,
  "Manufacturer": "Siemens",
  "ManufacturersModelName": "TrioTim",
  "EchoTime": 0.03,
  "FlipAngle": 85,
  "PhaseEncodingDirection": "j-"
}
```

Only the first resting-state run (`BOLD1`) is used.

## Credentials patch for AWS Batch

On Batch, C-PAC has no credentials file, and the `indi_aws` package then
falls back to anonymous S3 access. The image therefore replaces
`indi_aws/fetch_creds.py` with a version whose `return_bucket()`, when no
credentials file is given, connects through a plain `boto3.session.Session()`
(`session.resource('s3', use_ssl=True)`), i.e. boto3's default credential
chain (the Batch job's IAM role), instead of an unsigned client. That file
is not distributed here; create `indi_aws_fetch_creds_patched.py` in this
directory from the image's own
`/usr/share/fsl/6.0/lib/python3.10/site-packages/indi_aws/fetch_creds.py`
with that change before building. `run_single.sh` runs the pinned C-PAC
image directly (with `cpac_run_patched.py` mounted as `/code/run.py`) and
passes a temporary credentials file built from the local AWS CLI
configuration, so it does not need this patch.

## Running

### 1. Build and push the image

```bash
export AWS_REGION=eu-west-2
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/cpac-patched:latest"

aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build -f Dockerfile.cpac-patched -t cpac-patched .
docker tag cpac-patched:latest "$ECR_URI"
docker push "$ECR_URI"
```

### 2. Set up AWS Batch (once)

```bash
S3_BUCKET=<bucket> python setup_batch.py
```

### 3. Submit jobs

`--subject-list` takes a plain text file with one ID per line
(`SubNNNN_S1` or `sub-NNNN`), not the `data/subjects.txt` CSV. To list the
879 participants of the participants TSV (see the main README):

```bash
python -c "import pandas as pd; pd.read_csv('/path/to/GSP_participants.tsv', sep='\t').participant_id.to_csv('gsp_ids.txt', index=False, header=False)"
```

```bash
export S3_INPUT=s3://<bucket>/GSP_BIDS S3_OUTPUT=s3://<bucket>/GSP_PREPROC
python submit_jobs.py --subject-list gsp_ids.txt       # one ID per line (SubNNNN_S1 or sub-NNNN)
python submit_jobs.py --subjects sub-NNNN sub-NNNN
python submit_jobs.py --subject-list gsp_ids.txt --dry-run
```

### Single subject locally

```bash
S3_INPUT=s3://<bucket>/GSP_BIDS S3_OUTPUT=s3://<bucket>/GSP_PREPROC ./run_single.sh sub-NNNN
```

## Output

Per subject, under `<S3_OUTPUT>/output/pipeline_cpac_preproc/sub-NNNN/ses-1/`:

- `anat/`: skull-stripped T1w, tissue segmentation, registration transforms
- `func/`: preprocessed BOLD in MNI152NLin6ASym space (3 mm), motion
  parameters and confounds

The features in this repository use
`func/*space-MNI152NLin6ASym_reg-1_desc-preproc_bold.nii.gz`. Batch jobs can
exit with code 1 after completing successfully, so check for these files
on S3 rather than relying on the job status. For two GSP subjects
(Sub0105_S1 and Sub1382_S1) C-PAC functional preprocessing stopped at
nuisance regression and wrote no preprocessed BOLD; the analysed cohort is
the remaining 877 subjects (see `scripts/make_gsp_inputs.py`).
