#!/bin/bash
# ============================================================
# run_single.sh
# Run one subject through CPAC locally via Docker.
#
# Uses the C-PAC 'preproc' preconfig.
#
# Usage:
#   S3_INPUT=s3://<bucket>/GSP_BIDS S3_OUTPUT=s3://<bucket>/GSP_PREPROC \
#       ./run_single.sh <subject_id>
#
# Example:
#   ./run_single.sh sub-NNNN
#
# Environment variables:
#   S3_INPUT    BIDS input   (required)
#   S3_OUTPUT   C-PAC output (required)
#   CPAC_IMAGE  C-PAC Docker image (default: the pinned 1.8.7 release)
# ============================================================

set -euo pipefail

# ── Args ─────────────────────────────────────────────────────
SUBJECT_ID="${1:?Usage: ./run_single.sh <subject_id>}"

# ── Config ───────────────────────────────────────────────────
S3_INPUT="${S3_INPUT:?set S3_INPUT, e.g. s3://<bucket>/GSP_BIDS}"
S3_OUTPUT="${S3_OUTPUT:?set S3_OUTPUT, e.g. s3://<bucket>/GSP_PREPROC}"
CPAC_IMAGE="${CPAC_IMAGE:-fcpindi/c-pac:release-v1.8.7@sha256:590200a9f6b87e4c67a7b19627f332d54fab94a54c0fc5ed709d6fa31017569f}"
N_CPUS=8
MEM_GB=20
ANTS_THREADS=4

# ── Run ──────────────────────────────────────────────────────
echo "=========================================="
echo "Subject  : ${SUBJECT_ID}"
echo "Pipeline : preproc (C-PAC preconfig)"
echo "Started  : $(date)"
echo "=========================================="

START=$(date +%s)

# Write a temporary credentials file for C-PAC (the "env" option requires ECS,
# so we use a YAML creds file instead for standard key-based auth).
CREDS_FILE="$(mktemp /tmp/cpac_creds_XXXXXX.yml)"
cat > "${CREDS_FILE}" <<EOF
AWSAccessKeyId=$(aws configure get aws_access_key_id)
AWSSecretAccessKey=$(aws configure get aws_secret_access_key)
EOF
trap 'rm -f "${CREDS_FILE}"' EXIT
PATCHED_RUN="$(dirname "$0")/cpac_run_patched.py"
chmod +x "${PATCHED_RUN}"

docker run --rm \
    --env AWS_DEFAULT_REGION="$(aws configure get region)" \
    --volume "${CREDS_FILE}:/tmp/aws_creds.yml:ro" \
    --volume "${PATCHED_RUN}:/code/run.py:ro" \
    "${CPAC_IMAGE}" \
    "${S3_INPUT}" \
    "${S3_OUTPUT}" \
    participant \
    --participant-label "${SUBJECT_ID}" \
    --preconfig preproc \
    --n-cpus "${N_CPUS}" \
    --mem-gb "${MEM_GB}" \
    --num-ants-threads "${ANTS_THREADS}" \
    --aws-input-creds /tmp/aws_creds.yml \
    --aws-output-creds /tmp/aws_creds.yml \
    --skip-bids-validator \
    --tracking-opt-out

END=$(date +%s)
ELAPSED=$(( END - START ))

echo "=========================================="
echo "Finished : $(date)"
echo "Duration : $(( ELAPSED / 60 )) min $(( ELAPSED % 60 )) sec"
echo "=========================================="
