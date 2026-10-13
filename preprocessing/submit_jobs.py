#!/usr/bin/env python3
"""
submit_jobs.py
Submit CPAC preprocessing jobs to AWS Batch.

Environment variables:
    S3_INPUT     BIDS input   (default s3://<bucket>/GSP_BIDS)
    S3_OUTPUT    C-PAC output (default s3://<bucket>/GSP_PREPROC)
    AWS_REGION   AWS region (default eu-west-2)
    JOB_QUEUE    Batch job queue (default cpac_job_queue)

Usage:
    # Submit specific subjects
    python submit_jobs.py --subjects sub-NNNN sub-NNNN

    # Submit all subjects from a text file (one per line)
    python submit_jobs.py --subject-list gsp_ids.txt

    # Dry run (print jobs without submitting)
    python submit_jobs.py --subject-list gsp_ids.txt --dry-run
"""

import argparse
import os
import sys

import boto3

# ── Config ────────────────────────────────────────────────────
AWS_REGION          = os.environ.get("AWS_REGION", "eu-west-2")
JOB_QUEUE           = os.environ.get("JOB_QUEUE", "cpac_job_queue")
JOB_DEF             = "cpac_job_def"
S3_INPUT            = os.environ.get("S3_INPUT", "s3://<bucket>/GSP_BIDS")
S3_OUTPUT           = os.environ.get("S3_OUTPUT", "s3://<bucket>/GSP_PREPROC")
N_CPUS              = 8
MEM_GB              = 20
ANTS_THREADS        = 4
# ─────────────────────────────────────────────────────────────


def to_bids_id(raw_id):
    """Convert a GSP participant ID to BIDS: 'SubNNNN_S1' -> 'sub-NNNN'.
    Already-converted IDs (e.g. 'sub-NNNN') are passed through unchanged."""
    if raw_id.startswith("sub-"):
        return raw_id
    return "sub-" + raw_id.replace("Sub", "").replace("_S1", "")


def build_command(bids_id):
    """
    Build the C-PAC argument list for a single subject.

    run-with-freesurfer.sh (the container ENTRYPOINT) simply calls
    /code/run.py "$@", so we pass C-PAC arguments directly.

    The image is the ECR build of Dockerfile.cpac-patched:
      - run.py: no awsCredentialsFile assignment (not part of the
        C-PAC 1.8.7 pipeline-config schema)
      - indi_aws fetch_creds.py: uses the job's IAM role via boto3's
        default credential chain when no credentials file is supplied.
    No --aws-input-creds flag is passed.
    """
    return [
        S3_INPUT, S3_OUTPUT, "participant",
        "--participant-label", bids_id,
        "--preconfig",         "preproc",
        "--n-cpus",            str(N_CPUS),
        "--mem-gb",            str(MEM_GB),
        "--num-ants-threads",  str(ANTS_THREADS),
        "--skip-bids-validator",
        "--tracking-opt-out",
    ]


def submit_job(batch, raw_id, dry_run=False):
    """Submit one job to AWS Batch."""
    bids_id  = to_bids_id(raw_id)
    job_name = f"cpac-preproc-{bids_id}"
    command  = build_command(bids_id)

    if dry_run:
        print(f"  [DRY RUN] {job_name}")
        print(f"            {' '.join(command)}")
        return None

    resp = batch.submit_job(
        jobName=job_name,
        jobQueue=JOB_QUEUE,
        jobDefinition=JOB_DEF,
        containerOverrides={"command": command},
    )
    job_id = resp["jobId"]
    print(f"  Submitted {job_name} → {job_id}")
    return job_id


def load_subject_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit CPAC jobs to AWS Batch")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subjects",     nargs="+", help="Subject IDs to submit (SubNNNN_S1 or sub-NNNN)")
    group.add_argument("--subject-list", help="Path to text file with one subject ID per line")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print jobs without submitting"
    )
    args = parser.parse_args()

    subjects = args.subjects if args.subjects else load_subject_list(args.subject_list)
    if "<bucket>" in S3_INPUT + S3_OUTPUT:
        sys.exit("ERROR: set S3_INPUT and S3_OUTPUT (e.g. s3://<bucket>/GSP_BIDS "
                 "and s3://<bucket>/GSP_PREPROC).")
    batch = boto3.client("batch", region_name=AWS_REGION)

    print(f"\n=== CPAC Batch Submission ===")
    print("Pipeline  : C-PAC preconfig 'preproc'")
    print(f"Input     : {S3_INPUT}")
    print(f"Output    : {S3_OUTPUT}")
    print(f"Subjects  : {len(subjects)}")
    print(f"Dry run   : {args.dry_run}")
    print(f"{'=' * 30}\n")

    job_ids = []
    for subject in subjects:
        job_id = submit_job(batch, subject, dry_run=args.dry_run)
        if job_id:
            job_ids.append(job_id)

    if not args.dry_run:
        print(f"\nSubmitted {len(job_ids)} jobs to queue '{JOB_QUEUE}'")
        print(f"Monitor at: https://console.aws.amazon.com/batch/home?region={AWS_REGION}")
