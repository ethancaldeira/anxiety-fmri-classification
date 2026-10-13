#!/usr/bin/env python3
"""
setup_batch.py
One-time setup of AWS Batch infrastructure for CPAC preprocessing.
Run this once before submitting any jobs.

Creates:
  - IAM job role         (read/write on the S3 bucket + CloudWatch logs)
  - ecsInstanceRole      (EC2 instance profile for Batch compute nodes)
  - Compute environment  (managed, c5.4xlarge, on-demand)
  - Job queue
  - Job definition       (the patched C-PAC image in ECR, built from
                          Dockerfile.cpac-patched)

Note: AWSBatchServiceRole is not required — Batch uses the service-linked
      role AWSServiceRoleForBatch automatically.

Environment variables:
    S3_BUCKET    bucket holding the BIDS input and the C-PAC output
                 (the job role gets read/write access to it)
    AWS_REGION   AWS region for Batch, ECR and CloudWatch (default eu-west-2)
    JOB_QUEUE    Batch job queue name (default cpac_job_queue)

Usage:
    S3_BUCKET=<bucket> python setup_batch.py
"""

import json
import os
import sys
import time
import boto3
from botocore.exceptions import ClientError

# ── Config ────────────────────────────────────────────────────
S3_BUCKET           = os.environ.get("S3_BUCKET", "")
AWS_REGION          = os.environ.get("AWS_REGION", "eu-west-2")
JOB_ROLE_NAME       = "cpac_batch_job_role"
INSTANCE_ROLE_NAME  = "ecsInstanceRole"
COMPUTE_ENV_NAME    = "cpac_compute_env"
JOB_QUEUE_NAME      = os.environ.get("JOB_QUEUE", "cpac_job_queue")
JOB_DEF_NAME        = "cpac_job_def"
ECR_IMAGE_NAME      = "cpac-patched:latest"   # account ID resolved at runtime via STS
MAX_VCPUS           = 8792          # upper bound for the compute environment (~10 vCPU per GSP subject)
INSTANCE_TYPES      = ["c5.4xlarge"] # 16 vCPU, 32 GB per instance
# ─────────────────────────────────────────────────────────────

iam    = boto3.client("iam",   region_name=AWS_REGION)
batch  = boto3.client("batch", region_name=AWS_REGION)
ec2    = boto3.client("ec2",   region_name=AWS_REGION)


def get_account_id():
    sts = boto3.client("sts", region_name=AWS_REGION)
    return sts.get_caller_identity()["Account"]


def create_job_role():
    """IAM role assumed by each Batch job container."""
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{S3_BUCKET}",
                    f"arn:aws:s3:::{S3_BUCKET}/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "*"
            }
        ]
    }

    try:
        role = iam.create_role(
            RoleName=JOB_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="CPAC Batch job role - S3 + CloudWatch"
        )
        iam.put_role_policy(
            RoleName=JOB_ROLE_NAME,
            PolicyName="cpac_s3_cloudwatch",
            PolicyDocument=json.dumps(policy)
        )
        arn = role["Role"]["Arn"]
        print(f"  Created job role: {arn}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            arn = iam.get_role(RoleName=JOB_ROLE_NAME)["Role"]["Arn"]
            print(f"  Job role already exists: {arn}")
        else:
            raise
    return arn


def create_instance_role():
    """
    ecsInstanceRole — assumed by EC2 instances in the Batch compute environment.
    Requires AmazonEC2ContainerServiceforEC2Role managed policy.
    """
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }

    # Create the role
    try:
        iam.create_role(
            RoleName=INSTANCE_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="ECS instance role for AWS Batch compute nodes"
        )
        print(f"  Created instance role: {INSTANCE_ROLE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            print(f"  Instance role already exists: {INSTANCE_ROLE_NAME}")
        else:
            raise

    # Attach the required managed policy
    try:
        iam.attach_role_policy(
            RoleName=INSTANCE_ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
        )
        print(f"  Attached AmazonEC2ContainerServiceforEC2Role policy")
    except ClientError as e:
        if "already attached" not in str(e).lower():
            raise

    # Create instance profile and add role to it
    try:
        iam.create_instance_profile(InstanceProfileName=INSTANCE_ROLE_NAME)
        print(f"  Created instance profile: {INSTANCE_ROLE_NAME}")
        # Brief pause to allow IAM propagation
        time.sleep(5)
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            print(f"  Instance profile already exists: {INSTANCE_ROLE_NAME}")
        else:
            raise

    try:
        iam.add_role_to_instance_profile(
            InstanceProfileName=INSTANCE_ROLE_NAME,
            RoleName=INSTANCE_ROLE_NAME
        )
        print(f"  Added role to instance profile")
    except ClientError as e:
        if "LimitExceeded" in str(e) or "already" in str(e).lower():
            print(f"  Role already in instance profile")
        else:
            raise

    profile = iam.get_instance_profile(InstanceProfileName=INSTANCE_ROLE_NAME)
    arn = profile["InstanceProfile"]["Arn"]
    print(f"  Instance profile ARN: {arn}")
    return arn


def get_default_vpc_and_subnets():
    """Use the default VPC subnets."""
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpc_id = vpcs["Vpcs"][0]["VpcId"]
    subnets = ec2.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
    )
    subnet_ids = [s["SubnetId"] for s in subnets["Subnets"]]
    print(f"  VPC: {vpc_id}  Subnets: {subnet_ids}")
    return vpc_id, subnet_ids


def get_default_security_group(vpc_id):
    sgs = ec2.describe_security_groups(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "group-name", "Values": ["default"]}
        ]
    )
    sg_id = sgs["SecurityGroups"][0]["GroupId"]
    print(f"  Security group: {sg_id}")
    return [sg_id]


def create_compute_environment(instance_profile_arn, subnet_ids, sg_ids):
    """
    No serviceRole needed — Batch uses AWSServiceRoleForBatch automatically.
    """
    try:
        resp = batch.create_compute_environment(
            computeEnvironmentName=COMPUTE_ENV_NAME,
            type="MANAGED",
            state="ENABLED",
            computeResources={
                "type": "EC2",
                "allocationStrategy": "BEST_FIT_PROGRESSIVE",
                "minvCpus": 0,
                "maxvCpus": MAX_VCPUS,
                "instanceTypes": INSTANCE_TYPES,
                "subnets": subnet_ids,
                "securityGroupIds": sg_ids,
                "instanceRole": instance_profile_arn,
            }
        )
        arn = resp["computeEnvironmentArn"]
        print(f"  Created compute environment: {arn}")
    except ClientError as e:
        if "Object already exists" in str(e) or "already exists" in str(e):
            envs = batch.describe_compute_environments(
                computeEnvironments=[COMPUTE_ENV_NAME]
            )
            arn = envs["computeEnvironments"][0]["computeEnvironmentArn"]
            print(f"  Compute environment already exists: {arn}")
        else:
            raise

    # Wait for compute environment to reach VALID state
    print("  Waiting for compute environment to become VALID...", end="", flush=True)
    for _ in range(30):
        envs = batch.describe_compute_environments(computeEnvironments=[COMPUTE_ENV_NAME])
        status = envs["computeEnvironments"][0]["status"]
        if status == "VALID":
            print(" ready.")
            break
        print(".", end="", flush=True)
        time.sleep(5)
    else:
        raise RuntimeError("Compute environment did not become VALID in time")

    return arn


def create_job_queue(compute_env_arn):
    try:
        resp = batch.create_job_queue(
            jobQueueName=JOB_QUEUE_NAME,
            state="ENABLED",
            priority=1,
            computeEnvironmentOrder=[
                {"order": 1, "computeEnvironment": compute_env_arn}
            ]
        )
        arn = resp["jobQueueArn"]
        print(f"  Created job queue: {arn}")
    except ClientError as e:
        if "already exists" in str(e):
            queues = batch.describe_job_queues(jobQueues=[JOB_QUEUE_NAME])
            arn = queues["jobQueues"][0]["jobQueueArn"]
            print(f"  Job queue already exists: {arn}")
        else:
            raise
    return arn


def create_job_definition(job_role_arn, cpac_image):
    try:
        resp = batch.register_job_definition(
            jobDefinitionName=JOB_DEF_NAME,
            type="container",
            containerProperties={
                "image": cpac_image,
                "vcpus": 16,
                "memory": 30000,        # MB — leaves headroom on c5.4xlarge
                "jobRoleArn": job_role_arn,
                "environment": [],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": "/aws/batch/job",
                        "awslogs-region": AWS_REGION,
                        "awslogs-stream-prefix": "cpac"
                    }
                }
            }
        )
        arn = resp["jobDefinitionArn"]
        print(f"  Registered job definition: {arn}")
    except ClientError as e:
        print(f"  Job definition error: {e}")
        raise
    return arn


if __name__ == "__main__":
    if not S3_BUCKET:
        sys.exit("ERROR: set S3_BUCKET to the bucket holding the BIDS input "
                 "and the C-PAC output.")
    print("\n=== CPAC Batch Setup ===\n")

    account_id = get_account_id()
    cpac_image = f"{account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_IMAGE_NAME}"
    print(f"  ECR image: {cpac_image}\n")

    print("[ 1/5 ] IAM job role...")
    job_role_arn = create_job_role()

    print("[ 2/5 ] ecsInstanceRole...")
    instance_profile_arn = create_instance_role()

    print("[ 3/5 ] VPC + subnets...")
    vpc_id, subnet_ids = get_default_vpc_and_subnets()
    sg_ids = get_default_security_group(vpc_id)

    print("[ 4/5 ] Compute environment + job queue...")
    compute_env_arn = create_compute_environment(
        instance_profile_arn, subnet_ids, sg_ids
    )
    create_job_queue(compute_env_arn)

    print("[ 5/5 ] Job definition...")
    create_job_definition(job_role_arn, cpac_image)

    print("\n=== Setup complete. Ready to submit jobs. ===\n")
