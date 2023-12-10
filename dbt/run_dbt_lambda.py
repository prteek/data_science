import boto3
from datetime import datetime
import os
from github import Github


GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

g = Github(login_or_token=GITHUB_TOKEN)
repo = g.get_repo("PropertyLift/ds.bricklane")

client = boto3.client("sagemaker")


def run_dbt_handler(event, context=None):
    """Lambda handler to run dbt on Sagemaker docker container"""

    if "model" in event:
        model = event["model"]
        job_name = (
            f"{model.replace('_', '-')}T{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )
    else:
        model = "./models"
        job_name = f"all-modelsT{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    branch = event["branch"]

    latest_commit_hash = repo.get_branch(branch).commit.sha

    container_arguments = [
        branch,
        latest_commit_hash,
        model,
    ]  # list of strings

    # Create sagemaker processing job
    response = client.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.t3.medium",
                "VolumeSizeInGB": 30,
            }
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        AppSpecification={
            "ImageUri": os.environ["IMAGE_URI"],
            "ContainerEntrypoint": ["/bin/bash", "./run_dbt.sh"],
            "ContainerArguments": container_arguments,
        },
        RoleArn=os.environ["SAGEMAKER_EXECUTION_ROLE"],
        Tags=[
            {"Key": "branch", "Value": branch},
            {"Key": "commit", "Value": latest_commit_hash},
            {"Key": "model", "Value": model},
            {"Key": "type", "Value": "dbt"},
        ],
    )

    return response
