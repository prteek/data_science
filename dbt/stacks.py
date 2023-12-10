import os
from aws_cdk import (
    Duration,
    Stack,
    aws_iam,
    aws_lambda,
)
from constructs import Construct


lambda_exec_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name(
    "service-role/AWSLambdaBasicExecutionRole"
)

sagemaker_full_access_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name(
    "AmazonSageMakerFullAccess"
)


class RunDbt(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        with open("run_dbt_lambda.py", encoding="utf8") as fp:
            handler_code = fp.read()

        role = aws_iam.Role(
            self,
            "run_dbt_role",
            assumed_by=aws_iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                lambda_exec_policy,
                sagemaker_full_access_policy,
            ],
        )

        pygithub_layer = aws_lambda.LayerVersion.from_layer_version_arn(
            self,
            "pygithub_layer",
            layer_version_arn="arn:aws:lambda:eu-west-1:770693421928:layer:Klayers-p39-PyGithub:1",
        )

        _ = aws_lambda.Function(
            self,
            id="run_dbt",
            function_name="run_dbt",
            code=aws_lambda.InlineCode(handler_code),
            handler="index.run_dbt_handler",
            timeout=Duration.seconds(60),
            memory_size=256,
            runtime=aws_lambda.Runtime.PYTHON_3_9,
            role=role,
            environment={
                "IMAGE_URI": os.environ["IMAGE_URI"],
                "SAGEMAKER_EXECUTION_ROLE": os.environ["SAGEMAKER_EXECUTION_ROLE"],
                "GITHUB_USER": os.environ["GITHUB_USER"],
                "GITHUB_TOKEN": os.environ["GITHUB_TOKEN"],
            },
            layers=[pygithub_layer],
        )
