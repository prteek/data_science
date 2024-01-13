# %%
from aws_cdk import (
    CfnOutput,
    App,
    aws_apigateway as apigw,
    Duration,
    Stack,
    aws_iam,
    aws_lambda,
)

lambda_exec_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name(
    "AWSLambdaExecute"
)
s3_full_access_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name(
    "AmazonS3FullAccess"
)
glue_full_access_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name(
    "AWSGlueConsoleFullAccess"
)

athena_full_access_policy = aws_iam.ManagedPolicy.from_aws_managed_policy_name("AmazonAthenaFullAccess")


class SaveJsonLambdaStack(Stack):
    def __init__(self, app: App, id: str, **kwargs) -> None:
        super().__init__(app, id, **kwargs)

        with open("save_json_to_table.py", encoding="utf8") as fp:
            handler_code = fp.read()

        role = aws_iam.Role(
            self,
            "save_json_to_table_role",
            assumed_by=aws_iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                lambda_exec_policy,
                s3_full_access_policy,
                glue_full_access_policy
            ],
        )

        aws_wrangler_layer = aws_lambda.LayerVersion.from_layer_version_arn(
            self,
            "aws_wrangler_layer",
            layer_version_arn="arn:aws:lambda:eu-west-1:336392948345:layer:AWSSDKPandas-Python39:12",
        )

        lambda_function = aws_lambda.Function(
            self,
            id="save_json_to_table",
            function_name="save_json_to_table",
            code=aws_lambda.InlineCode(handler_code),
            handler="index.lambda_handler",
            timeout=Duration.seconds(120),
            memory_size=256,
            runtime=aws_lambda.Runtime.PYTHON_3_9,
            role=role,
            layers=[aws_wrangler_layer],
        )

        # Create an API Gateway with a POST method
        api = apigw.RestApi(
            self, 'post_json_api',
            rest_api_name='post_json_api',
        )

        integration_response = apigw.IntegrationResponse(
            status_code='200',
            response_templates={
                'application/json': '$input.json("$")'
            }
        )

        api_gateway_integration = apigw.LambdaIntegration(lambda_function, proxy=False, request_templates={
                "application/json": '$input.json("$")'
            }, integration_responses=[integration_response])

        post_method = api.root.add_method('POST', api_gateway_integration, api_key_required=True)
        # Add a Method Response
        post_method.add_method_response(status_code='200', response_models={'application/json': apigw.Model.EMPTY_MODEL})

        # Create an API Key
        plan = api.add_usage_plan("UsagePlan",
                                  name="Basic",
                                  throttle=apigw.ThrottleSettings(
                                      rate_limit=60,
                                      burst_limit=100
                                  )
                                  )

        api_key = api.add_api_key("ApiKey")
        plan.add_api_key(api_key)
        plan.add_api_stage(stage=api.deployment_stage)


class GetLastDataPointsLambdaStack(Stack):
    def __init__(self, app: App, id: str, **kwargs) -> None:
        super().__init__(app, id, **kwargs)

        with open("get_last_data_points.py", encoding="utf8") as fp:
            handler_code = fp.read()

        role = aws_iam.Role(
            self,
            "get_last_data_points_role",
            assumed_by=aws_iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                lambda_exec_policy,
                glue_full_access_policy,
                athena_full_access_policy
            ],
        )

        aws_wrangler_layer = aws_lambda.LayerVersion.from_layer_version_arn(
            self,
            "aws_wrangler_layer",
            layer_version_arn="arn:aws:lambda:eu-west-1:336392948345:layer:AWSSDKPandas-Python39:12",
        )

        lambda_function = aws_lambda.Function(
            self,
            id="get_last_data_points",
            function_name="get_last_data_points",
            code=aws_lambda.InlineCode(handler_code),
            handler="index.lambda_handler",
            timeout=Duration.seconds(120),
            memory_size=256,
            runtime=aws_lambda.Runtime.PYTHON_3_9,
            role=role,
            layers=[aws_wrangler_layer],
        )

        # Create an API Gateway with a POST method
        api = apigw.RestApi(
            self, 'get_last_data_points_api',
            rest_api_name='get_last_data_points_api',
        )

        integration_response = apigw.IntegrationResponse(
            status_code='200',
            response_templates={
                'application/json': '$input.json("$")'
            }
        )

        api_gateway_integration = apigw.LambdaIntegration(lambda_function, proxy=False, request_templates={
                "application/json": '$input.json("$")'
            }, integration_responses=[integration_response])

        get_method = api.root.add_method('GET', api_gateway_integration, api_key_required=True)
        # Add a Method Response
        get_method.add_method_response(status_code='200', response_models={'application/json': apigw.Model.EMPTY_MODEL})

        # Create an API Key
        plan = api.add_usage_plan("UsagePlan",
                                  name="Basic",
                                  throttle=apigw.ThrottleSettings(
                                      rate_limit=60,
                                      burst_limit=100
                                  )
                                  )

        api_key = api.add_api_key("ApiKey")
        plan.add_api_key(api_key)
        plan.add_api_stage(stage=api.deployment_stage)

# %%
if __name__ == '__main__':
    # %%
    # Define your stack
    app = App()
    SaveJsonLambdaStack(app, "SaveJsonLambdaStack")
    GetLastDataPointsLambdaStack(app, "GetLastDataPointsLambdaStack")
    app.synth()