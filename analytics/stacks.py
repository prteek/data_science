import os
from aws_cdk import (
    Stack,
    aws_events,
    aws_events_targets,
    aws_lambda,
)
from constructs import Construct
import configparser

config = configparser.ConfigParser()
config.read("deployment_config.txt")

model_projects = [
    i for i in os.listdir("models") if os.path.isdir(os.path.join("models", i))
]

ENV = os.environ["ENV"]
branch_map = {"dev": "dev/analytics", "prod": "production"}

sections = config.sections()


class ModelTriggers(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        lambdaFn = aws_lambda.Function.from_function_name(
            self,
            id="run_dbt",
            function_name="run_dbt",
        )

        for section in sections:
            _ = aws_events.Rule(
                self,
                id=config.get(section, "trigger_name"),
                schedule=aws_events.Schedule.expression(
                    expression=config.get(section, "cron")
                ),
                targets=[
                    aws_events_targets.LambdaFunction(
                        lambdaFn,
                        event=aws_events.RuleTargetInput.from_object(
                            {
                                "model": config.get(section, "model"),
                                "branch": branch_map[ENV],
                            }
                        ),
                    )
                ],
                enabled=True,
            )
