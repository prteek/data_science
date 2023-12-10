import os
import aws_cdk as cdk
from stacks import RunDbt

# from git import Repo

ENV = os.environ["ENV"]
app = cdk.App()
env = {"region": "eu-west-1"}

# repo = Repo()
# assert repo.is_dirty() is False, "Repo is dirty. Please commit changes before orchestrating pipeline"

# Can be deployed from any environment
RunDbt(app, "RunDbt", env=env)

app.synth()
