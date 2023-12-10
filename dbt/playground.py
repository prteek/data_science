# %%
import os

from sagemaker.processing import Processor

processor = Processor(
    image_uri=os.environ["IMAGE_URI"],
    role=os.environ["SAGEMAKER_EXECUTION_ROLE"],
    instance_count=1,
    instance_type="local",
    entrypoint=["/bin/bash", "./run_dbt.sh"],
)

processor.run(wait=False, logs=False, arguments=["sample_analytics"])

# %%
