import awswrangler as wr
from datetime import datetime
import pandas as pd


def lambda_handler(event, context):
    print(event)
    try:
        table = event["table"]
        df = pd.DataFrame.from_dict([event]).drop("table", axis="columns")
        filepath = f"s3://ds-dev-bkt/{table}"
        wr.s3.to_csv(df,
                     filepath,
                     index=False,
                     dataset=True,
                     mode="overwrite_partitions",
                     database="default",
                     partition_cols=["timestamp"],
                     table=table,
                     schema_evolution=True
                     )

        return {
            'body': f"{filepath}/{event['timestamp']}",
            'status': 200
        }
    except Exception as e:
        print(e)
        return {'body': "error, check lambda logs",
                'status': 500
                }
