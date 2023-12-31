import awswrangler as wr
from datetime import datetime
import pandas as pd


def lambda_handler(event, context):
    try:
        table = event["table"]
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        df = pd.DataFrame.from_dict([event]).drop("table", axis="columns").assign(timestamp=timestamp)
        filepath = f"s3://ds-dev-bkt/{table}/"
        wr.s3.to_csv(df,
                     filepath,
                     index=False,
                     dataset=True,
                     mode="overwrite_partitions",
                     database="default",
                     table=table,
                     schema_evolution=True
                     )

        return {
            'body': filepath,
            'status': 200
        }
    except Exception as e:
        print(e)
        return {'body': "error, check lambda logs",
                'status': 500
                }
