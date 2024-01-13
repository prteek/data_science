import awswrangler as wr
from datetime import datetime
import pandas as pd

QUERY_LAST_DATA_POINTS = """
select timestamp,
type,
value
from default.sugar_readings
where timestamp in (select max(timestamp) from default.sugar_readings group by type)

union
select timestamp,
type,
value
from default.insulin_intake
where timestamp in (select max(timestamp) from default.insulin_intake group by type)

union
select timestamp,
'meal' as type,
value
from default.meal 
where timestamp in (select max(timestamp) from default.meal)

order by timestamp desc
"""


def lambda_handler(event, context):
    print(event)
    try:
        df = (wr
              .athena
              .read_sql_query(QUERY_LAST_DATA_POINTS,
                             database="default")
              )

        return df.to_json(orient='records')
    except Exception as e:
        print(e)
        return {'body': "error, check lambda logs",
                'status': 500
                }
