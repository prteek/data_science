#%%
import sys
import os
import awswrangler as wr
from datetime import datetime
import pandas as pd
import boto3

wr.engine.set('python')

boto3_session = boto3.Session(region_name='eu-west-1')

LAND_REGISTRY_PATH = "s3://ds-dev-bkt/raw_land_registry_transactions"

INPUT_SCHEMA = {
    "id": str,
    "price": int,
    "purchase_date": str,
    "post_code": str,
    "property_type": str,
    "is_new_build": str,
    "tenure": str,
    "paon": str,
    "saon": str,
    "street": str,
    "locality": str,
    "post_town": str,
    "district": str,
    "county": str,
    "ppd_cat": str,
    "record_type": str
}

OUTPUT_SCHEMA = {
    "buy_transaction_id": "string",
    "buy_transaction_dt": "date",
    "buy_price_gbp": "int",
    "is_standard_transaction_type": "int",
    "buy_transaction_type_code": "string",
    "postcode": "string",
    "primary_addressable_object_name": "string",
    "secondary_addressable_object_name": "string",
    "street": "string",
    "locality": "string",
    "post_town": "string",
    "district": "string",
    "county": "string",
    "property_type": "string",
    "property_type_detail": "string",
    "tenure": "string",
    "is_new_build": "int",
    "record_type": "string",
    "etl_dt_id": "int"
}

INPUT_COLUMNS = list(INPUT_SCHEMA.keys())

OUTPUT_COLUMNS = list(OUTPUT_SCHEMA.keys())


def lowercase_string_cols(df):
    """Convert all string columns of the DataFrame to lowercase"""
    string_columns = df.select_dtypes(include="object").columns

    for column in string_columns:
        df[column] = df[column].str.lower()

    return df

def create_buy_transaction_id(df):
    df = df.assign(buy_transaction_id = lambda x: x["id"])
    return df


def create_buy_transaction_dt(df):
    df = df.assign(buy_transaction_dt = lambda x: pd.to_datetime(x.purchase_date).dt.date)
    return df


def create_buy_price_gbp(df):
    df = df.assign(buy_price_gbp = lambda x: x["price"])
    return df


def create_is_standard_transaction_type(df):
    ppd_cat_map = {
    "a": 1,
    "b": 0
    }

    df["is_standard_transaction_type"] = df["ppd_cat"].map(ppd_cat_map)
    return df


def create_buy_transaction_type_code(df):
    df = df.assign(buy_transaction_type_code=lambda x: x["ppd_cat"])
    return df


def create_postcode(df):
    df = df.assign(postcode=lambda x: x["post_code"].str.lower())
    return df


def create_primary_addressable_object_name(df):
    df = df.assign(primary_addressable_object_name = lambda x: x["paon"].str.lower())
    return df


def create_secondary_addressable_object_name(df):
    df = df.assign(secondary_addressable_object_name=lambda x: x["saon"].str.lower())
    return df


def create_property_type(df):
    type_map = {
    "d": "house",
    "s": "house",
    "t": "house",
    "f": "flat",
    "o": "other"
    }

    df["property_type"] = df["property_type"].map(type_map)
    return df


def create_property_type_detail(df):
    type_detail_map = {
        "d": "detached house",
        "s": "semi detached house",
        "t": "terraced house",
        "f": "flat/maisonette",
        "o": "other"
    }

    df["property_type_detail"] = df["property_type"].map(type_detail_map)
    return df


def create_tenure(df):
    tenure_map = {
        "f": "freehold",
        "l": "leasehold"
    }

    df["tenure"] = df["tenure"].map(tenure_map)
    return df


def create_is_new_build(df):
    is_new_build_map = {
        "y": 1,
        "n": 0
    }

    df["is_new_build"] = df["is_new_build"].map(is_new_build_map)
    return df


def create_record_type(df):
    record_type_map = {
        "a": "addition",
        "c": "change",
        "d": "delete"
    }

    df["record_type"] = df["record_type"].map(record_type_map)
    return df


def create_etl_dt(df):
    df["etl_dt_id"] = int(datetime.today().strftime("%Y%m%d"))
    return df


#%%

if __name__ == '__main__':
    #%%

    filepath = os.path.join(LAND_REGISTRY_PATH, 'pp-complete.csv')
    df_raw = pd.read_csv(filepath, header=None)

    #%%
    df = (df_raw
            .rename({i:INPUT_COLUMNS[i] for i in df_raw.columns}, axis=1)
            .astype(INPUT_SCHEMA)
        )

    df_processed = (
        df.pipe(lowercase_string_cols)
        .pipe(create_buy_transaction_id)
        .pipe(create_buy_transaction_dt)
        .pipe(create_buy_price_gbp)
        .pipe(create_is_standard_transaction_type)
        .pipe(create_buy_transaction_type_code)
        .pipe(create_postcode)
        .pipe(create_primary_addressable_object_name)
        .pipe(create_secondary_addressable_object_name)
        .pipe(create_property_type_detail)
        .pipe(create_property_type)
        .pipe(create_tenure)
        .pipe(create_is_new_build)
        .pipe(create_record_type)
        .pipe(create_etl_dt)
        .get(OUTPUT_SCHEMA.keys())
    )


    #%%

    path = os.path.join(LAND_REGISTRY_PATH, 'land_registry_transactions')

    wr.s3.to_csv(
        df_processed
        , path=path
        , index=False
        , boto3_session=boto3_session
        , dataset=True
        , mode='overwrite'
        , dtype=OUTPUT_SCHEMA
        , schema_evolution=False
        , database='default'
        , table='land_registry_transactions'
        , partition_cols=['post_town']
    )

    #%%