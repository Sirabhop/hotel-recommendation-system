# Databricks notebook source
from configs.config import (dict_main_action_encoder, RANDOM_SEED, ENV_PATH, DATA_PATH, 
                            UC_HOTEL_INFO_PATH, UC_USER_INFO_PATH, 
                            UC_TRAIN_DATA_PATH, UC_RECENT_BEHAVIOR_PATH, 
                            DBFS_PATH, UC_PREDICTION_PATH, MODEL_NAME, HOTEL_FEATURE_TABLE, PLAYLIST_TABLE, 
                            UC_MODEL_PATH, window_spec, MODEL_EXPERIMENT_ID, API_EXPERIMENT_ID)

# Set up variable
dbutils.widgets.text('DYNAMODB_SCOPE', '')
dbutils.widgets.text('DYNAMODB_ACCESS_KEY', '')
dbutils.widgets.text('DYNAMODB_SECRET_KEY', '')
dbutils.widgets.text('EXPERIMENT_ID', '')
dbutils.widgets.text('MODEL_NAME', MODEL_NAME)
dbutils.widgets.text('CATALOG', '')

list_variable = ['DYNAMODB_SCOPE', 'DYNAMODB_ACCESS_KEY', 'DYNAMODB_SECRET_KEY', 'EXPERIMENT_ID', 'MODEL_NAME', 'CATALOG']
dict_variable = {}
for variable in list_variable:
    dict_variable[variable] = dbutils.widgets.get(variable)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.feature_engineering.online_store_spec import AmazonDynamoDBSpec
from databricks.feature_store import FeatureStoreClient

from pyspark.sql.functions import *

from tqdm import tqdm

import pandas as pd
import logging
import boto3
import json
import os
import datetime

# COMMAND ----------

df = spark.sql(f"""WITH base AS (
  SELECT  hotel_code, 
          CONCAT(TRIM(address1), ' ', TRIM(address2), ' ', TRIM(address3), ' ', TRIM(zipcode)) AS address,
          location_name, location_code, 
          hotel_name, 
          city_name, city_code, 
          country_name, country_code,
          image_url,
          rating,
          language
  FROM ota_hotel_profile
), 
pivoted AS (
  SELECT *
  FROM base
  PIVOT (
    MAX(address) AS address,
    MAX(TRIM(location_name)) AS location_name,
    MAX(TRIM(hotel_name)) AS hotel_name,
    MAX(TRIM(city_name)) AS city_name,
    MAX(TRIM(country_name)) AS country_name
    FOR language IN ('en' as en, 'th' as th)
  )
),
basePlaylist AS (
  SELECT a.*, ROW_NUMBER() OVER(PARTITION BY a.user_id ORDER BY a.important_value DESC) AS row_num
  FROM {UC_PREDICTION_PATH} a
  LEFT JOIN {UC_HOTEL_INFO_PATH} b
    ON a.target_hotel_id = b.hotel_id
  LEFT JOIN ota_hotel_profile c
    ON b.hotel_code = c.hotel_code
  WHERE c.display = 'Y' AND c.language = 'en'
),
playlist AS (
  SELECT  d.raw_user_id AS user_id, 
          a.row_num, 
          c.*
  FROM basePlaylist a
  LEFT JOIN {UC_HOTEL_INFO_PATH} b
    ON a.target_hotel_id = b.hotel_id
  LEFT JOIN pivoted c
    ON b.hotel_code = c.hotel_code
  LEFT JOIN {UC_USER_INFO_PATH} d
    ON a.user_id = d.user_id
  WHERE a.row_num <= 10
  ORDER BY d.user_id, a.row_num
),
topHitPlaylist AS (
  SELECT a.user_id, a.row_num, c.*
  FROM basePlaylist a
  LEFT JOIN {UC_HOTEL_INFO_PATH} b
    ON a.target_hotel_id = b.hotel_id
  LEFT JOIN pivoted c
    ON b.hotel_code = c.hotel_code

)

SELECT DISTINCT * EXCEPT(user_id, row_num)
FROM playlist
""")

# COMMAND ----------

def convert_delta_to_feature(fe, delta_df, feature_table, primary_keys, description, mode='merge'):
    try:
        print("Check feature store is already existed or not")
        fs = FeatureStoreClient()
        fs.get_table(feature_table)
    except:
        print("Create new feature store table")
        fe.create_table(
        name=feature_table,
        primary_keys=primary_keys,
        df=delta_df,
        schema=delta_df.schema,
        description=description
    )
    print("Write data to feature store table")
    fe.write_table(
        name=feature_table,
        df = delta_df,
        mode = mode
    )

# COMMAND ----------

fe = FeatureEngineeringClient()

# COMMAND ----------

print(f'Add feature table as {HOTEL_FEATURE_TABLE}')
convert_delta_to_feature(fe,
                         df,
                         HOTEL_FEATURE_TABLE,
                         primary_keys='hotel_code',
                         description='The table contain hotel profile for today recommendation batch')

# COMMAND ----------

online_store = AmazonDynamoDBSpec(
            region='ap-southeast-1',
            read_secret_prefix=''
        )

# COMMAND ----------

fe.publish_table(
    name=HOTEL_FEATURE_TABLE,
    online_store=online_store,
    mode='merge'
)