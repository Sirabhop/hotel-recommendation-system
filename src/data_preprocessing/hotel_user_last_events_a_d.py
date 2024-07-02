# Databricks notebook source
# MAGIC %md
# MAGIC #Overview OTA data

# COMMAND ----------

import sys
import pandas as pd

# COMMAND ----------

from configs.config import (dict_main_action_encoder, RANDOM_SEED, ENV_PATH, DATA_PATH, 
                            UC_HOTEL_INFO_PATH, UC_USER_INFO_PATH, UC_TRAIN_DATA_PATH,
                            UC_MODEL_PATH, window_spec, MODEL_EXPERIMENT_ID, API_EXPERIMENT_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Import data

# COMMAND ----------

df_event = spark.sql(f"""
WITH completeData AS (
SELECT DISTINCT
  customerInfo_enc:customerId AS user_id,
  request:hotelId AS hotel_id,
  CASE  WHEN status = 'CONFIRMED' THEN 'purchase'
        ELSE 'check'
  END AS main_action,
  TIMESTAMPADD(HOUR, 7, updatedDateTime) AS event_timestamp
FROM reservation_booking_details_i_d
UNION ALL
SELECT 
  DISTINCT a.user_id, a.hotel_id, b.main_action, MAX(a.event_timestamp) AS event_timestamp
FROM base_ota_events_i_d a
LEFT JOIN hotel_event_name b
  ON a.event_name = b.event_name
WHERE a.hotel_id IS NOT NULL AND b.main_action IS NOT NULL
GROUP BY a.user_id, a.hotel_id, b.main_action

), rankedData AS (
SELECT 
    user_id, hotel_id, main_action, event_timestamp,
    LAG(hotel_id) OVER (PARTITION BY user_id, hotel_id ORDER BY event_timestamp ASC) AS prev_hotel_id
FROM completeData

), lagData AS (
SELECT 
    user_id,
    CASE 
        WHEN (hotel_id = prev_hotel_id) OR (prev_hotel_id IS NULL) THEN hotel_id
        ELSE NULL
    END AS hotel_id,
    main_action,
    event_timestamp
FROM rankedData
)

SELECT  DISTINCT
        b.user_id,
        c.hotel_id,
        a.main_action,
        a.event_timestamp
FROM lagData a
INNER JOIN {UC_USER_INFO_PATH} b
  ON a.user_id = b.raw_user_id
INNER JOIN {UC_HOTEL_INFO_PATH} c
  ON a.hotel_id = c.hotel_code
INNER JOIN {UC_TRAIN_DATA_PATH} d
  ON c.hotel_id = d.hotel_id
  """).toPandas()

# COMMAND ----------

df.groupby('main_action')[['user_id']].count().plot(kind='bar')

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Preprocessing data

# COMMAND ----------

# Encode main action
df_event['main_action'] = df_event['main_action'].replace(dict_main_action_encoder)
print('Successfully encoded')

# COMMAND ----------

# Create sequence
groupby_col = 'user_id'
df_event.sort_values(by=['user_id', 'event_timestamp'], inplace=True)
df_event = df_event.groupby(groupby_col).agg(list).reset_index()
del df_event['event_timestamp']

# COMMAND ----------

def get_latest_events(row, sequence_length):
    if row in (pd.NA, pd.NaT):
        return None
    else:
        if len(row) < sequence_length:
            return row
        else:
            return row[:sequence_length]

# COMMAND ----------

col_list = list(df_event.columns)
col_list.remove(groupby_col)

for col in col_list:
    df_event[col+'_sequence'] = df_event[col].apply(lambda x: get_latest_events(x, window_spec['SEQUENCE_LENGTH']))
    del df_event[col]

# COMMAND ----------

df_event_transformed = df_event.merge(
    df_user_info[['user_id', 'age_group', 'gender']], how='left', on='user_id'
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Upload data

# COMMAND ----------

ds_event_transformed = spark.createDataFrame(df_event_transformed)

# COMMAND ----------

ds_event_transformed.write.mode('overWrite').saveAsTable('hotel_user_last_events_a_d')