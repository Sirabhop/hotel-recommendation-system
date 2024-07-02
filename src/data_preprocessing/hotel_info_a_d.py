# Databricks notebook source
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import OneHotEncoder

# COMMAND ----------

from configs.config import UC_HOTEL_INFO_PATH

# COMMAND ----------

df_hotel_info = spark.sql(
"""
WITH basedTable AS (
SELECT
  prod_a.hotel_code,
  CASE
    WHEN (prod_a.location_name IS NULL) OR (prod_a.location_name = '') THEN LOWER(REPLACE(TRIM(prod_a.city_name), ' ', ''))
    ELSE LOWER(CONCAT(REPLACE(TRIM(prod_a.location_name), ' ', ''), '_', REPLACE(TRIM(prod_a.city_name), ' ', '')))
  END AS location_name,
  LOWER(REPLACE(c.style_name, ' ', '_')) AS style_name,
  CAST(prod_a.latitude AS FLOAT), 
  CAST(prod_a.longitude AS FLOAT)
FROM
  ota_hotel_profile_a_d AS prod_a
LEFT JOIN 
  hotel_style_tagging AS c 
  ON prod_a.hotel_code = c.hotel_code
WHERE
  prod_a.language = 'en'
),
rankedBasedTable AS (
  SELECT
    *,
    ROW_NUMBER() OVER (ORDER BY hotel_code) AS int_hotel_id
  FROM basedTable
)
SELECT
  hotel_code,
  int_hotel_id,
  CONCAT('hotel_', CAST(int_hotel_id AS STRING)) AS hotel_id,
  CONCAT('location_', REPLACE(location_name, ',', '')) AS location_name,
  style_name,
  latitude,
  longitude
FROM
  rankedBasedTable;
""").toPandas()

# COMMAND ----------

null_label = 'unspecified'

# COMMAND ----------

df_hotel_info.loc[df_hotel_info['location_name'] == 'location_ท่าอากาศยานดอนเมือง_bangkok', 'location_name'] = 'location_donmuang_bangkok'

# COMMAND ----------

df_hotel_info['style_name'] = df_hotel_info['style_name'].fillna(null_label)
df_hotel_info['location_name'] = df_hotel_info['location_name'].fillna(null_label)

# COMMAND ----------

# MAGIC %md
# MAGIC Encode hotel style

# COMMAND ----------

def one_hot_encoder(df_target, null_label, feature_to_encode):
    
    list_vocab = df_target[feature_to_encode].str.replace(' ', '_').apply(str.lower).unique()

    if null_label not in list_vocab:
        list_vocab = np.array(sorted(np.append(list_vocab, null_label)))
    else:
        list_vocab = np.array(sorted(list_vocab))

    list_vocab = list_vocab.reshape(-1,1)

    encoder = OneHotEncoder()
    encoder.fit(list_vocab)

    return encoder

# COMMAND ----------

hotel_style_encoder = one_hot_encoder(df_hotel_info, null_label, 'style_name')

# COMMAND ----------

encoded_hotel_style = hotel_style_encoder.transform(df_hotel_info[['style_name']])
df_hotel_info.loc[:, hotel_style_encoder.categories_[0]] = encoded_hotel_style.toarray().astype(int)

# COMMAND ----------

# MAGIC %md
# MAGIC Encode hotel location

# COMMAND ----------

hotel_location_encoder = one_hot_encoder(df_hotel_info, null_label, 'location_name')

# COMMAND ----------

encoded_hotel_location = hotel_location_encoder.transform(df_hotel_info[['location_name']])
df_hotel_info.loc[:, hotel_location_encoder.categories_[0]] = encoded_hotel_location.toarray().astype(int)

# COMMAND ----------

busi_dt = datetime.datetime.now().date()
df_hotel_info.loc[:, 'run_date'] = busi_dt

# COMMAND ----------

ds_hotel_info = spark.createDataFrame(df_hotel_info)
ds_hotel_info.write.mode('overwrite').saveAsTable(UC_HOTEL_INFO_PATH)