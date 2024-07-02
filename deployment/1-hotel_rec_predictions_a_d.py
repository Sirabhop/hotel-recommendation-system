# Databricks notebook source
import pandas as pd
import tensorflow as tf
import numpy as np

import mlflow

import os
import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from datetime import datetime, timedelta

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

from configs.config import (dict_main_action_encoder, RANDOM_SEED, ENV_PATH, DATA_PATH, 
                            UC_HOTEL_INFO_PATH, UC_USER_INFO_PATH, 
                            UC_TRAIN_DATA_PATH, UC_RECENT_BEHAVIOR_PATH, 
                            DBFS_PATH, UC_PREDICTION_PATH,
                            UC_MODEL_PATH, window_spec, MODEL_EXPERIMENT_ID, API_EXPERIMENT_ID)
                            
                            
from src.utils.preprocessing import get_sequence_data, create_sequences, merge_sequece_lists

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Get hotel and user info

# COMMAND ----------

df_hotel = spark.sql(f"""
                     SELECT * 
                     FROM {UC_HOTEL_INFO_PATH} a
                     INNER JOIN {UC_TRAIN_DATA_PATH} b
                        ON STRING(a.hotel_id) = STRING(b.hotel_id)
                    """).toPandas()
df_user = spark.table(UC_USER_INFO_PATH).toPandas()

# COMMAND ----------

df_hotel.groupby('location_name')[['hotel_id']].count().sort_values(by='hotel_id', ascending=False).head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Prepare search event

# COMMAND ----------

# MAGIC %md
# MAGIC Import `playground_prod.ml_hotel_recommendation.hotel_user_last_events_a_d` table

# COMMAND ----------

df_all_users = spark.table(UC_RECENT_BEHAVIOR_PATH).toPandas()
df_trained_users = spark.sql(f"SELECT DISTINCT user_id FROM {UC_TRAIN_DATA_PATH}").toPandas()

# COMMAND ----------

df_cold_start = df_all_users[(~df_all_users.user_id.isin(df_trained_users.user_id)) |
                             (df_all_users.hotel_id_sequence.apply(lambda x: len(x)) < window_spec['SEQUENCE_LENGTH'])].reset_index(drop=True)
df_eligible = df_all_users[(df_all_users.user_id.isin(df_trained_users.user_id)) &
                           (df_all_users.hotel_id_sequence.apply(lambda x: len(x)) >= window_spec['SEQUENCE_LENGTH'])].reset_index(drop=True)

# COMMAND ----------

print('Insanity check =', sum(df_cold_start.user_id.isin(df_eligible.user_id)))

# COMMAND ----------

print('Total eligible users =', df_eligible.shape[0], 'and total COLD-START users =', df_cold_start.shape[0])

# COMMAND ----------

df_eligible['latest_hotel_id'] = df_eligible.hotel_id_sequence.apply(lambda x: x[-1])

df_eligible = df_eligible.merge(
    df_hotel[['hotel_id', 'location_name']],
    left_on = 'latest_hotel_id',
    right_on = 'hotel_id',
    how='left'
)

del df_eligible['latest_hotel_id']
del df_eligible['hotel_id']

# COMMAND ----------

def join_value_in_sequence(df, col_list):
    for col in col_list:
        df[col] = df[col].apply(lambda x: [str(value) for value in x])
        df[col] = df[col].apply(lambda x: ",".join(x))
    return df

# COMMAND ----------

df_eligible['hotel_id_sequence'] = df_eligible.hotel_id_sequence.apply(lambda x: x[:5])
df_eligible['main_action_sequence'] = df_eligible.main_action_sequence.apply(lambda x: x[:5])

df_eligible = join_value_in_sequence(df_eligible, ['hotel_id_sequence', 'main_action_sequence'])

# COMMAND ----------

df_eligible = df_eligible.merge(df_hotel[['hotel_id', 'location_name']], 
                            on='location_name', 
                            how='outer')

df_eligible = df_eligible[~df_eligible.user_id.isnull()]
del df_eligible['location_name']

df_eligible.rename(columns={'hotel_id':'target_hotel_id'}, inplace=True)

# COMMAND ----------

def export_csv_parallel(df, chunk_size=1000000, num_processes=4):
    dfs = []
    total_rows = len(df)
    num_chunks = math.ceil(total_rows / chunk_size)
    chunks = [df[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for i, result in enumerate(tqdm(executor.map(partial(write_csv_chunk, base_path=DBFS_PATH), range(num_chunks), chunks), total=num_chunks, desc="Exporting CSV Files")):
            dfs.append(result)

    return dfs

def write_csv_chunk(chunk_id, chunk, base_path):
    file_path = os.path.join(base_path, f'final_{chunk_id}.csv')
    chunk.to_csv(file_path, sep='|', header=True, index=False)
    return file_path

# COMMAND ----------

dfs = export_csv_parallel(df_eligible)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Make prediction

# COMMAND ----------

model = mlflow.tensorflow.load_model(model_uri=f"models:/{UC_MODEL_PATH}@lastest_version")

# COMMAND ----------

def process(features):
    features["sequence_hotel_ids"] = tf.strings.split(features["hotel_id_sequence"], ",").to_tensor()
    features["main_action_sequence"] = tf.strings.to_number(
        tf.strings.split(features["main_action_sequence"], ","), tf.dtypes.float32
    ).to_tensor()

    return features

tfds = {}
for i in tqdm(range(len(dfs)), desc='Importing to tensorflow dataset'):
    tfds[f'dataset_{i}'] = tf.data.experimental.make_csv_dataset(
                                dfs[i],
                                batch_size=1024,
                                num_epochs=1,
                                header=True,
                                field_delim="|",
                                shuffle=False,
                            ).map(process)

# COMMAND ----------

# Recheck feature name
for elem in tfds['dataset_0'].take(1):
  print('Features Name =', ', '.join(list(elem.keys())))

# COMMAND ----------

def get_label_and_probability(predictions):
    final_results = []
    for prediction in predictions:
        max_index = np.argmax(prediction)
        probability = prediction[max_index]
        final_results.append((max_index, probability))
    return final_results

# COMMAND ----------

df_result = pd.DataFrame()
for i in tqdm(range(len(tfds)), desc='Predicting the hotels'):
    y_pred = model.predict(tfds[f'dataset_{i}'], batch_size=1024)
    df_result = pd.concat([pd.DataFrame(get_label_and_probability(y_pred)), df_result], axis=0)

# COMMAND ----------

df_result.columns=['pred_action', 'prob']
df_result.shape

# COMMAND ----------

df = pd.concat([df_eligible.reset_index(drop=True), df_result.reset_index(drop=True)], axis=1)

# COMMAND ----------

df['important_value'] = df['pred_action'] * df['prob']

# COMMAND ----------

df = df.sort_values(by=['user_id', 'important_value'], ascending=False).reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #4. Trending hotel

# COMMAND ----------

min_date = str((datetime.now() - timedelta(days=8)).date())
max_date = str((datetime.now() - timedelta(days=1)).date())

# COMMAND ----------

top10 = spark.sql(f"""
WITH completeData AS (
  SELECT
    bookingUrn AS booking_id,
    FROM_JSON(request,
      'hotelId STRING, 
      cityId STRING, 
      checkInDate STRING, 
      checkOutDate STRING, 
      room ARRAY<STRUCT
        <roomType:STRING, 
        adults:INT, 
        children:INT, 
        childAge1:INT, 
        childAge2:INT>>, 
      currency STRING, 
      countryId STRING, 
      roomCode STRING, 
      roomCategory STRING,
      price DOUBLE, 
      supplierId STRING, 
      supplierName STRING, 
      checkPrice INT, 
      mealCode STRING'
    ) AS request_data,
    FROM_JSON(customerInfo_enc,
      'customerId STRING, firstName STRING, lastName STRING, membershipId STRING'
    ) AS customerInfo_enc,
    FROM_JSON(hotelBookingDetails,
      'addOnServices ARRAY<STRING>,
      additionalNeedsTxt STRING,
      addressEn STRING,
      addressTh STRING,
      checkInDate STRING,
      checkInTime STRING,
      checkOutDate STRING,
      checkOutTime STRING,
      cityId STRING,
      contactNumber STRING,
      countryId STRING,
      discount DOUBLE,
      fees DOUBLE,
      hotelId STRING,
      hotelImage STRING,
      hotelNameEn STRING,
      hotelNameTh STRING,
      isAddOnServicesAvailable BOOLEAN,
      latitude DOUBLE,
      longitude DOUBLE,
      rating STRING,
      ratingCount STRING,
      ratingTextEn STRING,
      ratingTextTh STRING,
      roomDetails STRUCT<
        cancellationPolicy STRUCT<
          en ARRAY<STRUCT<
            cancellationChargeDescription STRING,
            cancellationDaysDescription STRING,
            cancellationStatus STRING,
            days INT
          >>,
          th ARRAY<STRUCT<
            cancellationChargeDescription STRING,
            cancellationDaysDescription STRING,
            cancellationStatus STRING,
            days INT
          >>
        >,
        facilities ARRAY<STRUCT<key STRING, value STRING>>,
        hotelBenefits STRUCT<en ARRAY<STRING>, th ARRAY<STRING>>,
        mealTypeEn STRING,
        mealTypeTh STRING,
        numberOfNights STRING,
        perNightPrice DOUBLE,
        promotion STRUCT<en ARRAY<STRING>, th ARRAY<STRING>>,
        roomCategories ARRAY<STRUCT<
          childAge1 INT,
          childAge2 INT,
          mealTypeId STRING,
          noOfAdults INT,
          noOfRooms INT,
          noOfRoomsAndName STRING,
          price DOUBLE,
          priceOfRoomWithChildMeal DOUBLE,
          rateKey STRING,
          rateKeyRa STRING,
          roomId STRING,
          roomName STRING,
          roomType STRING
        >>,
        roomInfo STRUCT<roomFacilities STRUCT<en ARRAY<STRING>, th ARRAY<STRING>>>,
        supplier STRING,
        supplierId STRING,
        supplierName STRING,
        totalPrice DOUBLE
        >,
        taxes DOUBLE'
    ) AS hotelBookingDetails,
    updatedDateTime,
    status
  FROM
    reservation_booking_details_i_d
), base AS (
SELECT customerInfo_enc.customerId AS user_id, request_data.hotelId AS hotel_id, DATE(TIMESTAMPADD(HOUR, 7, updatedDateTime)) AS reservation_date
FROM completeData
)
SELECT b.hotel_id FROM base a
LEFT JOIN hotel_info_a_d b
  ON a.hotel_id = b.hotel_code
WHERE a.reservation_date BETWEEN DATE('{min_date}') AND DATE('{max_date}')
GROUP BY b.hotel_id
ORDER BY COUNT(DISTINCT a.user_id) DESC
LIMIT 10
"""
).collect()

top10 = [v['hotel_id'] for v in top10]
top10

# COMMAND ----------

topHit = {}
cols = df.columns

for col in cols:
    if col == 'target_hotel_id':
        topHit[col] = top10
    elif col == 'user_id':
        topHit[col] = 'default'
    elif col == 'important_value':
        list_range = [i for i in range(1, 11)]
        list_range.reverse()
        topHit[col] = list_range
    else:
        topHit[col] = pd.NaT

# COMMAND ----------

# MAGIC %md
# MAGIC #5. Export

# COMMAND ----------

df_final = pd.concat([pd.DataFrame(topHit), df], axis=0)

# COMMAND ----------

ds = spark.createDataFrame(df_final)
ds.write.mode('overWrite').option('overwriteSchema', 'True').saveAsTable(UC_PREDICTION_PATH)