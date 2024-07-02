# Databricks notebook source
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import os
import numpy as np
import pandas as pd

# COMMAND ----------

from config.config import dict_main_action_encoder, RANDOM_SEED, ENV_PATH, DATA_PATH, windowing_spec

from model.bst_model_bkup import create_model
from preprocessing.get_dataset_from_csv import get_dataset_from_csv
from preprocessing.preprocessing import get_sequence_data

# COMMAND ----------

import mlflow
mlflow.set_registry_uri('databricks-uc')
mlflow.set_experiment(experiment_id='')

# COMMAND ----------

# MAGIC %md
# MAGIC **Trial 0017 summary**
# MAGIC Hyperparameters:
# MAGIC * sequence_length: 6
# MAGIC * learning_rate: 0.009625627781977193
# MAGIC * dropout_rate: 0.37162969737595364
# MAGIC * units_1: 352
# MAGIC * units_2: 64
# MAGIC * step_size: 5
# MAGIC * minimum_seen_hotels: 5
# MAGIC * shuffle: True
# MAGIC
# MAGIC Score: 0.7374942898750305

# COMMAND ----------

window_spec = {
    'SEQUENCE_LENGTH':6,
    'STEP_SIZE':5
}

minimum_seen_hotels = 5
data_shuffle_flag = True
dropout_rate = 0.37162969737595364
learning_rate = 0.009625627781977193
hidden_units = [1024, 512, 256]

full_loop_flag = True

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Get data

# COMMAND ----------

df_user_info = spark.table('user_info_i_d').toPandas()
df_hotel_info = spark.table('hotel_info_a_d').toPandas()
df_event_raw = spark.table('base_training_data_a_d').toPandas()

# COMMAND ----------

print(df_event_raw.event_timestamp.min(), df_event_raw.event_timestamp.max())

# COMMAND ----------

print(df_event_raw.event_timestamp.min(), df_event_raw.event_timestamp.max())

# COMMAND ----------

output = get_sequence_data(df_event_raw, df_user_info, 
                           window_spec, minimum_seen_hotels,
                           dict_main_action_encoder, DATA_PATH, False)

# COMMAND ----------

train_dataset = get_dataset_from_csv(f'{ENV_PATH}/data/full_train.csv', 'main_action_sequence', shuffle=data_shuffle_flag, batch_size=256)

# Recheck feature name
for elem in train_dataset.take(1):
  print('Features Name =', ', '.join(list(elem[0].keys())))

# COMMAND ----------

# List of hotel location
list_hotel_location = []
for item in df_hotel_info.columns:
    if (item.find('location') != -1) & (item != 'location_name'):
        list_hotel_location.append(item)

list_hotel_location = sorted(list_hotel_location)

# List of hotel style
list_hotel_style = [col for col in list(df_hotel_info.columns) if col not in ['hotel_code', 'latitude', 'longitude','hotel_id', 'location_name', 'run_date', 'int_hotel_id', 'style_name']+ list_hotel_location] 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train model

# COMMAND ----------

dict_to_include_feature = {
    'include_user_id': True,
    'include_user_features': True,
    'include_hotel_features': True
}

# COMMAND ----------

dict_features = {
    'CATEGORICAL_FEATURES_WITH_VOCABULARY':{
        "user_id": list(df_event_raw.user_id.unique()),
        "hotel_id": list(df_hotel_info.hotel_id.unique()),
        "gender": list(df_user_info.gender.unique()),
        "age_group": list(df_user_info.age_group.unique()),
    },
    'USER_FEATURES':["gender", "age_group"],
    'HOTEL_FEATURES':["style_name"]
}

# COMMAND ----------

window_spec['SEQUENCE_LENGTH']

# COMMAND ----------

model = create_model(
    dict_to_include_feature=dict_to_include_feature, 
    hidden_units=hidden_units, 
    sequence_length=window_spec['SEQUENCE_LENGTH'], 
    dropout_rate=dropout_rate, 
    df_hotel_style=df_hotel_info[list_hotel_style], 
    dict_features=dict_features
)

# COMMAND ----------

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# COMMAND ----------

reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=3, min_lr=1e-5)
er_stop = EarlyStopping(monitor='accuracy', patience=3)

# COMMAND ----------

result = model.fit(train_dataset, callbacks=[reduce_lr, er_stop], epochs=120)

# COMMAND ----------

last_run = mlflow.last_active_run()
last_run.info.run_id

# COMMAND ----------

model_uri = f"runs:/{str(last_run.info.run_id)}/model"
mlflow.register_model(model_uri, "hotel_recommender")

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Generate playlist

# COMMAND ----------

import tensorflow as tf

# COMMAND ----------

col_list = ['user_id', 'gender', 'age_group', 'sequence_hotel_ids', 'main_action_sequence', 'target_hotel_id']

# COMMAND ----------

df_to_predict = spark.sql("""
                          SELECT * 
                          FROM user_to_predict 
                          WHERE user_id IN (SELECT DISTINCT user_id FROM base_training_data) AND
                                target_hotel_id IN (SELECT DISTINCT hotel_id FROM base_training_data)
                        LIMIT 100
                        """).toPandas()

# COMMAND ----------

df_to_predict.to_csv()

# COMMAND ----------

print(df_to_predict.shape[0])

# COMMAND ----------

df_to_predict[:100].to_csv(f'{ENV_PATH}/data/test.csv', sep="|", header=True)

# COMMAND ----------

def save_to_csv(df, k, prefix="to_predict_"):
    if k < 1:
        raise ValueError("k must be at least 1")
    
    rows = df.shape[0]
    rows_per_file = rows // k  # Calculate the number of rows per file
    remainder = rows % k  # Calculate the remainder for distribution
    
    dfs = {}

    export_path = f"{ENV_PATH}/data"
    for i in range(k):
        start = i * rows_per_file
        end = (i + 1) * rows_per_file
        if i == k - 1:
            end += remainder  # Add the remainder to the last file
        file_name = f"{prefix}{i+1}.csv"
        file_path = os.path.join(export_path, file_name)
        
        dfs[f'file_{i+1}'] = df.loc[start:end-1, ['user_id', 'target_hotel_id']]

        # df.iloc[start:end].to_csv(file_path, sep="|", header=True, index=False)  # Write to CSV
    
    print(f"{k} files have been saved to {export_path} with prefix '{prefix}'")

    return dfs

# COMMAND ----------

df_to_predict.loc[0:99,['user_id', 'target_hotel_id']].tail()

# COMMAND ----------

df_to_predict.iloc[0:100].tail()

# COMMAND ----------

dfs = save_to_csv(df_to_predict, 10)

# COMMAND ----------

model = mlflow.tensorflow.load_model(model_uri="models:/{MODEL_NAME}@champion")

# COMMAND ----------

def process(features):
    hotel_id_sequence_string = features["sequence_hotel_ids"]
    sequence_hotel_ids = tf.strings.split(hotel_id_sequence_string, ",").to_tensor()

    # The last movie id in the sequence is the target movie.
    features["sequence_hotel_ids"] = sequence_hotel_ids[:, :]

    # duration_string = features["complete_flag_sequence"]
    # duration_string = features["main_action_sequence"]
    duration_string = features["main_action_sequence"]
    sequence_duration = tf.strings.to_number(
        tf.strings.split(duration_string, ","), tf.dtypes.float32
    ).to_tensor()

    features["main_action_sequence"] = sequence_duration[:, :]

    return features

# COMMAND ----------

    test_ds = tf.data.experimental.make_csv_dataset(
        f'{ENV_PATH}/data/test.csv',
        batch_size=512,
        num_epochs=1,
        header=True,
        field_delim="|",
        shuffle=False,
    ).map(process)

# COMMAND ----------

dataset = {}

for i in range(10):
    
    dataset['file_'+str(i+1)] = tf.data.experimental.make_csv_dataset(
        f'{ENV_PATH}/data/to_predict_{i+1}.csv',
        batch_size=512,
        num_epochs=1,
        header=True,
        field_delim="|",
        shuffle=False,
    ).map(process)

# COMMAND ----------

def get_label_and_probability(predictions):
    final_results = []
    for prediction in predictions:
        max_index = np.argmax(prediction)
        probability = prediction[max_index]
        final_results.append((max_index, probability))
    return final_results

# COMMAND ----------

y_pred = {}
for i in range(k):
    y_pred[f'file_{i+1}'] = model.predict(dataset[f'file_{i+1}'])

# COMMAND ----------

file_name = f'file_{1+1}'
file_name

# COMMAND ----------

result={}
result[file_name] = get_label_and_probability(y_pred[file_name])

# COMMAND ----------

df_result = pd.DataFrame(result[file_name])
df_result.columns=['pred_action', 'prob']
df_result.shape

# COMMAND ----------

df_to_append = pd.concat([dfs[file_name].reset_index(drop=True), df_result], axis=1, ignore_index=True)

# COMMAND ----------

df_final = df_to_append.copy()

# COMMAND ----------

df_to_append

# COMMAND ----------

result = {}
for i in range(k):
    file_name = f'file_{i+1}'
    result[file_name] = get_label_and_probability(y_pred[file_name])
    df_result = pd.DataFrame(result[file_name])
    df_result.columns=['pred_action', 'prob']
    df_to_append = pd.concat([dfs[file_name].reset_index(drop=True), 
                              df_result], 
                             axis=1, ignore_index=True)
    if i == 0:
        print('df_final created')
        df_final = df_to_append.copy()
    else:
        print('start append')
        df_final = pd.concat([df_final.reset_index(drop=True), 
                              df_to_append.reset_index(drop=True)], 
                             axis=0, ignore_index=True)

# COMMAND ----------


def get_label_and_probability(predictions):
    final_results = []
    for prediction in predictions:
        max_index = np.argmax(prediction)
        probability = prediction[max_index]
        final_results.append((max_index, probability))
    return final_results


# COMMAND ----------

df_final.columns = ['user_id','hotel_id','main_action','prob']

# COMMAND ----------

ds = spark.createDataFrame(df_final)

# COMMAND ----------

ds.write.mode('overwrite').saveAsTable('user_hotel_prediction')

# COMMAND ----------

df_final = spark.sql(
    """
SELECT  user_id,
        CASE
          WHEN main_action = 0 THEN 'unfav'
          WHEN main_action = 1 THEN 'view'
          WHEN main_action = 2 THEN 'click'
          WHEN main_action = 3 THEN 'fav'
          WHEN main_action = 4 THEN 'share'
          WHEN main_action = 5 THEN 'purchase'
        END AS main_action,
        hotel_id,
        prob
FROM user_hotel_prediction
    """
).toPandas()

# COMMAND ----------

user_preference_summary = df_final.pivot_table(
    index='user_id', columns='main_action', values='hotel_id', aggfunc='count'
)

user_preference_summary.fillna(0, inplace=True)
user_preference_summary = user_preference_summary.reset_index()

# COMMAND ----------

list_promising_user = user_preference_summary.loc[
    user_preference_summary['click']+user_preference_summary['fav']+user_preference_summary['purchase'] > 0, 'user_id'
].values

list_ordinal_user = user_preference_summary.loc[
    user_preference_summary['click']+user_preference_summary['fav']+user_preference_summary['purchase'] == 0, 'user_id'
].values

# COMMAND ----------

df_final.sort_values(by=['user_id', 'prob'], ascending=False, inplace=True)

# COMMAND ----------

def generate_playlist(hotels, k):
    return list(hotels)[:k]

df_playlist = df_final.groupby('user_id')[['hotel_id']].agg(lambda x: generate_playlist(x, 10)).reset_index()

# COMMAND ----------

df_to_eval = spark.sql("""
    SELECT 
        b.user_id, 
        b.gender, 
        b.age_group, 
        c.hotel_id, 
        d.encoded_main_action AS main_action, 
        a.max_event_timestamp AS event_timestamp
    FROM 
        hotel_events_i_d a
    LEFT JOIN 
        user_info_i_d b
    ON 
        a.user_id = b.raw_user_id
    LEFT JOIN 
        hotel_info_i_w c
    ON 
        a.hotel_id = c.hotel_code
    LEFT JOIN 
        hotel_event_name d
    ON 
        a.event_name = d.event_name
    WHERE   
        DATE(a.max_event_timestamp) > DATE('2024-03-01') AND
        a.hotel_id IS NOT NULL AND
        d.encoded_main_action IS NOT NULL AND
        b.user_id IS NOT NULL
""").toPandas()

# COMMAND ----------

df_to_eval.sort_values(by=['user_id', 'event_timestamp'], inplace=True)
df_to_eval = df_to_eval[df_to_eval.groupby('user_id').cumcount() > 5]

# COMMAND ----------

df_eval_list = df_to_eval.groupby(['user_id', 'gender', 'age_group'])[['hotel_id', 'main_action']].agg(list).reset_index()

# COMMAND ----------

df_to_eval.loc['row_num'] = df_to_eval['hotel_id'].ne(df_to_eval['hotel_id'].shift()).cumsum()

# Apply collapsing logic within each group
collapsed_df = df_to_eval.groupby(['user_id', 'gender', 'age_group', 'row_num'])[['hotel_id', 'main_action']]. \
    agg({'hotel_id':'first', 'main_action':'max'}). \
    reset_index()

# COMMAND ----------

df_eval_set = collapsed_df.groupby(['user_id', 'gender', 'age_group'])[['hotel_id', 'main_action']].agg(list).reset_index()
df_eval_set.columns = ['user_id', 'gender', 'age_group', 'top_n_hotel_ids', 'top_n_main_actions']

# COMMAND ----------

df_playlist = df_playlist.merge(df_eval_set[['user_id', 'top_n_hotel_ids']], on='user_id', how='left')

# COMMAND ----------

df_playlist = df_playlist[~df_playlist['top_n_hotel_ids'].isna()]

# COMMAND ----------

def top_n_eval(row, n):
    actual_seen_hotels = row['top_n_hotel_ids']
    predicted_seen_hotels = row['hotel_id']

    if len(actual_seen_hotels) < n:
        return np.NaN
    else:
        return len(set(predicted_seen_hotels).intersection(set(actual_seen_hotels[:n])))

# COMMAND ----------

df_playlist.apply(lambda row: top_n_eval(row, 10), axis=1)

# COMMAND ----------

df_playlist

# COMMAND ----------

def get_top_k(hotels_string, k):
    list_hotels = hotels_string.split(",")
    return hotels_string.split(",")[:k]
df_to_evaluate['10_sequence_hotel_ids'] = df_to_evaluate['sequence_hotel_ids'].apply(lambda x: get_top_k(x, 10))
df_to_evaluate = df_to_evaluate[df_to_evaluate.user_id.isin(df_playlist.user_id.values)]

# COMMAND ----------

df_playlist = df_playlist.merge(df_to_evaluate[['user_id', '10_sequence_hotel_ids']], on='user_id', how='left')

# COMMAND ----------

set(df_playlist['10_sequence_hotel_ids'][0])

# COMMAND ----------

se