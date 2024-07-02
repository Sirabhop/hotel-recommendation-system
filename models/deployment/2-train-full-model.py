# Databricks notebook source
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import os
import numpy as np
import pandas as pd
import mlflow

# COMMAND ----------

from configs.config import (dict_main_action_encoder, RANDOM_SEED, ENV_PATH, DATA_PATH, 
                            UC_USER_INFO_PATH, UC_HOTEL_INFO_PATH, UC_TRAIN_DATA_PATH, 
                            UC_MODEL_PATH, window_spec, MODEL_EXPERIMENT_ID, API_EXPERIMENT_ID)
from src.models.bst_model import create_model
from src.utils.preprocessing import get_dataset_from_csv, get_sequence_data

# COMMAND ----------

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(notebook_path))), "data")

data_directory

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')
mlflow.set_experiment(experiment_id=MODEL_EXPERIMENT_ID)

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

df_user_info = spark.table(UC_USER_INFO_PATH).toPandas()
df_hotel_info = spark.table(UC_HOTEL_INFO_PATH).toPandas()
df_event_raw = spark.table(UC_TRAIN_DATA_PATH).toPandas()

# COMMAND ----------

print(df_event_raw.event_timestamp.min(), df_event_raw.event_timestamp.max())

# COMMAND ----------

output = get_sequence_data(df_event_raw, df_user_info, 
                           window_spec, minimum_seen_hotels,
                           dict_main_action_encoder, data_directory, False)

# COMMAND ----------

train_dataset = get_dataset_from_csv(f'{data_directory}/full_train.csv', 'main_action_sequence', shuffle=data_shuffle_flag, batch_size=1024)

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
    dict_features=dict_features,
    output_units=len(dict_main_action_encoder)
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

result = model.fit(train_dataset, callbacks=[reduce_lr, er_stop], epochs=20)

# COMMAND ----------

last_run = mlflow.last_active_run()
last_run.info.run_id

# COMMAND ----------



# COMMAND ----------

model_uri = f"runs:/{str(last_run.info.run_id)}/model"
mlflow.register_model(model_uri, UC_MODEL_PATH)

# COMMAND ----------

client = mlflow.MlflowClient()

# COMMAND ----------

# Get run_id and latest version
model_version_infos = client.search_model_versions("name = '%s'" % UC_MODEL_PATH)

max_version_run_id = max(model_version_infos, key=lambda mv: int(mv.version)).run_id
max_version = max(model_version_infos, key=lambda mv: int(mv.version)).version

# COMMAND ----------

client.set_registered_model_alias(UC_MODEL_PATH, "lastest_version", int(max_version))