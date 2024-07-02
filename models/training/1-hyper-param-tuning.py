# Databricks notebook source
import os

import math
import pandas as pd

import numpy as np
import tensorflow as tf
import keras_tuner as kt

from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# COMMAND ----------

from config.config import dict_main_action_encoder, RANDOM_SEED, ENV_PATH, DATA_PATH, widowing_spec

# COMMAND ----------

from preprocessing.get_dataset_from_csv import get_dataset_from_csv
from preprocessing.preprocessing import get_sequence_data
from model.bst_model import create_model

# COMMAND ----------

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC Global parameters

# COMMAND ----------

# MAGIC %md
# MAGIC # Get data

# COMMAND ----------

df_user_info = spark.table('user_info_i_d').toPandas()
df_hotel_info = spark.table('hotel_info_i_w').toPandas()
df_event_raw = spark.table('base_training_data').toPandas()

# COMMAND ----------

df_train, df_test, df_eval = get_sequence_data(df_event_raw, df_user_info, widowing_spec, 8, dict_main_action_encoder, DATA_PATH)

# COMMAND ----------

train_dataset = get_dataset_from_csv(f'{ENV_PATH}/data/train.csv', 'main_action_sequence', shuffle=False, batch_size=256)
test_dataset = get_dataset_from_csv(f'{ENV_PATH}/data/test.csv', 'main_action_sequence', shuffle=False, batch_size=256)
eval_dataset = get_dataset_from_csv(f'{ENV_PATH}/data/eval.csv', 'main_action_sequence', shuffle=False, batch_size=256)

# Recheck feature name
for elem in train_dataset.take(1):
  print('Features Name =', ', '.join(list(elem[0].keys())))

# COMMAND ----------

list_hotel_style = [col for col in list(df_hotel_info.columns) if col not in ['hotel_code', 'hotel_id', 'location_name', 'run_date', 'int_hotel_id', 'style_name']]

# COMMAND ----------

# MAGIC %md
# MAGIC # Model

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

hp = kt.HyperParameters()

# COMMAND ----------

class BSTHyperModel(kt.HyperModel):
    def construct_model(self,
                        dict_to_include_feature,
                        hidden_units,
                        sequence_length,
                        dropout_rate,
                        lr):
        
        pre_built_model = create_model(
                    dict_to_include_feature=dict_to_include_feature,
                    hidden_units=hidden_units,
                    sequence_length=sequence_length,
                    dropout_rate=dropout_rate,
                    df_hotel_style=df_hotel_info[list_hotel_style],
                    dict_features=dict_features,
                )
        pre_built_model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                     tf.keras.metrics.F1Score()],
        )

        return pre_built_model
    
    def build_model(self, hp):

        dict_to_include_feature = {
            'include_user_id':hp.Boolean("include_user_id"),
            'include_user_features':hp.Boolean("include_user_features"),
            'include_hotel_features':hp.Boolean("include_hotel_features")
        }
        lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        dropout_rate = hp.Choice("dropout_rate", [0.1, 0.2, 0.3])
        num_layers = hp.Int("FC_layers", min_value=2, max_value=7)
        
        hidden_units = []
        for i in range(num_layers):
            # Append number of units
            hidden_units.append(hp.Int("units", min_value=32, max_value=512, step=32))

        constructed_model = self.construct_model(
            dict_to_include_feature=dict_to_include_feature,
            hidden_units=hidden_units,
            sequence_length=sequence_length,
            dropout_rate=dropout_rate,
            lr=lr,
            
        )
        
        return constructed_model
    
    def fit(self, hp, model, train_dataset, validation_data, *args, **kwargs):

        sequence_length = hp.Int("sequence_length", min_value=4, max_value=10, step=1)
        step_size_values = [i for i in range(1, sequence_length + 1)]
        step_size = hp.Choice("step_size", values=step_size_values)

        window_spec = {
            'SEQUENCE_LENGTH':sequence_length,
            'STEP_SIZE':step_size
        }

        minimum_seen_hotels = hp.Int("minimum_seen_hotels", min_value=4, max_value=8)
        shuffle_flag = hp.Boolean("shuffle")

        df_train, df_test, _ = get_sequence_data(
            df_event_raw, 
            df_user_info, 
            widowing_spec, 
            minimum_seen_hotels, 
            dict_main_action_encoder, 
            DATA_PATH
            )

        train_dataset = get_dataset_from_csv(f'{ENV_PATH}/data/train.csv', 'main_action_sequence', shuffle=shuffle_flag, batch_size=256)
        eval_dataset = get_dataset_from_csv(f'{ENV_PATH}/data/eval.csv', 'main_action_sequence', shuffle=shuffle_flag, batch_size=256)

        final_model = self.build_model()

        return final_model.fit(
            train_dataset,
            validation_data=eval_dataset
            *args,
            **kwargs,
        )
        

# COMMAND ----------

def get_dataset_from_csv(csv_file_path, target_col_name, shuffle=False, batch_size=128):

    def process(features):
        hotel_id_sequence_string = features["hotel_id_sequence"]
        sequence_hotel_ids = tf.strings.split(hotel_id_sequence_string, ",").to_tensor()

        # The last movie id in the sequence is the target movie.
        features["target_hotel_id"] = sequence_hotel_ids[:, -1]
        features["sequence_hotel_ids"] = sequence_hotel_ids[:, :-1]

        # duration_string = features["complete_flag_sequence"]
        # duration_string = features["main_action_sequence"]
        duration_string = features[target_col_name]
        print(duration_string[0])
        sequence_duration = tf.strings.to_number(
            tf.strings.split(duration_string, ","), tf.dtypes.float32
        ).to_tensor()

        # The last rating in the sequence is the target for the model to predict.
        target = sequence_duration[:, -1]
        features[target_col_name] = sequence_duration[:, :-1]

        return features, target

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        num_epochs=1,
        header=True,
        field_delim="|",
        shuffle=shuffle,
    ).map(process)

    return dataset


# COMMAND ----------

class BSTHyperModel(kt.HyperModel):
    def construct_model(self,
                        dict_to_include_feature,
                        hidden_units,
                        sequence_length,
                        dropout_rate,
                        lr,
                        df_hotel_info,
                        list_hotel_style,
                        dict_features):
        
        # Assuming create_model is a function defined elsewhere
        pre_built_model = create_model(
            dict_to_include_feature=dict_to_include_feature,
            hidden_units=hidden_units,
            sequence_length=sequence_length,
            dropout_rate=dropout_rate,
            df_hotel_style=df_hotel_info[list_hotel_style],
            dict_features=dict_features,
        )
        pre_built_model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'],
        )

        return pre_built_model
    
    def build(self, hp):
        sequence_length = hp.Int("sequence_length", min_value=4, max_value=10)  # Included hp parameter for sequence length

        dict_to_include_feature = {
            'include_user_id': True,

            'include_user_features': True,
            'include_hotel_features': True
        }
        lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2)  # Altered to use the hp object for tuning learning rate
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5)  # Altered to use hp object for tuning dropout rate
        
        first_unit = hp.Int("units_" + str(1), min_value=64, max_value=512, step=32)
        second_unit = hp.Int("units_" + str(2), min_value=32, max_value=first_unit, step=32)

        hidden_units = [first_unit, second_unit]

        constructed_model = self.construct_model(
            dict_to_include_feature=dict_to_include_feature,
            hidden_units=hidden_units,
            sequence_length=sequence_length,
            dropout_rate=dropout_rate,
            lr=lr,
            df_hotel_info=df_hotel_info,
            list_hotel_style=list_hotel_style,
            dict_features=dict_features,
        )
        
        return constructed_model
    
    def fit(self, hp, model, train_dataset, validation_data, **kwargs):
        
        from config.config import dict_main_action_encoder, RANDOM_SEED, ENV_PATH, DATA_PATH

        sequence_length = hp.Int("sequence_length", min_value=4, max_value=10, step=1)
        step_size_values = [i for i in range(1, sequence_length + 1)]
        step_size = hp.Choice("step_size", values=step_size_values)

        window_spec = {
            'SEQUENCE_LENGTH':sequence_length,
            'STEP_SIZE':step_size
        }

        minimum_seen_hotels = hp.Int("minimum_seen_hotels", min_value=4, max_value=8)
        shuffle_flag = hp.Boolean("shuffle")

        df_user_info = spark.table('user_info_i_d').toPandas()
        df_event_raw = spark.table('base_training_data').toPandas()

        df_train, df_test, _ = get_sequence_data(df_event_raw, df_user_info, window_spec, minimum_seen_hotels, dict_main_action_encoder, DATA_PATH)

        # Assuming these functions are defined elsewhere
        train_dataset = get_dataset_from_csv(f'{ENV_PATH}/data/train.csv', 'main_action_sequence', shuffle=shuffle_flag, batch_size=256)
        eval_dataset = get_dataset_from_csv(f'{ENV_PATH}/data/eval.csv', 'main_action_sequence', shuffle=shuffle_flag, batch_size=256)

        final_model = model  # Use the model parameter directly since 'build' method is not used here

        return final_model.fit(
            train_dataset,
            validation_data=eval_dataset,
            **kwargs,  # Corrected the asterisks; it should be **kwargs instead of *args, **kwargs
        )

# COMMAND ----------

tuner = kt.Hyperband(BSTHyperModel(),
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory=OUT_DIR,
                     project_name='trial',
                     )

# COMMAND ----------

callbacks = [ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, min_lr=1e-5),
             EarlyStopping(monitor='val_accuracy', patience=5)]

# COMMAND ----------

tuner.search(train_dataset, validation_data=eval_dataset, epochs=50, callbacks=callbacks)

# COMMAND ----------

tuner.results_summary(3)

# COMMAND ----------

y = pd.DataFrame(y_interpreted)

# COMMAND ----------

main_action_encoder = LabelEncoder()
main_action_encoder.classes_ = np.load(f'{ENV_PATH}/label_encoder/main_action_encoder.npy', allow_pickle=True)
y['main_action'] = main_action_encoder.inverse_transform(y.actions)

# COMMAND ----------

y.groupby('main_action').count()