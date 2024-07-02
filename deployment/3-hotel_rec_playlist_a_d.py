# Databricks notebook source
# MAGIC %md
# MAGIC **3 Ways to productionize**
# MAGIC
# MAGIC 1. Publish 1 table which agg in list format -> 🐞 Bug: DynamoDB does not support struct value
# MAGIC 2. Publish 2 tables then joined them later -> ✅ Currently using this approach
# MAGIC 3. Publish 1 table in string and use json to load -> 🐞 Bug: limited capacity 400KB per row

# COMMAND ----------

# MAGIC %md
# MAGIC **The output table from this notebook are**
# MAGIC 1. `ml_hotel_recommendation.hotel_rec_playlist_a_d` -> Serve as online feature store for table inference API
# MAGIC 2. `ml_hotel_recommendation.hotel_rec_playlist_i_d` -> Collected for feedback learning in the future

# COMMAND ----------

dev_env = 'pd'
data_env = 'prod'

# COMMAND ----------

from configs.config import (dict_main_action_encoder, RANDOM_SEED, ENV_PATH, DATA_PATH, 
                            UC_HOTEL_INFO_PATH, UC_USER_INFO_PATH, 
                            UC_TRAIN_DATA_PATH, UC_RECENT_BEHAVIOR_PATH, 
                            DBFS_PATH, UC_PREDICTION_PATH, MODEL_NAME, HOTEL_FEATURE_TABLE, PLAYLIST_TABLE, 
                            UC_MODEL_PATH, window_spec, MODEL_EXPERIMENT_ID, API_EXPERIMENT_ID,
                            ENTITY_NAME, ENDPOINT_NAME)

# Set up variable
dbutils.widgets.text('DYNAMODB_SCOPE', '')
dbutils.widgets.text('DYNAMODB_ACCESS_KEY', '')
dbutils.widgets.text('DYNAMODB_SECRET_KEY', '')
dbutils.widgets.text('EXPERIMENT_ID', '')
dbutils.widgets.text('MODEL_NAME', MODEL_NAME)
dbutils.widgets.text('CATALOG', 'playground_prod')

list_variable = ['DYNAMODB_SCOPE', 'DYNAMODB_ACCESS_KEY', 'DYNAMODB_SECRET_KEY', 'EXPERIMENT_ID', 'MODEL_NAME', 'CATALOG']
dict_variable = {}
for variable in list_variable:
    dict_variable[variable] = dbutils.widgets.get(variable)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.feature_engineering.online_store_spec import AmazonDynamoDBSpec
from databricks.feature_store import FeatureStoreClient

import pyspark.sql.functions as F

from mlflow import start_run, register_model, last_active_run
from mlflow.tracking import MlflowClient
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc import log_model, load_model, PythonModel
from mlflow.deployments import get_deploy_client

from tqdm import tqdm

import logging
import boto3
import json
import mlflow
import os
import datetime
from boto3.dynamodb.conditions import Key, Attr

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Publish 1 table in string and use json to load

# COMMAND ----------

df = spark.sql(f"""
WITH playlist AS (
  SELECT  a.user_id, a.target_hotel_id, 
          ROW_NUMBER() OVER(PARTITION BY a.user_id ORDER BY a.important_value DESC) AS rank
  FROM {UC_PREDICTION_PATH} a
)
SELECT b.raw_user_id AS user_id, c.hotel_code AS hotel_code, a.rank
FROM playlist a
INNER JOIN {UC_USER_INFO_PATH} b
  ON a.user_id = b.user_id
INNER JOIN {UC_HOTEL_INFO_PATH} c
  ON a.target_hotel_id = c.hotel_id
WHERE a.rank <= 10
""")

# COMMAND ----------

df = df \
    .select("user_id", "hotel_code") \
    .groupBy('user_id') \
    .agg(F.array_agg('hotel_code').alias('sv_playlist')) \
    .filter("user_id IS NOT NULL")

# COMMAND ----------

# df.write.mode('append').saveAsTable(DELTA_TABLE)

# COMMAND ----------

fe = FeatureEngineeringClient()

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

print(f'Add feature table as {PLAYLIST_TABLE}')
convert_delta_to_feature(fe,
                         df,
                         PLAYLIST_TABLE,
                         primary_keys='user_id',
                         description='The table contain hotel_id list for each user')

# COMMAND ----------

online_store = AmazonDynamoDBSpec(
            region='ap-southeast-1',
            read_secret_prefix=''
        )

# COMMAND ----------

fe.publish_table(
    name=PLAYLIST_TABLE,
    online_store=online_store,
    mode='merge'
)

# COMMAND ----------

DYNAMODB_ACCESS_KEY = dbutils.secrets.get(dict_variable['DYNAMODB_SCOPE'], dict_variable['DYNAMODB_ACCESS_KEY'])
DYNAMODB_SECRET_KEY = dbutils.secrets.get(dict_variable['DYNAMODB_SCOPE'], dict_variable['DYNAMODB_SECRET_KEY'])

client = boto3.client('dynamodb', 
                region_name='ap-southeast-1',
                            aws_access_key_id=DYNAMODB_ACCESS_KEY,
                            aws_secret_access_key=DYNAMODB_SECRET_KEY
                            )

# COMMAND ----------

class getPersonalizedHotels(PythonModel):

    def __init__(self, keys):
        logging.basicConfig()
        self.logger = logging.getLogger(__name__)
        self.aws_access_key = keys[0]
        self.aws_secret_key = keys[1]
        self.recommended_hotels_table = PLAYLIST_TABLE
        self.hotel_info_table = HOTEL_FEATURE_TABLE

    def load_context(self, context):
        self.client = boto3.client('dynamodb', 
                            region_name='ap-southeast-1',
                            aws_access_key_id=self.aws_access_key,
                            aws_secret_access_key=self.aws_secret_key
                            )

    def __query_recommended_hotels(self, user_id):
        
        response = self.client.get_item(
            TableName=self.recommended_hotels_table,
            Key={
                '_feature_store_internal__primary_keys': {
                    'S': f'["{user_id}"]'
                }
            },
            AttributesToGet=[
                'sv_playlist'
            ]
        )
        return response
    
    def __get_hotel_info(self, hotel_code):
        # Optimize by fetching only required attributes
        response = self.client.get_item(
            TableName=self.hotel_info_table,
            Key={
                '_feature_store_internal__primary_keys': {
                    'S': f'["{hotel_code}"]'
                }
            },
            AttributesToGet=[
                'th_hotel_name', 'en_hotel_name', 'en_address', 'th_address', 
                'location_code', 'en_location_name', 'th_location_name',
                'city_code', 'en_city_name', 'th_city_name', 
                'country_code', 'en_country_name', 'th_country_name', 
                'rating', 'review_score', 'review_count', 'review_description',
                '11infotech_promotion', 'admin_promotion_line1_th', 'admin_promotion_line2_th',
                'admin_promotion_line1_en', 'admin_promotion_line2_en', 
                'capsule_promotion', 'image_url'
            ]
        )
        return response.get('Item', {})  # Handle case where item is not found

    def __structure_hotel_info(self, hotel_code):
        """
        Extracts and structures hotel data from a raw data source.

        Args:
            hotel_code (str): The unique identifier for the hotel.
            hotel_info (dict): A dictionary containing raw hotel information.

        Returns:
            dict: A cleaned and structured dictionary representing the hotel data.
        """

        # Query
        hotel_info = self.__get_hotel_info(hotel_code)

        # Construct info
        data = {
            'hotelId': hotel_code,
            'hotelNameTh': hotel_info.get('th_hotel_name', {}).get('S', ''),
            'hotelNameEn': hotel_info.get('en_hotel_name', {}).get('S', ''),
            'address': {
                'addressEn': hotel_info.get('en_address', {}).get('S', ''),
                'addressTh': hotel_info.get('th_address', {}).get('S', ''),
                'locationId': hotel_info.get('location_code', {}).get('S', ''),
                'locationNameEn': hotel_info.get('en_location_name', {}).get('S', ''),
                'locationNameTh': hotel_info.get('th_location_name', {}).get('S', ''),
                'cityId': hotel_info.get('city_code', {}).get('S', ''),
                'cityNameEn': hotel_info.get('en_city_name', {}).get('S', ''),
                'cityNameTh': hotel_info.get('th_city_name', {}).get('S', ''),
                'countryId': hotel_info.get('country_code', {}).get('S', ''),
                'countryNameEn': hotel_info.get('en_country_name', {}).get('S', ''),
                'countryNameTh': hotel_info.get('th_country_name', {}).get('S', ''),
            },
            'rating': hotel_info.get('rating', {}).get('N', 0),
            'review': {
                'score': hotel_info.get('review_score', 5),  # Default to 5 if not found
                'numReview': hotel_info.get('review_count', 100), # Default to 100 if not found
                'description': hotel_info.get('review_description', ""),
            },
            '11infotechPromotion': hotel_info.get('11infotech_promotion', []), 
            'adminPromotionLine1Th': hotel_info.get('admin_promotion_line1_th', ""),
            'adminPromotionLine2Th': hotel_info.get('admin_promotion_line2_th', ""),
            'adminPromotionLine1En': hotel_info.get('admin_promotion_line1_en', ""),
            'adminPromotionLine2En': hotel_info.get('admin_promotion_line2_en', ""),
            'capsulePromotion': hotel_info.get('capsule_promotion', []),
            'image': hotel_info.get('image_url', {}).get('S', ''),
        }
        return data
    
    def __format_output(self, items):
        """
        Get hotel info
        """
        recommended_items = [self.__structure_hotel_info(v['S']) for v in items]
        return recommended_items
        
    def __postprocess_result(self, user_id, response):
        try:
            if "Item" in response:
                return {'user_id': user_id, 'recommended_items': self.__format_output(response['Item']['sv_playlist']['L'])[:10]}
            else:
                response = self.__query_recommended_hotels('default')  # Ensure this method is defined
                return {'user_id': user_id, 'recommended_items': self.__format_output(response['Item']['sv_playlist']['L'])[:10]}
        except Exception as e:  # Add error handling
            self.logger.error(f"Error processing results for user {user_id}: {e}")
            return {'user_id': user_id, 'recommended_items': []}  # Return an empty list on error

    def predict(self, context, model_input):
        """
        Get list of hotel_id from given user_id
        """
        user_id = model_input['user_id'].values[0]
        query_items = self.__query_recommended_hotels(user_id)
        result = self.__postprocess_result(user_id, query_items)
        return result

# COMMAND ----------

# Wrap model
DYNAMODB_ACCESS_KEY = dbutils.secrets.get(dict_variable['DYNAMODB_SCOPE'], dict_variable['DYNAMODB_ACCESS_KEY'])
DYNAMODB_SECRET_KEY = dbutils.secrets.get(dict_variable['DYNAMODB_SCOPE'], dict_variable['DYNAMODB_SECRET_KEY'])

wrapped_model = getPersonalizedHotels(keys=[DYNAMODB_ACCESS_KEY, DYNAMODB_SECRET_KEY])

# COMMAND ----------

# Define Pyfunc model signature
input_schema = Schema([ColSpec("string", 'user_id')])
signature = ModelSignature(inputs=input_schema)

# COMMAND ----------

# Define input example
input_example = {
    'user_id': '112'
}

# COMMAND ----------

# Set experiment
with start_run(experiment_id=dict_variable['EXPERIMENT_ID']):
    log_model("model", 
              python_model=wrapped_model, 
              input_example=input_example, 
              signature=signature
              )

# COMMAND ----------

last_run_info =last_active_run()
run_id = last_run_info.info.run_id
run_id

# COMMAND ----------

loaded_model = load_model(f"runs:/{run_id}/model")

# COMMAND ----------

predicted_result = loaded_model.predict({'user_id':'112'})

# COMMAND ----------

model_uri = f"runs:/{str(run_id)}/model"
register_model(model_uri, MODEL_NAME)

# COMMAND ----------

def get_latest_model_version(model_name, stage=None):
    """Fetches the latest version of the specified registered MLflow model.

    Args:
        model_name (str): The name of the registered model.
        stage (str, optional): The stage to filter by (e.g., "Staging", "Production", "None"). Defaults to None (latest overall version).

    Returns:
        mlflow.entities.model_registry.ModelVersion: The latest model version object, or None if no model is found.
    """
    client = MlflowClient()

    # Get all versions of the model
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except mlflow.exceptions.MlflowException as e:
        print(f"Error retrieving model versions: {e}")
        return None
    
    # Check if no versions exist
    if not versions:
        print(f"No versions found for model: {model_name}")
        return None

    versions = [int(v.version) for v in versions]

    # Sort by version number (descending)
    latest_version = sorted(versions, reverse=True)[0]

    print(f"Latest version of model '{model_name}': {latest_version}")
    return latest_version

# COMMAND ----------

API_VERSION = get_latest_model_version(MODEL_NAME)
api_client = get_deploy_client("databricks")

# COMMAND ----------

CONFIG = {
  "served_entities": [
    {
    "name": ENTITY_NAME,
    "entity_name": ENTITY_NAME,
    "entity_version": str(API_VERSION),
    "workload_size": "Large",
    "scale_to_zero_enabled": False
    }
  ]
}

# COMMAND ----------

endpoint = api_client.update_endpoint(
    endpoint=ENDPOINT_NAME,
    config=CONFIG)

# COMMAND ----------
