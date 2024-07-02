# Databricks notebook source
import pandas as pd
from pyspark.sql.functions import col
from datetime import datetime, timedelta

# COMMAND ----------

from configs.config import (dict_main_action_encoder, RANDOM_SEED, ENV_PATH, DATA_PATH, 
                            UC_HOTEL_INFO_PATH, UC_USER_INFO_PATH, UC_TRAIN_DATA_PATH,
                            UC_MODEL_PATH, window_spec, MODEL_EXPERIMENT_ID, API_EXPERIMENT_ID)

# COMMAND ----------

busi_dt = str(datetime.today().date())
print('Running on', busi_dt, f'and save to {UC_USER_INFO_PATH}')

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Update user id encoder

# COMMAND ----------

latest_user_id = spark.sql(f"""
SELECT max(int_user_id)
FROM {UC_USER_INFO_PATH}
""").collect()[0][0]

print('Latest user id =', latest_user_id)

# COMMAND ----------

df_user_info = spark.sql(
f"""
WITH base AS (
  SELECT  user_id, 
          MAX(register_date) AS register_date,
          LOWER(gender) AS gender,
          LAST(birth_date) AS birth_date
  FROM customer_profile_a_d
  WHERE user_id NOT IN (
    SELECT DISTINCT raw_user_id
    FROM {UC_USER_INFO_PATH}
  )
  GROUP BY user_id, gender
), rowNum AS (
SELECT  *,
        (ROW_NUMBER() OVER(ORDER BY register_date ASC)) + {latest_user_id} AS int_user_id
FROM base
ORDER BY register_date ASC
)
SELECT
  user_id AS raw_user_id, 
  int_user_id, 
  CONCAT('user_', int_user_id) AS user_id, 
  gender,
  TIMESTAMPDIFF(YEAR, birth_date, current_date) - (
        CASE
        WHEN MONTH(birth_date) > MONTH(current_date)
        OR (
            MONTH(birth_date) = MONTH(current_date)
            AND DAY(birth_date) > DAY(current_date)
        ) THEN 1
        ELSE 0
        END
    ) AS age,
  register_date,
  DATE('{busi_dt}') AS busi_dt
FROM rowNum
""").toPandas()

# COMMAND ----------

new_users = df_user_info.shape[0]
print('New user =', new_users)

# COMMAND ----------

df_user_info.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Groupping age

# COMMAND ----------

# MAGIC %md
# MAGIC - 60-80 retirement
# MAGIC - 45-60 senior
# MAGIC - 30-45 mid_career
# MAGIC - 26-30 establishment
# MAGIC - 23-26 early_jobber
# MAGIC - 19-23 undergrad
# MAGIC - 19-16 high_school
# MAGIC - 13-16 junior_high
# MAGIC - 6-12 elementary

# COMMAND ----------

dict_age_label = {
    'elementary': {
        'age_min': 0,
        'age_max': 13,
        'age_group_label': 'elementary',
        'age_group': 'group_1'
    },
    'junior_high': {
        'age_min': 13,
        'age_max': 16,
        'age_group_label': 'junior_high',
        'age_group': 'group_2'
    },
    'high_school': {
        'age_min': 16,
        'age_max': 19,
        'age_group_label': 'high_school',
        'age_group': 'group_3'
    },
    'undergrad': {
        'age_min': 19,
        'age_max': 23,
        'age_group_label': 'undergrad',
        'age_group': 'group_4'
    },
    'early_jobber': {
        'age_min': 23,
        'age_max': 26,
        'age_group_label': 'early_jobber',
        'age_group': 'group_5'
    },
    'establishment': {
        'age_min': 26,
        'age_max': 30,
        'age_group_label': 'establishment',
        'age_group': 'group_6'
    },
    'mid_career': {
        'age_min': 30,
        'age_max': 45,
        'age_group_label': 'mid_career',
        'age_group': 'group_7'
    },
    'senior': {
        'age_min': 45,
        'age_max': 60,
        'age_group_label': 'senior',
        'age_group': 'group_8'
    },
    'retirement': {
        'age_min': 60,
        'age_max': 999999,
        'age_group_label': 'retirement',
        'age_group': 'group_9'
    }
}

# COMMAND ----------

med_ages = spark.sql(
    f"""
    SELECT gender, MEDIAN(age) as med_age 
    FROM {UC_USER_INFO_PATH}
    GROUP BY gender
    ORDER BY gender
    """
).collect()

female_med_age = med_ages[0]['med_age']
male_med_age = med_ages[1]['med_age']
none_med_age = med_ages[2]['med_age']
other_med_age = med_ages[3]['med_age']

print('Female median age =', female_med_age, ', male median age =', male_med_age, 'and other gender age =', other_med_age, 'and none age =', none_med_age)

# COMMAND ----------

if new_users > 0:
    df_user_info.loc[df_user_info['gender']=='female', 'age'] = df_user_info.loc[df_user_info['gender']=='female', 'age'].fillna(female_med_age)
    df_user_info.loc[df_user_info['gender']=='male', 'age'] = df_user_info.loc[df_user_info['gender']=='male', 'age'].fillna(male_med_age)
    df_user_info.loc[df_user_info['gender']=='other', 'age'] = df_user_info.loc[df_user_info['gender']=='other', 'age'].fillna(other_med_age)
    df_user_info.loc[df_user_info['gender']=='none', 'age'] = df_user_info.loc[df_user_info['gender']=='none', 'age'].fillna(none_med_age)

    for key in dict_age_label.keys():
        df_user_info.loc[(df_user_info['age'] >= dict_age_label[key]['age_min']) & 
                        (df_user_info['age'] < dict_age_label[key]['age_max']), 'age_group_label'] = dict_age_label[key]['age_group_label']
        df_user_info.loc[(df_user_info['age'] >= dict_age_label[key]['age_min']) & 
                        (df_user_info['age'] < dict_age_label[key]['age_max']), 'age_group'] = dict_age_label[key]['age_group']
        
    print('Appending', df_user_info.shape[0], 'of new user')

    ds_user_info = spark.createDataFrame(df_user_info)
    ds_user_info = ds_user_info.withColumn("age", col("age").cast("double"))
    ds_user_info.write.mode('append').option('mergeSchema', True).saveAsTable(f'{UC_USER_INFO_PATH}')
    print('Successfully appended')
    
else:
    latest_date = spark.sql(f"""
        SELECT max(busi_dt)
        FROM {UC_USER_INFO_PATH}
        """).collect()[0][0]
    print('New users already catched up on', latest_date)

# COMMAND ----------

latest_user_id = spark.sql(f"""
SELECT max(int_user_id)
FROM {UC_USER_INFO_PATH}
""").collect()[0][0]

print('Latest user id =', latest_user_id)