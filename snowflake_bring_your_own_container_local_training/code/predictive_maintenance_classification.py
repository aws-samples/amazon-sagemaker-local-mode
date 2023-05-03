#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from __future__ import print_function

import argparse
import json
import os

import boto3
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from snowflake.snowpark.functions import *
from snowflake.snowpark.session import Session
from snowflake.snowpark.types import *


def get_snowflake_session(args):
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=args.region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=args.secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = json.loads(get_secret_value_response['SecretString'])
    
    connection_parameters = {
        "account": secret["account"],
        "user": secret["user"], 
        "password": secret["password"],
        "role": "ACCOUNTADMIN",
        "database": "HOL_DB",
        "schema": "PUBLIC",
        "warehouse": "HOL_WH"
        }
    session = Session.builder.configs(connection_parameters).create()
    return(session)
    

if __name__ == "__main__":
    print("Training Started")
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--secret-name", type=str)
    parser.add_argument("--region-name", type=str)
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])

    args = parser.parse_args()
    print("Got Args: {}".format(args))

    session = get_snowflake_session(args)

    print("Start fetching from Snowflake")
    maintenance_df = session.table('maintenance')
    humidity_df = session.table('humidity')
    hum_udi_df = session.table('city_udf')

    print(f"maintenance table: {maintenance_df.to_pandas().shape}")
    print(f"humidity table: {humidity_df.to_pandas().shape}")
    print(f"city_udf table: {hum_udi_df.to_pandas().shape}")
    print("Fetching from Snowflake completed")

    # Join together the dataframes and prepare training dataset
    maintenance_city = maintenance_df.join(hum_udi_df, ["UDI"])
    maintenance_hum = maintenance_city.join(humidity_df,
    (maintenance_city.col("CITY") == humidity_df.col("CITY_NAME"))).select(
        col("TYPE"),
        col("AIR_TEMPERATURE_K"), col("PROCESS_TEMPERATURE"), col("ROTATIONAL_SPEED_RPM"), col("TORQUE_NM"),
        col("TOOL_WEAR_MIN"), col("HUMIDITY_RELATIVE_AVG"), col("MACHINE_FAILURE"))

    # Write training set to snowflake and materialize the data frame into a pandas data frame
    maintenance_hum.write.mode("overwrite").save_as_table("MAINTENANCE_HUM")
    maintenance_hum_df = session.table('MAINTENANCE_HUM').to_pandas()

    # Drop redundant column
    maintenance_hum_df = maintenance_hum_df.drop(columns=["TYPE"])

    # Split data into train and test
    y = maintenance_hum_df[["MACHINE_FAILURE"]].to_numpy()
    X = maintenance_hum_df.drop(columns=["MACHINE_FAILURE"]).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=123)

    # Train the model
    logistic_model = LogisticRegression(random_state=0, verbose=1).fit(X_train, y_train)

    # AUC score
    y_pred = logistic_model.predict_proba(X_test)[:, 1]
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}")

    # Save the model file
    joblib.dump(logistic_model, os.path.join(args.model_dir, "model.joblib"))
    print("Training Completed")

