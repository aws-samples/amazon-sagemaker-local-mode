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
import os
import numpy as np

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import delta_sharing


if __name__ == "__main__":
    print("Training Started")
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument("--max_leaf_nodes", type=int, default=-1)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args = parser.parse_args()
    print("Got Args: {}".format(args))

    # Take the profile file, create a SharingClient, and read data from the delta lake table
    profile_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(profile_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.train, "train")
        )

    profile_file = profile_files[0]
    print(f'Found profile file: {profile_file}')

    # Create a SharingClient
    client = delta_sharing.SharingClient(profile_file)
    table_url = profile_file + "#delta_sharing.default.boston-housing"

    # Load the table as a Pandas DataFrame
    print('Loading boston-housing table from Delta Lake')
    train_data = delta_sharing.load_as_pandas(table_url)
    print(f'Train data shape: {train_data.shape}')

    # Drop null values - THIS SHOULD BE DONE IN PRE-PROCESSING STAGE AS BEST PRACTISE!
    train_data.dropna(inplace=True)

    # Split the data into training and testing sets
    X = train_data.iloc[:, 1:14]
    Y = train_data.iloc[:, 14]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    print(f'X_train.shape: {X_train.shape}')
    print(f'X_test.shape: {X_test.shape}')
    print(f'Y_train.shape: {Y_train.shape}')
    print(f'Y_test.shape: {Y_test.shape}')

    linear_model = LinearRegression()
    linear_model.fit(X_train, Y_train)

    # model evaluation for training set
    y_train_predict = linear_model.predict(X_train)
    rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
    r2 = r2_score(Y_train, y_train_predict)

    print("The model performance for training set")
    print("--------------------------------------")
    print(f'RMSE is {rmse}')
    print(f'R2 score is {r2}')
    print("\n")

    # Save model
    joblib.dump(linear_model, os.path.join(args.model_dir, "model.joblib"))

    print("Training Completed")


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
