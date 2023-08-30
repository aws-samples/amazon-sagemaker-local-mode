# This is a sample Python program that trains a BYOC CatBoost model with sagemaker-training-toolkit..
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build  -t sagemaker-catboost-regressor-training-toolkit-local container/.
########################################################################################################################

import os

import pandas as pd
from sagemaker.estimator import Estimator
from sagemaker.local import LocalSession
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

# For local training a dummy role will be sufficient
DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def main():
    data = fetch_california_housing()

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=45)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=45)

    trainX = pd.DataFrame(X_train, columns=data.feature_names)
    trainX['target'] = y_train

    valX = pd.DataFrame(X_test, columns=data.feature_names)
    valX['target'] = y_test

    testX = pd.DataFrame(X_test, columns=data.feature_names)

    os.makedirs('data/train', exist_ok=True)
    local_train = './data/train/california_train.csv'
    os.makedirs('data/validation', exist_ok=True)
    local_validation = './data/validation/california_validation.csv'
    os.makedirs('data/test', exist_ok=True)
    local_test = './data/test/california_test.csv'

    trainX.to_csv(local_train, header=None, index=False)
    valX.to_csv(local_validation, header=None, index=False)
    testX.to_csv(local_test, header=None, index=False)

    image = 'sagemaker-catboost-regressor-training-toolkit-local'

    local_regressor = Estimator(
        image_uri=image,
        entry_point='california_housing_train.py',
        source_dir='code',
        role=DUMMY_IAM_ROLE,
        instance_count=1,
        instance_type="local")

    train_location = 'file://'+local_train
    validation_location = 'file://'+local_validation

    local_regressor.fit({'train':train_location, 'validation': validation_location}, logs=True)

if __name__ == "__main__":
    main()
