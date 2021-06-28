# This is a sample Python program that trains a simple CatBoost model using SageMaker scikit-learn Docker image, and then performs inference.
# This implementation will work on your *local computer* or in the *AWS Cloud*.
#
# Prerequisites:
#   1. Install required Python packages:
#      `pip install -r requirements.txt`
#   2. Docker Desktop installed and running on your computer:
#      `docker ps`
#   3. You should have AWS credentials configured on your local machine
#      in order to be able to pull the docker image from ECR.
###############################################################################################

import os

import pandas as pd
from sagemaker.predictor import csv_serializer
from sagemaker.sklearn import SKLearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

local_train = './data/train/boston_train.csv'
local_validation = './data/validation/boston_validation.csv'
local_test = './data/test/boston_test.csv'

def download_training_and_eval_data():
    if os.path.isfile('./data/train/boston_train.csv') and \
            os.path.isfile('./data/validation/boston_validation.csv') and \
            os.path.isfile('./data/test/boston_test.csv'):
        print('Training dataset exist. Skipping Download')
    else:
        print('Downloading training dataset')

        os.makedirs("./data", exist_ok=True)
        os.makedirs("./data/train", exist_ok=True)
        os.makedirs("./data/validation", exist_ok=True)
        os.makedirs("./data/test", exist_ok=True)

        data = load_boston()

        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=45)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=45)

        trainX = pd.DataFrame(X_train, columns=data.feature_names)
        trainX['target'] = y_train

        valX = pd.DataFrame(X_test, columns=data.feature_names)
        valX['target'] = y_test

        testX = pd.DataFrame(X_test, columns=data.feature_names)

        trainX.to_csv(local_train, header=None, index=False)
        valX.to_csv(local_validation, header=None, index=False)
        testX.to_csv(local_test, header=None, index=False)

        print('Downloading completed')


def main():
    download_training_and_eval_data()

    print('Starting model training.')
    print('Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')

    sklearn = SKLearn(
        entry_point="catboost_train_deploy.py",
        source_dir='code',
        framework_version="0.23-1",
        instance_type="local",
        role=DUMMY_IAM_ROLE,
    )

    train_location = 'file://' + local_train
    validation_location = 'file://' + local_validation

    sklearn.fit({'train': train_location, 'validation': validation_location})
    print('Completed model training')

    print('Deploying endpoint in local mode')
    predictor = sklearn.deploy(1, 'local', serializer=csv_serializer)

    with open(local_test, 'r') as f:
        payload = f.read().strip()

    predictions = predictor.predict(payload)
    print('predictions: {}'.format(predictions))

    predictor.delete_endpoint(predictor.endpoint)


if __name__ == "__main__":
    main()
