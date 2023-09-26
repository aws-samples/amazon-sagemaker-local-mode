# This is a sample Python program that trains a simple scikit-learn model on the California dataset.
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

import itertools
import pandas as pd
import numpy as np
import os

from sagemaker.sklearn import SKLearn
from sagemaker.local import LocalSession

import sagemaker
import boto3
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

local_mode = True

if local_mode:
    instance_type = "local"
    IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'
    sess = LocalSession()
    sess.config = {'local': {'local_code': True}}  # Ensure full code locality, see: https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode
else:
    instance_type = "ml.m5.xlarge"
    IAM_ROLE = 'arn:aws:iam::<ACCOUNT>:role/service-role/AmazonSageMaker-ExecutionRole-XXX'
    sess = sagemaker.Session()
    bucket = sess.default_bucket()                    # Set a default S3 bucket

prefix = 'DEMO-local-and-managed-infrastructure'

def download_training_and_eval_data():
    print('Downloading training dataset')

    # Load California Housing dataset, then join labels and features
    california = datasets.fetch_california_housing()
    dataset = np.insert(california.data, 0, california.target, axis=1)
    # Create directory and write csv
    os.makedirs("./data/train", exist_ok=True)
    os.makedirs("./data/validation", exist_ok=True)
    os.makedirs("./data/test", exist_ok=True)

    train, other = train_test_split(dataset, test_size=0.3)
    validation, test = train_test_split(other, test_size=0.5)

    np.savetxt("./data/train/california_train.csv", train, delimiter=",")
    np.savetxt("./data/validation/california_validation.csv", validation, delimiter=",")
    np.savetxt("./data/test/california_test.csv", test, delimiter=",")

    print('Downloading completed')

def do_inference_on_local_endpoint(predictor):
    print(f'\nStarting Inference on endpoint (local).')
    test_data = pd.read_csv("data/test/california_test.csv", header=None)
    test_X = test_data.iloc[:, 1:]
    test_y = test_data.iloc[:, 0]
    predictions = predictor.predict(test_X.values)
    print("Predictions: {}".format(predictions))
    print("Actual: {}".format(test_y.values))
    print(f"RMSE: {mean_squared_error(predictions, test_y.values)}")

def main():
    download_training_and_eval_data()

    print('Starting model training.')
    print('Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')

    sklearn = SKLearn(
        entry_point="scikit_learn_california.py",
        source_dir='code',
        framework_version="1.2-1",
        sagemaker_session=sess,
        instance_type=instance_type,
        role=IAM_ROLE,
        hyperparameters={"max_leaf_nodes": 30},
    )

    if local_mode:
        train_input = "file://./data/train/california_train.csv"
        validation_input = "file://./data/validation/california_validation.csv"
    else:
        # upload data to S3
        boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'data/train/california_train.csv')).upload_file('data/train/california_train.csv')
        boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'data/validation/california_validation.csv')).upload_file('data/validation/california_validation.csv')
        boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'data/test/california_test.csv')).upload_file('data/test/california_test.csv')

        train_input =f"s3://{bucket}/{prefix}/data/train/california_train.csv"
        validation_input =f"s3://{bucket}/{prefix}/data/validation/california_validation.csv"
        test_input =f"s3://{bucket}/{prefix}/data/test/california_test.csv"

    sklearn.fit({"train": train_input, "validation": validation_input})
    print('Completed model training')

    if local_mode:
        print('Deploying endpoint in local mode')
    else:
        print(f"deploying on the SageMaker managed infrastructure using a {instance_type} instance type")
    predictor = sklearn.deploy(initial_instance_count=1, instance_type=instance_type)

    do_inference_on_local_endpoint(predictor)

    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint()


if __name__ == "__main__":
    main()
