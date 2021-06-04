# This is a sample Python program that trains a simple scikit-learn model on the Iris dataset.
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
from sklearn import datasets

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

def download_training_and_eval_data():
    if os.path.isfile('./data/iris.csv'):
        print('Training and dataset exist. Skipping Download')
    else:
        print('Downloading training dataset')

        # Load Iris dataset, then join labels and features
        iris = datasets.load_iris()
        joined_iris = np.insert(iris.data, 0, iris.target, axis=1)

        # Create directory and write csv
        os.makedirs("./data", exist_ok=True)
        np.savetxt("./data/iris.csv", joined_iris, delimiter=",", fmt="%1.1f, %1.3f, %1.3f, %1.3f, %1.3f")

        print('Downloading completed')

def do_inference_on_local_endpoint(predictor):
    print(f'\nStarting Inference on endpoint (local).')
    shape = pd.read_csv("data/iris.csv", header=None)

    a = [50 * i for i in range(3)]
    b = [40 + i for i in range(10)]
    indices = [i + j for i, j in itertools.product(a, b)]

    test_data = shape.iloc[indices[:-1]]
    test_X = test_data.iloc[:, 1:]
    test_y = test_data.iloc[:, 0]
    print("Predictions: {}".format(predictor.predict(test_X.values)))
    print("Actual: {}".format(test_y.values))


def main():
    download_training_and_eval_data()

    print('Starting model training.')
    print('Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')

    sklearn = SKLearn(
        entry_point="scikit_learn_iris.py",
        source_dir='code',
        framework_version="0.23-1",
        instance_type="local",
        role=DUMMY_IAM_ROLE,
        hyperparameters={"max_leaf_nodes": 30},
    )

    train_input = "file://./data/iris.csv"

    sklearn.fit({"train": train_input})
    print('Completed model training')

    print('Deploying endpoint in local mode')
    predictor = sklearn.deploy(initial_instance_count=1, instance_type='local')

    do_inference_on_local_endpoint(predictor)

    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint(predictor.endpoint_name)


if __name__ == "__main__":
    main()
