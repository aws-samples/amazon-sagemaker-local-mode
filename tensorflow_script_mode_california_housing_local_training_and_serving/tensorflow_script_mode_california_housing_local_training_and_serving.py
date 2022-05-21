# This is a sample Python program that trains a simple TensorFlow California Housing model.
# This implementation will work on your *local computer*.
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

import numpy as np
import pandas as pd
from sklearn.datasets import *
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sagemaker.tensorflow import TensorFlow


DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def download_training_and_eval_data():
    if os.path.isfile('./data/train/x_train.npy') and \
            os.path.isfile('./data/test/x_test.npy') and \
            os.path.isfile('./data/train/y_train.npy') and \
            os.path.isfile('./data/test/y_test.npy'):
        print('Training and evaluation datasets exist. Skipping Download')
    else:
        print('Downloading training and evaluation dataset')
        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)

        train_dir = os.path.join(os.getcwd(), 'data/train')
        os.makedirs(train_dir, exist_ok=True)

        test_dir = os.path.join(os.getcwd(), 'data/test')
        os.makedirs(test_dir, exist_ok=True)

        data_set = fetch_california_housing()

        X = pd.DataFrame(data_set.data, columns=data_set.feature_names)
        Y = pd.DataFrame(data_set.target)

        # We partition the dataset into 2/3 training and 1/3 test set.
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.33)

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        np.save(os.path.join(train_dir, 'x_train.npy'), x_train)
        np.save(os.path.join(test_dir, 'x_test.npy'), x_test)
        np.save(os.path.join(train_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(test_dir, 'y_test.npy'), y_test)

        print('Downloading completed')


def do_inference_on_local_endpoint(predictor):
    print(f'\nStarting Inference on endpoint (local).')

    x_test = np.load('./data/test/x_test.npy')
    y_test = np.load('./data/test/y_test.npy')

    results = predictor.predict(x_test[:10])['predictions']
    flat_list = [float('%.1f' % (item)) for sublist in results for item in sublist]
    print('predictions: \t{}'.format(np.array(flat_list)))
    print('target values: \t{}'.format(y_test[:10].round(decimals=1)))


def main():
    download_training_and_eval_data()

    print('Starting model training.')
    print(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    california_housing_estimator = TensorFlow(entry_point='california_housing_tf2.py',
                                              source_dir='code',
                                              role=DUMMY_IAM_ROLE,
                                              instance_count=1,
                                              instance_type='local',
                                              framework_version='2.8',
                                              py_version='py39')

    inputs = {'train': 'file://./data/train', 'test': 'file://./data/test'}
    california_housing_estimator.fit(inputs)
    print('Completed model training')

    print('Deploying endpoint in local mode')
    predictor = california_housing_estimator.deploy(initial_instance_count=1, instance_type='local')

    do_inference_on_local_endpoint(predictor)

    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint(predictor.endpoint_name)


if __name__ == "__main__":
    main()
