# This is a sample Python program that trains a simple TensorFlow CIFAR-10 model.
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

import numpy as np
from sagemaker.tensorflow import TensorFlow
from tensorflow.keras.datasets import cifar10

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def download_training_and_eval_data():
    if os.path.isfile('./data/training/train_data.npy') and \
            os.path.isfile('./data/training/train_labels.npy') and \
            os.path.isfile('./data/validation/validation_data.npy') and \
            os.path.isfile('./data/validation/validation_labels.npy'):
        print('Training and evaluation datasets exist. Skipping Download')
    else:
        print('Downloading training and evaluation dataset')
        (X_train, y_train), (X_valid, y_valid) = cifar10.load_data()

        with open('./data/training/train_data.npy', 'wb') as f:
            np.save(f, X_train)

        with open('./data/training/train_labels.npy', 'wb') as f:
            np.save(f, y_train)

        with open('./data/validation/validation_data.npy', 'wb') as f:
            np.save(f, X_valid)

        with open('./data/validation/validation_labels.npy', 'wb') as f:
            np.save(f, y_valid)

        print('Downloading completed')


def main():
    download_training_and_eval_data()

    print('Starting model training.')
    print(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    mnist_estimator = TensorFlow(entry_point='cifar10_tf2.py',
                                 source_dir='source_dir',
                                 role=DUMMY_IAM_ROLE,
                                 instance_count=1,
                                 instance_type='local',
                                 framework_version='2.4.1',
                                 disable_profiler=True,
                                 debugger_hook_config=False,
                                 py_version='py37')

    inputs = {'training': 'file://./data/training', 'validation': 'file://./data/validation'}
    mnist_estimator.fit(inputs)
    print('Completed model training')


if __name__ == "__main__":
    main()
