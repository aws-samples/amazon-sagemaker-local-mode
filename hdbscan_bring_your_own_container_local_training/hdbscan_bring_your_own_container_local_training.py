# This is a sample Python program that trains a simple HDBSCAN model.
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build  -t sagemaker-hdbscan-local container/.
########################################################################################################################

import os
import boto3
import pickle
import tarfile
import pandas as pd
from sagemaker.estimator import Estimator
from sklearn.datasets import make_blobs


DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'
local_train = './data/train/blobs.csv'
s3 = boto3.resource('s3')


def download_training_and_eval_data():
    if os.path.isfile('./data/train/blobs.csv'):
        print('Training dataset exist. Skipping Download')
    else:
        print('Downloading training dataset')

        os.makedirs("./data", exist_ok=True)
        os.makedirs("./data/train", exist_ok=True)

        blobs, labels = make_blobs(n_samples=2000, n_features=10)
        train_data = pd.DataFrame(blobs)
        train_data.to_csv(local_train, header=None, index=False)

        print('Downloading completed')


def main():
    download_training_and_eval_data()

    print('Starting model training.')
    print('Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')

    image = 'sagemaker-hdbscan-local'

    local_estimator = Estimator(
        image,
        DUMMY_IAM_ROLE,
        instance_count=1,
        instance_type="local",
        hyperparameters={
            "min_cluster_size": 50,
        })

    train_location = 'file://' + local_train

    local_estimator.fit({'train':train_location})
    print('Completed model training')

    model_data = local_estimator.model_data
    print(model_data)


if __name__ == "__main__":
    main()
