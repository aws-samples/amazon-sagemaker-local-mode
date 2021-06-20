# This is a sample Python program that trains a BYOC TensorFlow model, and then performs inference.
# This implementation will work on your local computer.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build  -t sagemaker-tensorflow2-batch-transform-local container/.
########################################################################################################################

import os

import pandas as pd
import sklearn.model_selection
from sagemaker.estimator import Estimator
from sklearn.datasets import *
from sklearn.preprocessing import StandardScaler

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def download_training_and_eval_data():
    if os.path.isfile('./data/train/x_train.csv') and \
            os.path.isfile('./data/test/x_test.csv') and \
            os.path.isfile('./data/train/y_train.csv') and \
            os.path.isfile('./data/test/y_test.csv'):
        print('Training and evaluation datasets exist. Skipping Download')
    else:
        print('Downloading training and evaluation dataset')
        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)

        train_dir = os.path.join(os.getcwd(), 'data/train')
        os.makedirs(train_dir, exist_ok=True)

        test_dir = os.path.join(os.getcwd(), 'data/test')
        os.makedirs(test_dir, exist_ok=True)

        input_dir = os.path.join(os.getcwd(), 'data/input')
        os.makedirs(input_dir, exist_ok=True)

        output_dir = os.path.join(os.getcwd(), 'data/output')
        os.makedirs(output_dir, exist_ok=True)

        data_set = fetch_california_housing()

        X = pd.DataFrame(data_set.data, columns=data_set.feature_names)
        Y = pd.DataFrame(data_set.target)

        # We partition the dataset into 2/3 training and 1/3 test set.
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.33)

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        pd.DataFrame(x_train).to_csv(os.path.join(train_dir, 'x_train.csv'), header=None, index=False)
        pd.DataFrame(x_test).to_csv(os.path.join(test_dir, 'x_test.csv'),header=None, index=False)
        pd.DataFrame(x_test).to_csv(os.path.join(input_dir, 'x_test.csv'),header=None, index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(train_dir, 'y_train.csv'), header=None, index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(test_dir, 'y_test.csv'), header=None, index=False)

        print('Downloading completed')


def main():
    download_training_and_eval_data()

    image = 'sagemaker-tensorflow2-batch-transform-local'

    print('Starting model training.')
    california_housing_estimator = Estimator(
        image,
        DUMMY_IAM_ROLE,
        hyperparameters={'epochs': 10,
                         'batch_size': 64,
                         'learning_rate': 0.1},
        instance_count=1,
        instance_type="local")

    inputs = {'train': 'file://./data/train', 'test': 'file://./data/test'}
    california_housing_estimator.fit(inputs, logs=True)
    print('Completed model training')

    print('Running Batch Transform in local mode')
    tensorflow_serving_transformer = california_housing_estimator.transformer(
        instance_count=1,
        instance_type='local',
        output_path='file:./data/output',
    )

    tensorflow_serving_transformer.transform('file://./data/input',
                                             split_type='Line',
                                             content_type='text/csv')

    print('Printing Batch Transform output file content')
    output_file = open('./data/output/x_test.csv.out', 'r').read()
    print(output_file)


if __name__ == "__main__":
    main()
