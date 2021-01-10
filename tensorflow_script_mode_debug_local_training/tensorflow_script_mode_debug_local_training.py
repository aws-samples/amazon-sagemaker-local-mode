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

import boto3
import numpy as np
import sagemaker.session
from sagemaker.local import LocalSession
from sagemaker.tensorflow import TensorFlow


DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'
data_files_list = ('train_data.npy', 'train_labels.npy', 'eval_data.npy', 'eval_labels.npy')


def download_training_and_eval_data():
    if os.path.isfile('./data/train_data.npy') and \
            os.path.isfile('./data/train_labels.npy') and \
            os.path.isfile('./data/eval_data.npy') and \
            os.path.isfile('./data/eval_labels.npy'):
        print('Training and evaluation datasets exist. Skipping Download')
    else:
        print('Downloading training and evaluation dataset')
        s3 = boto3.resource('s3')
        for filename in data_files_list:
            s3.meta.client.download_file('sagemaker-sample-data-us-east-1', 'tensorflow/mnist/' + filename,
                                         './data/' + filename)


def upload_data_to_s3(bucket, prefix):
    # Required if running in cloud mode. Skips upload if file exist in S3
    s3 = boto3.resource('s3')
    result = s3.meta.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    existing_files = [item['Key'] for item in result['Contents']] if 'Contents' in result else []
    for filename in data_files_list:
        if prefix + filename not in existing_files:
            print('Uploading ' + filename + ' to s3://' + bucket + '/' + prefix + filename)
            s3.meta.client.upload_file('./data/' + filename, bucket, prefix + filename)
        else:
            print('File already in bucket. Skipping uploading for: ' + prefix + filename)


def get_config():

    print('Will run training locally in a container image.')
    session = LocalSession()
    session.config = {'local': {'local_code': True}}
    instance_type = 'local'
    training_dataset_path = "file://./data/"
    role = DUMMY_IAM_ROLE  # not needed in local training
    s3_data_prefix = None  # not needed in local training
    bucket = None  # not needed in local training

    config = {
        's3_data_prefix': s3_data_prefix,
        'sagemaker_session': session,
        'bucket': bucket,
        'instance_type': instance_type,
        'training_dataset_path': training_dataset_path,
        'role': role}
    return config


def main():
    config = get_config()

    download_training_and_eval_data()

    print('Starting model training.')
    print(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    mnist_estimator = TensorFlow(entry_point='mnist_tf2.py',
                                 source_dir='./source_dir',
                                 role=config['role'],
                                 instance_count=1,
                                 instance_type=config['instance_type'],
                                 framework_version='2.3.0',
                                 py_version='py37',
                                 distribution={'parameter_server': {'enabled': True}})

    mnist_estimator.fit(config['training_dataset_path'])
    print('Completed model training')


if __name__ == "__main__":
    main()
