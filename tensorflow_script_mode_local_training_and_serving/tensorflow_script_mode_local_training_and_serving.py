# This is a sample Python program that trains a simple TensorFlow CIFAR-10 model.
# This implementation will work on your *local computer* or in the *AWS Cloud*.
# To run training and inference *locally* set: `config = get_config(LOCAL_MODE)`
# To run training and inference on the *cloud* set: `config = get_config(CLOUD_MODE)` and set a valid IAM role value in get_config()
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

LOCAL_MODE = 'LOCAL_MODE'
CLOUD_MODE = 'CLOUD_MODE'

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


def do_inference_on_local_endpoint(predictor, mode):
    print(f'\nStarting Inference on endpoint ({mode}).')
    correct_predictions = 0

    train_data = np.load('./data/train_data.npy')
    train_labels = np.load('./data/train_labels.npy')

    predictions = predictor.predict(train_data[:50])
    for i in range(0, 50):
        prediction = np.argmax(predictions['predictions'][i])
        label = train_labels[i]
        print('prediction is {}, label is {}, matched: {}'.format(prediction, label, prediction == label))
        if prediction == label:
            correct_predictions = correct_predictions + 1

    print('Calculated Accuracy from predictions: {}'.format(correct_predictions / 50))


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


def get_config(mode):
    assert mode is CLOUD_MODE or mode is LOCAL_MODE, f'unknown mode selected: {mode}'

    if mode == CLOUD_MODE:
        ## REPLACE WITH A VALID IAM ROLE - START ##
        role = DUMMY_IAM_ROLE
        ## REPLACE WITH A VALID IAM ROLE - END ##
        assert role is not DUMMY_IAM_ROLE, "For cloud mode set a valid sagemaker iam role"

        print('Will run training on an ML instance in AWS.')
        session = sagemaker.Session()
        bucket = session.default_bucket()
        s3_data_prefix = 'tensorflow_script_mode_cloud_training/mnist/'
        instance_type = 'ml.m5.large'
        training_dataset_path = 's3://' + bucket + '/' + s3_data_prefix

    else:  # mode == LOCAL_MODE
        print('Will run training locally in a container image.')
        session = LocalSession()
        session.config = {'local': {'local_code': True}}
        instance_type = 'local'
        training_dataset_path = "file://./data/"
        role = DUMMY_IAM_ROLE  # not needed in local training
        s3_data_prefix = None  # not needed in local training
        bucket = None  # not needed in local training

    config = {
        'mode': mode,
        's3_data_prefix': s3_data_prefix,
        'sagemaker_session': session,
        'bucket': bucket,
        'instance_type': instance_type,
        'training_dataset_path': training_dataset_path,
        'role': role}
    return config


def main():
    config = get_config(LOCAL_MODE)
    #config = get_config(CLOUD_MODE)

    download_training_and_eval_data()

    if config['mode'] is CLOUD_MODE:
        upload_data_to_s3(config['bucket'], config['s3_data_prefix'])

    print('Starting model training.')
    print(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    mnist_estimator = TensorFlow(entry_point='mnist_tf2.py',
                                 source_dir='code',
                                 role=config['role'],
                                 instance_count=1,
                                 instance_type=config['instance_type'],
                                 framework_version='2.4.1',
                                 py_version='py37',
                                 distribution={'parameter_server': {'enabled': True}})

    mnist_estimator.fit(config['training_dataset_path'])
    print('Completed model training')

    print('Deploying endpoint in ' + config['mode'])
    predictor = mnist_estimator.deploy(initial_instance_count=1, instance_type=config['instance_type'])

    do_inference_on_local_endpoint(predictor, config['mode'])

    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint(predictor.endpoint_name)

if __name__ == "__main__":
    main()
