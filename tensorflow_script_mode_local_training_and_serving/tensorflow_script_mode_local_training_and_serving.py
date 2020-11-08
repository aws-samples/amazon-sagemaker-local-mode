# This is a sample Python program that trains a simple TensorFlow CIFAR-10 model.
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas tensorflow
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
##############################################################################################

import os
import boto3
import numpy as np
from sagemaker.local import LocalSession
from sagemaker.tensorflow import TensorFlow


def download_training_and_eval_data():
    if os.path.isfile('./data/train_data.npy') and \
            os.path.isfile('./data/train_labels.npy') and \
            os.path.isfile('./data/eval_data.npy') and \
            os.path.isfile('./data/eval_labels.npy'):
        print('Training and evaluation datasets exist')
    else:
        print('Downloading training and evaluation dataset')
        s3 = boto3.resource('s3')
        s3.meta.client.download_file('sagemaker-sample-data-us-east-1', 'tensorflow/mnist/train_data.npy',
                                     './data/train_data.npy')
        s3.meta.client.download_file('sagemaker-sample-data-us-east-1', 'tensorflow/mnist/train_labels.npy',
                                     './data/train_labels.npy')
        s3.meta.client.download_file('sagemaker-sample-data-us-east-1', 'tensorflow/mnist/eval_data.npy',
                                     './data/eval_data.npy')
        s3.meta.client.download_file('sagemaker-sample-data-us-east-1', 'tensorflow/mnist/eval_labels.npy',
                                     './data/eval_labels.npy')


def do_inference_on_local_endpoint(predictor):
    print('Starting Inference on local mode endpoint')
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

    print('Calculated Accuracy from predictions: {}'.format(correct_predictions/50))


def main():
    download_training_and_eval_data()

    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    # For local training a dummy role will be sufficient
    role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

    print('Starting model training')
    mnist_estimator = TensorFlow(entry_point='mnist_tf2.py',
                                 role=role,
                                 instance_count=1,
                                 instance_type='local',
                                 framework_version='2.3.0',
                                 py_version='py37',
                                 distribution={'parameter_server': {'enabled': True}})

    mnist_estimator.fit("file://./data/")

    print('Deploying local mode endpoint')
    predictor = mnist_estimator.deploy(initial_instance_count=1, instance_type='local')

    do_inference_on_local_endpoint(predictor)

    predictor.delete_endpoint(predictor.endpoint)
    predictor.delete_model()


if __name__ == "__main__":
    main()
