# This is a sample Python program that trains a simple PyTorch CIFAR-10 model.
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas matplotlib torch torchvision
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. You should have AWS credentials configured on your local machine
#      in order to be able to pull the docker image from ECR.
##############################################################################################

import os

import numpy as np
import torch
from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorch
from utils_cifar import get_train_data_loader, get_test_data_loader, classes


def download_training_data():
    if os.path.isfile('./data/cifar-10-batches-py/batches.meta') and \
            os.path.isfile('./data/cifar-10-python.tar.gz') :
        print('Training and evaluation datasets exist')
        test_loader = get_test_data_loader(False)
    else:
        print('Downloading training and evaluation dataset')
        test_loader = get_test_data_loader(True)
    return test_loader


def do_inference_on_local_endpoint(predictor, testloader):
    print('Starting Inference on local mode endpoint')
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    outputs = predictor.predict(images.numpy())

    _, predicted = torch.max(torch.from_numpy(np.array(outputs)), 1)

    print('Predicted: ', ' '.join('%4s' % classes[predicted[j]]
                                  for j in range(4)))

def main():
    test_loader = download_training_data()

    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    # For local training a dummy role will be sufficient
    role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

    print('Starting model training')
    print('Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    cifar10_estimator = PyTorch(entry_point='cifar10_pytorch.py',
                                source_dir='./code',
                                role=role,
                                framework_version='1.10',
                                py_version='py38',
                                instance_count=1,
                                instance_type='local',
                                hyperparameters={
                                    'epochs': 1,
                                })

    cifar10_estimator.fit('file://./data/')

    print('Deploying local mode endpoint')
    predictor = cifar10_estimator.deploy(initial_instance_count=1, instance_type='local')

    do_inference_on_local_endpoint(predictor, test_loader)

    predictor.delete_endpoint(predictor.endpoint)

if __name__ == "__main__":
    main()