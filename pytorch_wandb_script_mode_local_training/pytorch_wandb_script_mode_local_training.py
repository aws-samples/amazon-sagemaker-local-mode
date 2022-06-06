# This is a sample Python program that trains a simple PyTorch MNIST model,
# integrated with Weights & Biases to track experiments orchestrated.
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas matplotlib torch torchvision==0.9.1
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. You should have AWS credentials configured on your local machine
#      in order to be able to pull the docker image from ECR.
##############################################################################################

import os

from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorch
from torchvision import transforms
from torchvision.datasets import MNIST


def download_training_data():

    if os.path.isdir('./data/MNIST') :
        print('Training and evaluation datasets exist')
    else:
        print('Downloading training and evaluation dataset')
        MNIST.mirrors = ["https://sagemaker-sample-files.s3.amazonaws.com/datasets/image/MNIST/"]

        MNIST(
            'data',
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        )


def main():
    download_training_data()

    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    # For local training a dummy role will be sufficient
    role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

    # Go to https://wandb.ai to signup for a free account.
    # After which you could go to https://wandb.ai/authorize to get your API key!
    current_api_key = "<YOUR API KEY>"

    print('Starting model training')
    print('Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    mnist_estimator = PyTorch(entry_point='mnist.py',
                                source_dir='./code',
                                role=role,
                                framework_version='1.8',
                                py_version='py3',
                                instance_count=1,
                                instance_type='local',
                                hyperparameters={
                                    'epochs': 1,
                                    'backend': 'gloo'
                                },
                                environment={"WANDB_API_KEY": current_api_key})

    mnist_estimator.fit({'training': 'file://./data/'})


if __name__ == "__main__":
    main()