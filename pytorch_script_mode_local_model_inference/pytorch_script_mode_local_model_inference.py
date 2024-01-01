# This is a sample Python program that inference with a pretrained PyTorch CIFAR-10 model.
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
from sagemaker.pytorch import PyTorch, PyTorchModel
from utils_cifar import get_train_data_loader, get_test_data_loader, classes

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def download_data_for_inference():
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
    test_loader = download_data_for_inference()

    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    # For local training a dummy role will be sufficient
    role = DUMMY_IAM_ROLE
    model_dir = 's3://aws-ml-blog/artifacts/pytorch-script-mode-local-model-inference/model.tar.gz'

    model = PyTorchModel(
        role=role,
        model_data=model_dir,
        framework_version='2.1',
        py_version='py310',
        entry_point='inference.py'
    )

    print('Deploying endpoint in local mode')
    print(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='local',
    )

    do_inference_on_local_endpoint(predictor, test_loader)

    predictor.delete_endpoint(predictor.endpoint)

if __name__ == "__main__":
    main()