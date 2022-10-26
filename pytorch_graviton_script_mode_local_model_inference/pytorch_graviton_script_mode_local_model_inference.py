# This is a sample Python program that inference with a pretrained PyTorch CIFAR-10 model using Graviton instance.
# This implementation will work on your *ARM based local computer*.
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
from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorchModel

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def main():
    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    # For local training a dummy role will be sufficient
    role = DUMMY_IAM_ROLE
    model_dir = 's3://aws-ml-blog/artifacts/pytorch-script-mode-local-model-inference/model.tar.gz'
    region = sagemaker_session.boto_region_name
    print(f'Region: {region}')

    model = PyTorchModel(
        role=role,
        model_data=model_dir,
        image_uri=f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference-graviton:1.12.1-cpu-py38-ubuntu20.04-sagemaker',
        entry_point='inference.py'
    )

    print('Deploying endpoint in local mode')
    print(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='local',
    )

    print('Endpoint deployed in local mode')
    payload = np.random.randn(4, 3, 32, 32).astype(np.float32)

    predictions = predictor.predict(payload)
    print("predictions: {}".format(predictions))

    print('About to delete the endpoint')
    predictor.delete_endpoint(predictor.endpoint)

if __name__ == "__main__":
    main()