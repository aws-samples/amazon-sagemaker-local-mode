# This is a sample Python program that use scikit-learn container to perform average on input.
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
import tarfile
from pathlib import Path

import numpy as np
from sagemaker import LocalSession
from sagemaker.sklearn import SKLearn, SKLearnModel


DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def do_inference_on_local_endpoint(predictor):
    print(f'\nStarting Inference on endpoint (local).')
    inputs = np.array([1, 2, 3, 4 , 5])
    print("Endpoint response: {}".format(predictor.predict(inputs)))


def main():
    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    dummy_model_file = Path("dummy.model")
    dummy_model_file.touch()

    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add(dummy_model_file.as_posix())

    # For local training a dummy role will be sufficient
    role = DUMMY_IAM_ROLE

    model = SKLearnModel(
        role=role,
        model_data='file://./model.tar.gz',
        framework_version='1.2-1',
        py_version='py3',
        source_dir='code',
        entry_point='inference.py'
    )

    print('Deploying endpoint in local mode')
    print(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='local',
    )

    do_inference_on_local_endpoint(predictor)

    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint(predictor.endpoint_name)


if __name__ == "__main__":
    main()
