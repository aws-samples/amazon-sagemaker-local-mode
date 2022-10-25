# This is a sample Python program for deploying a trained model to a SageMaker endpoint.
# This implementation will work on your *ARM based local computer*.
#
# This example is based on: https://github.com/aws/amazon-sagemaker-examples/blob/master/frameworks/tensorflow/get_started_mnist_deploy.ipynb
#
# Prerequisites:
#   1. Install required Python packages:
#      `pip install -r requirements.txt`
#   2. Docker Desktop installed and running on your computer:
#      `docker ps`
#   3. You should have AWS credentials configured on your local machine
#      in order to be able to pull the docker image from ECR.
###############################################################################################

from sagemaker.local import LocalSession
from sagemaker.tensorflow import TensorFlowModel
import numpy as np

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def main():
    session = LocalSession()
    session.config = {'local': {'local_code': True}}

    role = DUMMY_IAM_ROLE
    model_dir = 's3://aws-ml-blog/artifacts/run-ml-inference-on-graviton-based-instances-with-amazon-sagemaker/model.tar.gz'
    region = session.boto_region_name
    print(f'Region: {region}')

    model = TensorFlowModel(
        role=role,
        model_data=model_dir,
        image_uri=f'763104351884.dkr.ecr.{region}.amazonaws.com/tensorflow-inference-graviton:2.9.1-cpu-py38-ubuntu20.04-sagemaker'
    )

    print('Deploying endpoint in local mode')
    print(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='local',
    )

    print('Endpoint deployed in local mode')
    payload = np.random.randn(1, 32, 32, 3)

    predictions = predictor.predict(payload)
    print("predictions: {}".format(predictions))

    print('About to delete the endpoint')
    predictor.delete_endpoint(predictor.endpoint_name)


if __name__ == "__main__":
    main()
