# This is a sample Python program for deploying a trained model to a SageMaker endpoint.
# Inference is done with a file in S3 instead of http payload for the SageMaker Endpoint.
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#      `pip install -r requirements.txt`
#   2. Install boto3 on `libs` folder for the Tensor Flow Model object to use for inference:
#      `pip install boto3 -t ./libs`
#   3. Docker Desktop installed and running on your computer:
#      `docker ps`
#   4. You should have AWS credentials configured on your local machine
#      in order to be able to pull the docker image from ECR.
###############################################################################################

from sagemaker.local import LocalSession
from sagemaker.tensorflow import TensorFlow, TensorFlowModel

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def main():
    session = LocalSession()
    session.config = {'local': {'local_code': True}}

    role = DUMMY_IAM_ROLE
    model_dir = 's3://tensorflow-script-mode-local-model-inference/model.tar.gz'

    model = TensorFlowModel(
        entry_point='inference.py',
        dependencies=['./libs/'],
        role=role,
        model_data=model_dir,
        framework_version='2.3.0',
    )

    print('Deploying endpoint in local mode')
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='local',
    )

    dummy_inputs = {
        'bucket_name': 'tensorflow-script-mode-local-model-inference',
        'object_name': 'instances.json'
    }

    results = predictor.predict(dummy_inputs)
    print("results: {}".format(results))

    print('About to delete the endpoint')
    predictor.delete_endpoint(predictor.endpoint_name)
    predictor.delete_model()


if __name__ == "__main__":
    main()
