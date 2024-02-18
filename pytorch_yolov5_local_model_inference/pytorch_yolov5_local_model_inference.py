# This is a sample Python program for deploying a YOLOV5 pre-trained model to a SageMaker endpoint.
# Inference is done with a URL of the image, which is the http payload for the SageMaker Endpoint.
# This implementation will work on your *local computer*.
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
from sagemaker.deserializers import JSONDeserializer
from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def main():
    session = LocalSession()
    session.config = {'local': {'local_code': True}}

    role = DUMMY_IAM_ROLE
    model_dir = 's3://aws-ml-blog/artifacts/pytorch-yolov5-local-model-inference/model.tar.gz'

    model = PyTorchModel(
        entry_point='inference.py',
        source_dir = './code',
        role=role,
        model_data=model_dir,
        framework_version='2.1',
        py_version='py310'
    )

    print('Deploying endpoint in local mode')
    print(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='local',
    )

    print('Endpoint deployed in local mode')

    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()
    predictions = predictor.predict("https://ultralytics.com/images/zidane.jpg")
    print("predictions: {}".format(predictions))

    print('About to delete the endpoint')
    predictor.delete_endpoint()


if __name__ == "__main__":
    main()
