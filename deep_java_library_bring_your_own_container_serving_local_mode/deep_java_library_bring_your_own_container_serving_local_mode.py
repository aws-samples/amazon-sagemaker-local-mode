# This is a sample Python program that performs inference with Deep Java Library (DJL).
# Example was referenced from: https://docs.djl.ai/jupyter/load_pytorch_model.html
# This implementation will work on your local computer.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build  -t sagemaker-djl-serving-local ./container/.
########################################################################################################################

import os

from sagemaker import Model, LocalSession, Predictor
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}


def main():

    image = 'sagemaker-djl-serving-local'
    endpoint_name = "my-local-endpoint"

    role = DUMMY_IAM_ROLE

    model = Model(
        image_uri=image,
        role=role,
        model_data="s3://aws-ml-blog/artifacts/deep-java-library-bring-your-own-container-serving/model.tar.gz",
    )

    print('Deploying endpoint in local mode')
    print(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')

    model.deploy(
        initial_instance_count=1,
        instance_type='local',
        endpoint_name=endpoint_name
    )

    predictor = Predictor(endpoint_name=endpoint_name,
                          sagemaker_session=sagemaker_session,
                          serializer=JSONSerializer(),
                          deserializer=JSONDeserializer())

    predictions = predictor.predict("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg")
    print(f'predictions: {predictions}')

    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint()


if __name__ == "__main__":
    main()
