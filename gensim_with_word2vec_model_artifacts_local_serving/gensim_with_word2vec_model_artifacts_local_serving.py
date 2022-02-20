# This is a sample Python program that serves a Word2Vec model, trained with BlazingText algorithm with inference, using gensim.
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

import boto3
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer
from sagemaker.sklearn import SKLearnModel


DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'
s3 = boto3.client('s3')


def main():

    # Download a pre-trained model archive file
    print('Downloading a pre-trained model archive file')
    s3.download_file('aws-ml-blog', 'artifacts/word2vec_algorithm_model_artifacts/model.tar.gz', 'model.tar.gz')

    print('Deploying endpoint in local mode')
    print(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    model = SKLearnModel(
        role=DUMMY_IAM_ROLE,
        model_data='file://./model.tar.gz',
        framework_version='0.23-1',
        py_version='py3',
        source_dir='code',
        entry_point='inference.py'
    )

    print('Deploying endpoint in local mode')
    predictor = model.deploy(initial_instance_count=1, instance_type='local')

    payload = {"instances": ["dog","cat"]}
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()
    predictions = predictor.predict(payload)
    print(f"Predictions: {predictions}")

    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint(predictor.endpoint_name)


if __name__ == "__main__":
    main()
