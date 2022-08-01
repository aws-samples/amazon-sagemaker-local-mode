# This is a sample Python program that use scikit-learn container to perfrom inference using 4 models.
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

import numpy as np
import sagemaker
from sagemaker import LocalSession
from sagemaker.sklearn import SKLearnModel
import boto3
import tarfile


DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'
MAX_YEAR = 2022

def gen_price(house):
    _base_price = int(house["SQUARE_FEET"] * 150)
    _price = int(
        _base_price
        + (10000 * house["NUM_BEDROOMS"])
        + (15000 * house["NUM_BATHROOMS"])
        + (15000 * house["LOT_ACRES"])
        + (15000 * house["GARAGE_SPACES"])
        - (5000 * (MAX_YEAR - house["YEAR_BUILT"]))
    )
    return _price

def gen_random_house():
    _house = {
        "SQUARE_FEET": int(np.random.normal(3000, 750)),
        "NUM_BEDROOMS": np.random.randint(2, 7),
        "NUM_BATHROOMS": np.random.randint(2, 7) / 2,
        "LOT_ACRES": round(np.random.normal(1.0, 0.25), 2),
        "GARAGE_SPACES": np.random.randint(0, 4),
        "YEAR_BUILT": min(MAX_YEAR, int(np.random.normal(1995, 10))),
    }
    _price = gen_price(_house)
    return [
        _price,
        _house["YEAR_BUILT"],
        _house["SQUARE_FEET"],
        _house["NUM_BEDROOMS"],
        _house["NUM_BATHROOMS"],
        _house["LOT_ACRES"],
        _house["GARAGE_SPACES"],
    ]


def main():

    print('Downloading model file from S3')
    s3 = boto3.client('s3')
    s3.download_file('aws-ml-blog', 'artifacts/scikit_learn_serving_multiple_models_with_one_invocation/model.tar.gz', 'model.tar.gz')
    print('Model downloaded')

    tarf = tarfile.open('model.tar.gz', 'r:gz')
    print(f'Found model files in model.tar.gz: {tarf.getnames()}')

    model = SKLearnModel(
        role=DUMMY_IAM_ROLE,
        model_data='file://model.tar.gz',
        framework_version='0.23-1',
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

    predictor.serializer = sagemaker.serializers.JSONSerializer()
    predictor.deserializer = sagemaker.deserializers.JSONDeserializer()

    payload = gen_random_house()[1:]

    predictions = predictor.predict(payload)
    print(f'predictions: {predictions}')

    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint(predictor.endpoint_name)


if __name__ == "__main__":
    main()
