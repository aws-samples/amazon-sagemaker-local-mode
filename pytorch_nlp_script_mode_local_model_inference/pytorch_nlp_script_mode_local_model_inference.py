# This is a sample Python program that inference with a pretrained HuggingFace sentiment Analysis model.
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

import pandas as pd
from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorchModel
import sagemaker

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def main():

    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    # For local training a dummy role will be sufficient
    role = DUMMY_IAM_ROLE
    model_dir = 's3://aws-ml-blog/artifacts/pytorch-nlp-script-mode-local-model-inference/model.tar.gz'

    test_data = pd.read_csv('./data/test_data.csv', header=None)
    print(f'test_data: {test_data}')

    model = PyTorchModel(
        role=role,
        model_data=model_dir,
        framework_version='1.7.1',
        source_dir='code',
        py_version='py3',
        entry_point='inference.py'
    )

    print('Deploying endpoint in local mode')
    print(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='local',
    )

    predictor.serializer = sagemaker.serializers.CSVSerializer()
    predictor.deserializer = sagemaker.deserializers.CSVDeserializer()

    predictions = predictor.predict(test_data.to_csv(header=False, index=False))
    print(f'predictions: {predictions}')

    predictor.delete_endpoint(predictor.endpoint)

if __name__ == "__main__":
    main()