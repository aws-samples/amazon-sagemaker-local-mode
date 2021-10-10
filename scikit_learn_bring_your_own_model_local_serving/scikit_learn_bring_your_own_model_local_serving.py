# This is a sample Python program that serve a scikit-learn model pre-trained on the California Housing dataset.
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
import pandas as pd
import tarfile

from sagemaker.sklearn import SKLearnModel
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'
s3 = boto3.client('s3')


def main():

    # Prepare data for model inference - we use the Boston housing dataset
    print('Preparing data for model inference')
    data = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42
    )

    # we don't train a model, so we will need only the testing data
    testX = pd.DataFrame(X_test, columns=data.feature_names)

    # Download a pre-trained model file
    print('Downloading a pre-trained model file')
    s3.download_file('aws-ml-blog', 'artifacts/scikit_learn_bring_your_own_model/model.joblib', 'model.joblib')

    # Creating a model.tar.gz file
    tar = tarfile.open('model.tar.gz', 'w:gz')
    tar.add('model.joblib')
    tar.close()

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

    predictions = predictor.predict(testX[data.feature_names])
    print("Predictions: {}".format(predictor.predict(testX.values)))

    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint(predictor.endpoint_name)


if __name__ == "__main__":
    main()
