# This is a sample Python program that serve a scikit-learn model pre-trained on the California Housing dataset with your own Docker container.
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build  -t sagemaker-sklearn-rf-regressor-local container/.
########################################################################################################################
import tarfile

import boto3
import pandas as pd
from sagemaker import Model, LocalSession
from sagemaker.deserializers import CSVDeserializer
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'
s3 = boto3.client('s3')


def main():

    image_name = "sagemaker-sklearn-rf-regressor-local"

    # Prepare data for model inference - we use the Boston housing dataset
    print('Preparing data for model inference')
    data = fetch_california_housing()
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

    model = Model(
        image_uri=image_name,
        role=DUMMY_IAM_ROLE,
        model_data='file://./model.tar.gz'
    )

    print('Deploying endpoint in local mode')
    endpoint = model.deploy(initial_instance_count=1,
                            instance_type='local',
                            endpoint_name="my-local-endpoint")

    predictor = Predictor(endpoint_name="my-local-endpoint",
                          sagemaker_session=sagemaker_session,
                          serializer=CSVSerializer(),
                          deserializer=CSVDeserializer())

    predictions = predictor.predict(testX[data.feature_names].head(5).to_csv(header=False, index=False))
    print(f"Predictions: {predictions}")

    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint(predictor.endpoint_name)


if __name__ == "__main__":
    main()
