# This is a sample Python program that trains a simple CatBoost Regressor tree model
# on the california-housing dataset fetched from Delta Lake, directly from S3, and then performs inference.
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build  -t sagemaker-delta-lake-training-local container/.
########################################################################################################################

import pandas as pd
from sagemaker.estimator import Estimator
from sagemaker.local import LocalSession
from sagemaker.predictor import csv_serializer

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

# For local training a dummy role will be sufficient
role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

image = 'sagemaker-delta-lake-training-local'

print('Starting model training.')
local_regressor = Estimator(
    image,
    role,
    instance_count=1,
    instance_type="local")

train_location = "s3://aws-ml-blog/artifacts/delta-lake-bring-your-own-container/delta-table/california-housing/"
local_regressor.fit({'train':train_location}, logs=True)

print('Deploying endpoint in local mode')
predictor = local_regressor.deploy(1, 'local', serializer=csv_serializer)

payload = "-122.230003,37.880001,41.0,880.0,129.0,322.0,126.0,8.3252"
predicted = predictor.predict(payload).decode('utf-8')
print(f'Prediction: {predicted}')

print('About to delete the endpoint to stop paying (if in cloud mode).')
predictor.delete_endpoint()
