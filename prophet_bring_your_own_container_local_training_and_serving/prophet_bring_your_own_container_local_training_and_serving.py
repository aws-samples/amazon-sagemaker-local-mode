# This is a sample Python program that trains a simple Prophet Forecasting model, and then performs inference.
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build  -t sagemaker-prophet-local container/.
########################################################################################################################

import pandas as pd
from sagemaker.estimator import Estimator
from sagemaker.local import LocalSession
from sagemaker.predictor import csv_serializer

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

# For local training a dummy role will be sufficient
role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

image = 'sagemaker-prophet-local'
print(image)

local_tseries = Estimator(
    image,
    role,
    instance_count=1,
    instance_type="local")

local_tseries.fit('file://./data/')

local_predictor = local_tseries.deploy(1, 'local', serializer=csv_serializer)

predicted = local_predictor.predict("30").decode('utf-8')
print(predicted)

local_predictor.delete_endpoint()
local_predictor.delete_model()

