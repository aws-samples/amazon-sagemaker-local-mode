# This is a sample Python program that trains a simple LightGBM Regression model, and then performs inference.
# This implementation will work on your local computer.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build  -t sagemaker-lightgbm-regression-local container/.
########################################################################################################################

import pandas as pd
from sagemaker.estimator import Estimator
from sagemaker.local import LocalSession
from sagemaker.predictor import csv_serializer
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

# For local training a dummy role will be sufficient
role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

data = load_boston()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=45)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=45)

trainX = pd.DataFrame(X_train, columns=data.feature_names)
trainX['target'] = y_train

valX = pd.DataFrame(X_test, columns=data.feature_names)
valX['target'] = y_test

testX = pd.DataFrame(X_test, columns=data.feature_names)

local_train = './data/train/boston_train.csv'
local_validation = './data/validation/boston_validation.csv'
local_test = './data/test/boston_test.csv'

trainX.to_csv(local_train, header=None, index=False)
valX.to_csv(local_validation, header=None, index=False)
testX.to_csv(local_test, header=None, index=False)

image = 'sagemaker-lightgbm-regression-local'

local_lightgbm = Estimator(
    image,
    role,
    instance_count=1,
    instance_type="local",
    hyperparameters={'boosting_type': 'gbdt',
            'objective': 'regression',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0})

train_location = 'file://'+local_train
validation_location = 'file://'+local_validation
local_lightgbm.fit({'train':train_location, 'validation': validation_location}, logs=True)

predictor = local_lightgbm.deploy(1, 'local', serializer=csv_serializer)

with open(local_test, 'r') as f:
    payload = f.read().strip()

predicted = predictor.predict(payload).decode('utf-8')
print(predicted)

predictor.delete_endpoint(predictor.endpoint)
