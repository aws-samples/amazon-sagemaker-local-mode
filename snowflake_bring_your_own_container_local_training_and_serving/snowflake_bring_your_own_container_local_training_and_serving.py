# This is a sample Python program that trains a simple scikit-learn machine predictive maintenance classification model
# on the dataset fetched from Snowflake, using Snowpark Python package.
#
# Getting Started with Snowpark for Machine Learning on SageMaker:
#  - https://quickstarts.snowflake.com/guide/getting_started_with_snowpark_for_machine_learning_on_sagemaker/index.html
#  - https://github.com/Snowflake-Labs/sfguide-getting-started-snowpark-python-sagemaker
#
# To be able to securely store the database access credentials, we strongly recommend using AWS Secrets Manager with Snowflake connections:
# - https://docs.aws.amazon.com/secretsmanager/latest/userguide/create_secret.html 
# - https://aws.amazon.com/blogs/big-data/simplify-snowflake-data-loading-and-processing-with-aws-glue-databrew/
#
# This implementation will work on your local computer.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build -t sagemaker-scikit-learn-snowpark-local container/.
########################################################################################################################

import os

import numpy as np
import pandas as pd
import sklearn.model_selection
from sagemaker.deserializers import JSONDeserializer
from sagemaker.estimator import Estimator
from sagemaker.serializers import JSONSerializer
from sklearn.datasets import *
from sklearn.preprocessing import StandardScaler


DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def main():
    image = 'sagemaker-scikit-learn-snowpark-local'

    hyperparameters={
        "secret-name": "dev/ml/snowflake",
        "region-name": "us-east-1"
    }

    print('Starting model training.')
    estimator = Estimator(
        image_uri=image,
        entry_point='predictive_maintenance_classification.py',
        source_dir='code',
        role=DUMMY_IAM_ROLE,
        instance_count=1,
        instance_type='local',
        hyperparameters=hyperparameters
    )

    estimator.fit()
    print('Completed model training')


if __name__ == "__main__":
    main()
