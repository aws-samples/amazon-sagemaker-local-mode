
# A sample training component that trains a simple CatBoost Regressor tree model.
# This implementation works in File mode and makes no assumptions about the input file names.
# Input is specified as CSV with a data point in each row and the labels in the first column.

import logging
import os
import json
import traceback
import sys

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

prefix = '/opt/ml/'
input_path = prefix + 'input/data'
train_channel_name = 'train'
validation_channel_name = 'validation'

output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
model_file_name = 'catboost-regressor-model.dump'
train_path = os.path.join(input_path, train_channel_name)
validation_path = os.path.join(input_path, validation_channel_name)

param_path = os.path.join(prefix, 'input/config/hyperparameters.json')


# The function to execute the training.
def train():
    print('Starting the training.')

    try:
        # Read in any hyperparameters that the user passed with the training job
        print('Reading hyperparameters data: {}'.format(param_path))
        with open(param_path) as json_file:
            hyperparameters_data = json.load(json_file)
        print('hyperparameters_data: {}'.format(hyperparameters_data))

        # Take the set of train files and read them all into a single pandas dataframe
        train_input_files = [os.path.join(train_path, file) for file in os.listdir(train_path)]
        if len(train_input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(train_path, train_channel_name))
        print('Found train files: {}'.format(train_input_files))
        raw_data = [pd.read_csv(file) for file in train_input_files]
        train_df = pd.concat(raw_data)

        # Take the set of train files and read them all into a single pandas dataframe
        validation_input_files = [os.path.join(validation_path, file) for file in os.listdir(validation_path)]
        if len(validation_input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(validation_path, train_channel_name))
        print('Found validation files: {}'.format(validation_input_files))
        raw_data = [pd.read_csv(file) for file in validation_input_files]
        validation_df = pd.concat(raw_data)

        # Assumption is that the label is the last column
        print('building training and validation datasets')
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1:].values
        X_validation = validation_df.iloc[:, :-1].values
        y_validation = validation_df.iloc[:, -1:].values

        # define and train model
        model = CatBoostRegressor()

        model.fit(X_train, y_train, eval_set=(X_validation, y_validation), logging_level='Silent')

        # print abs error
        print('validating model')
        abs_err = np.abs(model.predict(X_validation) - y_validation)

        # print couple perf metrics
        for q in [10, 50, 90]:
            print('AE-at-' + str(q) + 'th-percentile: '+ str(np.percentile(a=abs_err, q=q)))

        # persist model
        path = os.path.join(model_path, model_file_name)
        print('saving model file to {}'.format(path))
        model.save_model(path)

        print('Training complete.')
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc)
        # A non-zero exit dependencies causes the training job to be marked as Failed.
        sys.exit(255)

if __name__ == '__main__':
    train()

    # A zero exit dependencies causes the job to be marked a Succeeded.
    sys.exit(0)
