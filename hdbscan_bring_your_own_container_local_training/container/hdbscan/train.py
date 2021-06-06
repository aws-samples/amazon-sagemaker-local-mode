#!/usr/bin/env python

# A sample training component that trains a simple HDBSCAN model.
# This implementation works in File mode and makes no assumptions about the input file names.

import json
import os
import pickle
import sys
import traceback

import hdbscan
import pandas as pd

prefix = '/opt/ml/'
input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
train_channel_name = 'train'
validation_channel_name = 'validation'

model_path = os.path.join(prefix, 'model')
model_file_name = 'hdbscan-model.pkl'
train_path = os.path.join(input_path, train_channel_name)
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
        raw_data = [pd.read_csv(file, header=None) for file in train_input_files]
        train_df = pd.concat(raw_data)

        min_cluster_size = hyperparameters_data.get('min_cluster_size', None)
        if min_cluster_size is not None:
            min_cluster_size = int(min_cluster_size)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                cluster_selection_method='eom')
        print("Start HDBSCAN clustering...")
        clusterer = clusterer.fit(train_df)

        labels = clusterer.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("Clustering finished, found", n_clusters_, 'clusters')
        print(n_noise_, "samples marked as noise (not in any cluster)")

        # save the model
        with open(os.path.join(model_path, 'hdbscan-model.pkl'), 'wb') as out:
            pickle.dump(clusterer, out)

        print("model {} saved.".format('hdbscan-model.pkl'))
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
