# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os

import flask
import pandas as pd
import tensorflow as tf

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class CaliforniaHousingService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(chs):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if chs.model == None:
            print('Loading Model')
            chs.model = tf.keras.models.load_model('/opt/ml/model/1/')
        return chs.model

    @classmethod
    def predict(chs, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        model = chs.get_model()
        return model.predict(input)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = CaliforniaHousingService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as JSON, convert
    it to a pandas dataframe for internal use and then convert the predictions back to JSON (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from JSON to Pandas
    if flask.request.content_type == 'application/json':
        data = flask.request.data.decode('utf-8')
        instances = json.loads(data)["instances"]
        print(f"instances: {instances}")
        data = pd.DataFrame(data=instances)
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    predictions = CaliforniaHousingService.predict(data)

    # Convert from numpy back to JSON
    predictions_lists = predictions.tolist()
    print(f"Returning {len(predictions_lists)} predictions")
    result = json.dumps({"predictions": predictions_lists})

    return flask.Response(response=result, status=200, mimetype='application/json')
