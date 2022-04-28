"""
ModelHandler defines an example model handler for load and inference requests for TensorFlow CPU models
"""
import json
import os

import pandas as pd
import tensorflow as tf


class ModelHandler(object):
    """
    A sample Model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.tf_model = None
        self.shapes = None


    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        print('initialize')
        self.initialized = True
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir")
        print(f'model_dir: {model_dir}')

        for currentpath, folders, files in os.walk(model_dir):
            print(currentpath, folders, files)

        gpu_id = properties.get("gpu_id")
        print(f'gpu_id: {gpu_id}')

        self.tf_model = tf.keras.models.load_model(model_dir+'/1/')
        print('Model Loaded')


    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        print(f'handle')

        payload = data[0]["body"].decode()
        instances = json.loads(payload)["instances"]
        print(f"instances: {instances}")
        payload = pd.DataFrame(data=instances)
        print('Invoked with {} records'.format(payload.shape[0]))

        predictions = self.tf_model.predict(payload)

        # Convert from numpy back to JSON
        predictions_lists = predictions.tolist()
        print(f"Returning {len(predictions_lists)} predictions")
        result = [[{"predictions": predictions_lists}]]
        return result

_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
