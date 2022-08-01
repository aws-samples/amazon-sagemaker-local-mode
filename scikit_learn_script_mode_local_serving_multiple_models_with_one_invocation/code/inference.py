import logging
import sys
import numpy as np
import os
import joblib
import glob
import json

JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def input_fn(serialized_input_data, content_type):
    logger.info(f'input_fn - serialized_input_data: {serialized_input_data}, content_type: {content_type}')
    
    if content_type == JSON_CONTENT_TYPE:
        payload = [np.array(json.loads(serialized_input_data))]
        return payload
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return


# Perform prediction on the deserialized object, with the loaded models, and returns the max result
def predict_fn(input_object, models_list):
    logger.info("predict_fn")
    logger.info(f"predict_fn - input_object: {input_object}")

    max_prediction = 0
    for i in range(len(models_list)):
        model = models_list[i]
        prediction = model.predict(input_object)
        logger.info(f"predict_fn - result for model #{i}: {prediction}")
        if prediction > max_prediction:
            max_prediction = prediction
        
    logger.info(f"returning response: {max_prediction}")
    return max_prediction


# Load the model files from model_dir
def model_fn(model_dir):
    logger.info(f'model_fn - model_dir: {model_dir}')
    for file in glob.glob(model_dir+'/model_*', recursive=True):
        print(file)

    logger.info(f"model_fn - loading models from: {model_dir}")

    models_list = []
    for model_file in glob.glob(model_dir+'/model_*', recursive=True):
        print(f'Loading model file: {model_file}')
        loaded_model = joblib.load(model_file)
        models_list.append(loaded_model)

    logger.info(f"model_fn - models_list length: {len(models_list)}")
    
    return models_list