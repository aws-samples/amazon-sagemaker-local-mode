import logging
import sys
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("predict_fn")
    logger.info(f"input_object: {input_object}")

    response = np.average(input_object)
    logger.info(f"returning response: {response}")

    return response

# Dummy model_fn function
def model_fn(model_dir):
    dummy_model = {}
    return dummy_model