import os
import json
from gensim.models import KeyedVectors

def input_fn(request_body, request_content_type):
    print(f"request_body: {request_body}")
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        instances = payload["instances"]
        return instances


def predict_fn(instances, word_vectors):
    #########################################
    # Do your custom preprocessing logic here
    #########################################

    print(f"instances: {instances}")
    print("calling model")
    predictions = word_vectors.most_similar(positive=instances)
    return predictions


def model_fn(model_dir):
    print("loading model from: {}".format(model_dir))
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(model_dir, "vectors.txt"), binary=False)
    print(f'word vectors length: {len(word_vectors)}')
    return word_vectors
