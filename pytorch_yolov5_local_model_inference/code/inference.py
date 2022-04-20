import os.path
import torch
import json


def model_fn(model_dir):
    model_path = os.path.join(model_dir,'yolov5s.pt')
    print(f'model_fn - model_path: {model_path}')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    return model


def input_fn(serialized_input_data, content_type):
    if content_type == 'application/json':
        print(f'input_fn - serialized_input_data: {serialized_input_data}')
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return


def predict_fn(input_data, model):
    print(f'predict_fn - input_data: {input_data}')
    imgs = [input_data]
    results = model(imgs)
    df = results.pandas().xyxy[0]
    return(df.to_json(orient="split"))
