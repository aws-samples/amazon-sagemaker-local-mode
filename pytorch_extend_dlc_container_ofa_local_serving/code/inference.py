import glob
import io

import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAForConditionalGeneration

NPY_CONTENT_TYPE = 'application/x-npy'


class OFAImageCaptionPredictor(object):
    def __init__(self, model_dir):
        self.model = OFAForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = OFATokenizer.from_pretrained(model_dir)

    def patch_resize_transform(self, image):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        resolution = 256
        transform_func = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return transform_func(image)

    def predict_caption(self, image):
        txt = " what does the image describe?"
        inputs = self.tokenizer([txt], max_length=1024, return_tensors="pt")["input_ids"]
        patch_image = self.patch_resize_transform(image).unsqueeze(0)
        gen = self.model.generate(inputs, patch_images=patch_image, num_beams=4)
        ofa_caption = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return ofa_caption


def model_fn(model_dir):
    print(f'model_fn - model_dir: {model_dir}')

    for file in glob.glob(model_dir+'/*', recursive=True):
        print(file)

    predictor = OFAImageCaptionPredictor(model_dir)
    return predictor


def input_fn(serialized_input_data, content_type=NPY_CONTENT_TYPE):
    print(f'input_fn - serialized_input_data length: {len(serialized_input_data)}, content_type: {content_type}')
    if content_type == NPY_CONTENT_TYPE:
        io_bytes_obj = io.BytesIO(serialized_input_data)
        npy_payload = np.load(io_bytes_obj)
        image = Image.fromarray(npy_payload)
        return image
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return


def predict_fn(image, predictor):
    print(f'predict_fn - image: {image}')
    print(f'predict_fn - image data length: {image}')
    result = predictor.predict_caption(image)
    print(f'predict_fn - result: {result}')
    return result
