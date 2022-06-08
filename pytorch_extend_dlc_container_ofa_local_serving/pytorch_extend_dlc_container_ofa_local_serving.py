# This is a sample Python program that uses the OFA pretrained model to perform inference using a Docker image that extends AWS DLC PyTorch.
# https://huggingface.co/OFA-Sys/OFA-tiny
# This implementation will work on your local computer.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build  -t sagemaker-ofa-pytorch-extended-local container/.
########################################################################################################################

import sagemaker
from PIL import Image
import numpy as np
from sagemaker.pytorch import PyTorchModel

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def main():

    image = 'sagemaker-ofa-pytorch-extended-local'

    ofa_hf_model = PyTorchModel(
        source_dir="code",
        entry_point="inference.py",
        role=DUMMY_IAM_ROLE,
        model_data="s3://aws-ml-blog/artifacts/pytorch-extend-dlc-container-ofa-tiny/model.tar.gz",
        image_uri=image,
        framework_version='1.8'
    )

    print('Deploying endpoint in local mode')
    print(
        'Note: model download might take a few minutes to complete due to its size.')
    predictor = ofa_hf_model.deploy(
        initial_instance_count=1,
        instance_type='local',
        serializer=sagemaker.serializers.NumpySerializer(),
        deserializer=sagemaker.deserializers.JSONDeserializer()
    )

    img = Image.open("./test_image.jpg")
    payload = np.asarray(img)

    predictions = predictor.predict(payload)
    print(f'predictions: {predictions}')

    predictor.delete_endpoint(predictor.endpoint)

if __name__ == "__main__":
    main()
