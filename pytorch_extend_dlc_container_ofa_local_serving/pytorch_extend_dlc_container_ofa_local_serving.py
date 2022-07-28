# This is a sample Python program that uses the OFA pretrained model to perform inference using a Docker image that extends AWS DLC PyTorch.
# https://huggingface.co/OFA-Sys/OFA-tiny
# This implementation will work on your local computer.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn Pillow
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build  -t sagemaker-ofa-pytorch-extended-local container/.
########################################################################################################################

import sagemaker
from PIL import Image
import numpy as np
from sagemaker.pytorch import PyTorchModel
from sagemaker.local import LocalSession
import boto3

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'
LOCAL_SESSION = LocalSession()
LOCAL_SESSION.config={'local': {'local_code': True}} # Ensure full code locality, see: https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode


def main():

    image = 'sagemaker-ofa-pytorch-extended-local'

    print('Downloading model file from S3')
    s3 = boto3.client('s3')
    s3.download_file('aws-ml-blog', 'artifacts/pytorch-extend-dlc-container-ofa-tiny/model.tar.gz', 'model.tar.gz')
    print('Model downloaded')

    ofa_hf_model = PyTorchModel(
        source_dir="code",
        entry_point="inference.py",
        role=DUMMY_IAM_ROLE,
        model_data="file://model.tar.gz",
        image_uri=image,
        framework_version='1.8',
        sagemaker_session=LOCAL_SESSION
    )

    print('Deploying endpoint in local mode')
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
