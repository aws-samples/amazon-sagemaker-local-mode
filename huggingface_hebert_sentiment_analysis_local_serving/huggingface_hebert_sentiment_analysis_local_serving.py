# This is a sample Python program that deploy a pre-trained PyTorch HeBERT model on Amazon SageMaker Endpoint.
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Create Python Virtual Environment:
#      `python3 -m venv env`
#   2. Start the virtual environment
#      `source env/bin/activate`
#   3. Install required Python packages:
#      `pip install -r requirements.txt`
#   4. Run `hebert_model.py` to create the model:
#      `python hebert_model.py`
#   5. Create `model.tar.gz` file for SageMaker to use:
#      `cd model && tar -czf ../model.tar.gz * && cd ..`
#   6. Docker Desktop installed and running on your computer:
#      `docker ps`
#   7. You should have AWS credentials configured on your local machine
#      in order to be able to pull the docker image from ECR.
##############################################################################################

import sagemaker
from sagemaker.local import LocalSession
from sagemaker.pytorch.model import PyTorchModel


def main():
    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    # For local training a dummy role will be sufficient
    role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

    print('Deploying local mode endpoint')
    print('Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')

    pytorch_model = PyTorchModel(model_data='./model.tar.gz',
                                 role=role,
                                 framework_version="1.8",
                                 source_dir="code",
                                 py_version="py3",
                                 entry_point="inference.py")

    predictor = pytorch_model.deploy(initial_instance_count=1, instance_type='local')

    predictor.serializer = sagemaker.serializers.JSONSerializer()
    predictor.deserializer = sagemaker.deserializers.JSONDeserializer()

    result = predictor.predict("אני אוהב לעבוד באמזון")
    print('result: {}'.format(result))

    predictor.delete_endpoint(predictor.endpoint)

if __name__ == "__main__":
    main()