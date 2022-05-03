# This is a sample Python program that deploy a TensorFlow model using multi-model server, and then performs inference.
# multi-model server: https://github.com/awslabs/multi-model-server
# This implementation will work on your local computer.
# *NOTE*: With SageMaker Local Mode you will be able to test only one model in MME mode.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build  -t sagemaker-tensorflow2-mms-local container/.
########################################################################################################################

from sagemaker import Model, LocalSession, Predictor
from sagemaker.deserializers import JSONDeserializer
from sagemaker.multidatamodel import MultiDataModel
from sagemaker.serializers import JSONSerializer

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}


def main():

    image = 'sagemaker-tensorflow2-mms-local'
    endpoint_name = "my-local-endpoint"

    role = DUMMY_IAM_ROLE
    model_data_prefix = 's3://aws-ml-blog/artifacts/tensorflow-script-mode-no-tfs-inference/'
    model_dir = model_data_prefix + 'model.tar.gz'

    model = Model(
        image_uri=image,
        role=role,
        model_data=model_data_prefix,
    )

    mme = MultiDataModel(
        name=endpoint_name,
        model_data_prefix=model_data_prefix,
        model=model,
        sagemaker_session=sagemaker_session,
    )

    print('Deploying endpoint in local mode')
    print(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')

    endpoint = mme.deploy(
        initial_instance_count=1,
        instance_type='local',
        endpoint_name=endpoint_name
    )

    print(f"Model list on MME: {list(mme.list_models())}")

    predictor = Predictor(endpoint_name=endpoint_name,
                          sagemaker_session=sagemaker_session,
                          serializer=JSONSerializer(),
                          deserializer=JSONDeserializer())

    data = {"instances": [[1.53250854, -2.03172922, 1.15884022, 0.38779065, 0.1527185, -0.03002725, -0.925089, 0.9848863],
                          [-2.03172922, 0.1527185, -2.03172922, 1.15884022, -0.03002725, -0.925089, 0.9848863, 0.38779065]]}

    predictions = predictor.predict(data)
    print(f'predictions: {predictions}')

    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint()


if __name__ == "__main__":
    main()
