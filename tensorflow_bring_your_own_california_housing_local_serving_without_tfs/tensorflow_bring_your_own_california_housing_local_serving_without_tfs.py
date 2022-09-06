# This is a sample Python program that shows how to serve a BYOC TensorFlow model with no TFS, and perform inference.
# This implementation will work on your local computer.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build  -t sagemaker-tensorflow2-no-tfs-local container/.
########################################################################################################################

import numpy as np
from sagemaker import Model, LocalSession, Predictor
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}


def main():

    image = 'sagemaker-tensorflow2-no-tfs-local'
    endpoint_name = "my-local-endpoint"

    role = DUMMY_IAM_ROLE
    model_dir = 's3://aws-ml-blog/artifacts/tensorflow-script-mode-no-tfs-inference/model.tar.gz'

    model = Model(
        image_uri=image,
        role=role,
        model_data=model_dir,
    )

    print('Deploying endpoint in local mode')
    print(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')

    endpoint = model.deploy(
        initial_instance_count=1,
        instance_type='local',
        endpoint_name=endpoint_name
    )

    predictor = Predictor(endpoint_name=endpoint_name,
                          sagemaker_session=sagemaker_session,
                          serializer=JSONSerializer(),
                          deserializer=JSONDeserializer())

    data = {"instances": [[1.53250854, -2.03172922, 1.15884022, 0.38779065, 0.1527185, -0.03002725, -0.925089, 0.9848863]]}
    results = predictor.predict(data)['predictions']

    flat_list = [float('%.1f' % (item)) for sublist in results for item in sublist]
    print('predictions: \t{}'.format(np.array(flat_list)))

    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint()


if __name__ == "__main__":
    main()
