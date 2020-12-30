# This is a sample Python program that runs a simple scikit-learn processing based on a docker image you build.
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build -t sagemaker-scikit-learn-processing-local container/.
########################################################################################################################

from sagemaker.local import LocalSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

# For local training a dummy role will be sufficient
role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

processor = ScriptProcessor(command=['python3'],
                    image_uri='sagemaker-scikit-learn-processing-local',
                    role=role,
                    instance_count=1,
                    instance_type='local')

processor.run(code='processing_script.py',
                    inputs=[ProcessingInput(
                        source='./input_data/',
                        destination='/opt/ml/processing/input_data/')],
                    outputs=[ProcessingOutput(
                        output_name='word_count_data',
                        source='/opt/ml/processing/processed_data/')],
                    arguments=['job-type', 'word-count']
                    )

preprocessing_job_description = processor.jobs[-1].describe()
output_config = preprocessing_job_description['ProcessingOutputConfig']

print(output_config)

for output in output_config['Outputs']:
    if output['OutputName'] == 'word_count_data':
        word_count_data_file = output['S3Output']['S3Uri']

print('Output file is located on: {}'.format(word_count_data_file))