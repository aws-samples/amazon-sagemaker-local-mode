# This is a sample Python program that runs a simple scikit-learn processing using the FrameworkProcessor to perform
# word_tokenize using nltk. nltk is installed using dependencies installed from requirements.txt file.
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop installed and running on your computer:
#      `docker ps`
#   3. You should have AWS credentials configured on your local machine
#      in order to be able to pull the docker image from ECR.
########################################################################################################################
from sagemaker.local import LocalSession
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.processing import FrameworkProcessor
from sagemaker.sklearn.estimator import SKLearn

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

# For local training a dummy role will be sufficient
role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

processor = FrameworkProcessor(
    estimator_cls=SKLearn,
    framework_version='1.2-1',
    instance_count=1,
    instance_type='local',
    role=role
)

print('Starting processing job.')
print('Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
processor.run(
    code='processing_script.py',
    dependencies=['./dependencies/requirements.txt'],
    inputs=[
        ProcessingInput(
          source='./input_data/',
          destination='/opt/ml/processing/input_data/')
   ],
    outputs=[ProcessingOutput(
        output_name='tokenized_words_data',
        source='/opt/ml/processing/processed_data/')],
    arguments=['job-type', 'word-tokenize']
)

preprocessing_job_description = processor.jobs[-1].describe()
output_config = preprocessing_job_description['ProcessingOutputConfig']

print(output_config)

for output in output_config['Outputs']:
    if output['OutputName'] == 'tokenized_words_data':
        tokenized_words_data_file = output['S3Output']['S3Uri']

print('Output file is located on: {}'.format(tokenized_words_data_file))