# This is a sample Python program that runs a Dask Processing job on a JSON fetched from a web site.
# The output of the processing will be total files found in the JSON.
# This implementation will work on your *local computer*.
#
# Prerequisites:
#   1. Install required Python packages:
#       pip install boto3 sagemaker pandas scikit-learn
#       pip install 'sagemaker[local]'
#   2. Docker Desktop has to be installed on your computer, and running.
#   3. Open terminal and run the following commands:
#       docker build -t sagemaker-dask-processing-local container/.
########################################################################################################################

from sagemaker.local import LocalSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
import boto3


s3 = boto3.client('s3')
sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

# For local training a dummy role will be sufficient
role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

dask_processor = ScriptProcessor(command=["/opt/program/bootstrap.py"],
                    image_uri='sagemaker-dask-processing-local',
                    role=role,
                    instance_count=1,
                    instance_type='local')

dask_processor.run(code='processing_script.py',
                   outputs=[ProcessingOutput(
                       output_name='filenames_processed_data',
                       source='/opt/ml/processing/processed_data/')],
                   arguments=['site_uri', 'https://archive.analytics.mybinder.org/index.jsonl']
                   )

preprocessing_job_description = dask_processor.jobs[-1].describe()
output_config = preprocessing_job_description['ProcessingOutputConfig']

print(output_config)

for output in output_config['Outputs']:
    if output['OutputName'] == 'filenames_processed_data':
        filenames_processed_data_file = output['S3Output']['S3Uri']
        bucket = filenames_processed_data_file.split("/")[:3][2]
        output_file_name = '/'.join(filenames_processed_data_file.split("/")[3:])+"/filenames_in_json.txt"

print(f'Opening processing output file: {"s3://"+bucket+"/"+output_file_name}')
data = s3.get_object(Bucket=bucket, Key=output_file_name)
contents = data['Body'].read()
print('Processing output file content\n-----------\n')
print(contents.decode("utf-8"))