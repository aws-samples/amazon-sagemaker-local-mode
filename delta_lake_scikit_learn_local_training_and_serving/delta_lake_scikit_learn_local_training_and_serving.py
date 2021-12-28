# This is a sample Python program that trains a simple scikit-learn model
# on the boston-housing dataset fetched from Delta Lake.
# This implementation will work on your *local computer* or in the *AWS Cloud*.
#
# Delta Sharing: An Open Protocol for Secure Data Sharing
# https://github.com/delta-io/delta-sharing
#
# Prerequisites:
#   1. Install required Python packages:
#      `pip install -r requirements.txt`
#   2. Docker Desktop installed and running on your computer:
#      `docker ps`
#   3. You should have AWS credentials configured on your local machine
#      in order to be able to pull the docker image from ECR.
###############################################################################################


from sagemaker.sklearn import SKLearn


DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def main():

    print('Starting model training.')
    print('Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')

    sklearn = SKLearn(
        entry_point="scikit_boston_housing.py",
        source_dir='code',
        framework_version="0.23-1",
        instance_type="local",
        role=DUMMY_IAM_ROLE
    )

    delta_lake_profile_file = "file://./profile/open-datasets.share"

    sklearn.fit({"train": delta_lake_profile_file})
    print('Completed model training')

    # print('Deploying endpoint in local mode')
    # predictor = sklearn.deploy(initial_instance_count=1, instance_type='local')
    #
    #
    # print('About to delete the endpoint to stop paying (if in cloud mode).')
    # predictor.delete_endpoint(predictor.endpoint_name)


if __name__ == "__main__":
    main()
