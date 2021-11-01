# This is a sample Python program that trains a simple XGBoost model on Abalone dataset.
# This implementation will work on your *local computer* or in the *AWS Cloud*.
#
# Prerequisites:
#   1. Install required Python packages:
#      `pip install -r requirements.txt`
#   2. Docker Desktop installed and running on your computer:
#      `docker ps`
#   3. You should have AWS credentials configured on your local machine
#      in order to be able to pull the docker image from ECR.
###############################################################################################

from sagemaker import TrainingInput
from sagemaker.xgboost import XGBoost, XGBoostModel

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def do_inference_on_local_endpoint(predictor, libsvm_str):
    label, *features = libsvm_str.strip().split()
    predictions = predictor.predict(" ".join(["-99"] + features))  # use dummy label -99
    print("Prediction: {}".format(predictions))


def main():
    print('Starting model training.')
    print('Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')

    hyperparameters = {
        "max_depth": "5",
        "eta": "0.2",
        "gamma": "4",
        "min_child_weight": "6",
        "subsample": "0.7",
        "objective": "reg:squarederror",
        "num_round": "50",
        "verbosity": "2",
    }

    xgb_script_mode_estimator = XGBoost(
        entry_point="./code/abalone.py",
        hyperparameters=hyperparameters,
        role=DUMMY_IAM_ROLE,
        instance_count=1,
        instance_type='local',
        framework_version="1.2-1"
    )

    train_input = TrainingInput("file://./data/train/abalone", content_type="text/libsvm")

    xgb_script_mode_estimator.fit({"train": train_input, "validation": train_input})

    print('Completed model training')

    model_data = xgb_script_mode_estimator.model_data
    print(model_data)

    xgb_inference_model = XGBoostModel(
        model_data=model_data,
        role=DUMMY_IAM_ROLE,
        entry_point="./code/inference.py",
        framework_version="1.2-1",
    )

    print('Deploying endpoint in local mode')
    predictor = xgb_inference_model.deploy(
        initial_instance_count=1,
        instance_type="local",
    )

    a_young_abalone = "6 1:3 2:0.37 3:0.29 4:0.095 5:0.249 6:0.1045 7:0.058 8:0.067"
    do_inference_on_local_endpoint(predictor, a_young_abalone)

    an_old_abalone = "15 1:1 2:0.655 3:0.53 4:0.175 5:1.2635 6:0.486 7:0.2635 8:0.415"
    do_inference_on_local_endpoint(predictor, an_old_abalone)

    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint(predictor.endpoint_name)


if __name__ == "__main__":
    main()
