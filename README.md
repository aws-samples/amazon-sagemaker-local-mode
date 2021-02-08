## Amazon SageMaker Local Mode Examples

This repository contains examples and related resources showing you how to preprocess, train, debug your training script with breakpoints, and serve on your local machine using Amazon SageMaker Local mode for processing jobs, training and serving. 

## Overview

The local mode in the Amazon SageMaker Python SDK can emulate CPU (single and multi-instance) and GPU (single instance) SageMaker training jobs by changing a single argument in the TensorFlow, PyTorch or MXNet estimators.  To do this, it uses Docker compose and NVIDIA Docker.  It will also pull the Amazon SageMaker TensorFlow, PyTorch or MXNet containers from Amazon ECS, so you’ll need to be able to access a public Amazon ECR repository from your local environment.

## SageMaker local mode training and serving in PyCharm
This repository examples will work in any IDE on your local machine. 

Here you can see a TensorFlow example running on PyCharm. **The data for training and serving is also located on your local machine file system**.

#### SageMaker local mode training in PyCharm

![SageMaker local mode training in PyCharm](img/pycharm_sagemaker_local_training.png)

#### SageMaker local mode serving in PyCharm

![SageMaker local mode serving in PyCharm](img/pycharm_sagemaker_local_serving.png)

#### SageMaker local mode processing jobs in PyCharm

![SageMaker local mode processing jobs in PyCharm](img/pycharm_sagemaker_local_processing_jobs.png)

#### Debugging your training script running SageMaker local mode training in PyCharm

![Debug your application](img/debug_your_application_2.png)


### Repository Structure

The repository contains the following resources:

- **TensorFlow resources:**  

  - [**TensorFlow Script Mode Training and Serving**](tensorflow_script_mode_local_training_and_serving):  This example shows how to train and serve your model with TensorFlow and SageMaker script mode, on your local machine using SageMaker local mode.
  - [**TensorFlow Script Mode Debug Training Script**](tensorflow_script_mode_debug_local_training):  This example shows how to debug your training script running inside a prebuilt SageMaker Docker image for TensorFlow, on your local machine using SageMaker local mode.
  - [**TensorFlow Script Mode Deploy a Trained Model and inference on file from S3**](tensorflow_script_mode_local_model_inference):  This example shows how to deploy a trained model to a SageMaker endpoint, on your local machine using SageMaker local mode, and inference with a file in S3 instead of http payload for the SageMaker Endpoint.
  
- **PyTorch resources:**  

  - [**PyTorch Script Mode Training and Serving**](pytorch_script_mode_local_training_and_serving):  This example shows how to train and serve your model with PyTorch and SageMaker script mode, on your local machine using SageMaker local mode.   

- **Bring Your Own Container resources:**  

  - [**Bring Your Own Container CatBoost Algorithm**](catboost_bring_your_own_container_local_training_and_serving):  This example provides a detailed walkthrough on how to package a CatBoost algorithm for training and production-ready hosting. We have included also a Python file for local training and serving that can run on your local computer, for faster development.    
  - [**Bring Your Own Container LightGBM Algorithm**](lightgbm_bring_your_own_container_local_training_and_serving):  This example provides a detailed walkthrough on how to package a LightGBM algorithm for training and production-ready hosting. We have included also a Python file for local training and serving that can run on your local computer, for faster development.
  - [**Bring Your Own Container Prophet Algorithm**](prophet_bring_your_own_container_local_training_and_serving):  This example provides a detailed walkthrough on how to package a Prophet algorithm for training and production-ready hosting. We have included also a Python file for local training and serving that can run on your local computer, for faster development.
  
- **Built-in scikit-learn Processing Job :**  

  - [**Built-in scikit-learn Processing Job**](scikit_learn_local_processing):  This example provides a detailed walkthrough on how to use the built-in scikit-learn Docker image for processing jobs. We have included also a Python file for processing jobs that can run on your local computer, for faster development.

- **Bring Your Own Container scikit-learn Processing Job :**  

  - [**Bring Your Own Container scikit-learn Processing Job**](scikit_learn_bring_your_own_container_local_processing):  This example provides a detailed walkthrough on how to package a scikit-learn Docker image for processing jobs. We have included also a Python file for processing jobs that can run on your local computer, for faster development.
    
**Note**: Those examples were tested on macOS and Linux.

### Installation Instructions

1. [Create an AWS account](https://portal.aws.amazon.com/gp/aws/developer/registration/index.html) if you do not already have one and login.

2. Install [Docker Desktop for Mac](https://hub.docker.com/editions/community/docker-ce-desktop-mac)

3. [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-mac.html#cliv2-mac-install-gui) and [Configure AWS credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-config).

4. Clone the repo onto your local development machine using `git clone`.

5. Open the project in any IDE of your choice in order to run the example Python files.

6. Follow the instructions on which Python packages to install in each of the example Python file.

## Questions?

Please contact [@e_sela](https://twitter.com/e_sela) or raise an issue on this repo.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
