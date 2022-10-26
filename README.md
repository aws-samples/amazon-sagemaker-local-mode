## Amazon SageMaker Local Mode Examples
![AWS ML](img/aws_ml.png) ![Docker](img/docker.png) ![Local Machine](img/local_machine.png) 

This repository contains examples and related resources showing you how to preprocess, train, debug your training script with breakpoints, and serve on your local machine using Amazon SageMaker Local mode for processing jobs, training and serving. 

## Overview

The local mode in the Amazon SageMaker Python SDK can emulate CPU (single and multi-instance) and GPU (single instance) SageMaker training jobs by changing a single argument in the TensorFlow, PyTorch or MXNet estimators.  To do this, it uses Docker compose and NVIDIA Docker.  It will also pull the Amazon SageMaker TensorFlow, PyTorch or MXNet containers from Amazon ECS, so you’ll need to be able to access a public Amazon ECR repository from your local environment.

For full details on how this works:

- Read the Machine Learning Blog post at: https://aws.amazon.com/blogs/machine-learning/use-the-amazon-sagemaker-local-mode-to-train-on-your-notebook-instance/

## SageMaker local mode training and serving in PyCharm
This repository examples will work in any IDE on your local machine. 

Here you can see a TensorFlow example running on PyCharm. **The data for processing, training and serving is also located on your local machine file system**.

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

- **scikit-learn resources:**  

  - [**scikit-learn Script Mode Training and Serving**](scikit_learn_script_mode_local_training_and_serving):  This example shows how to train and serve your model with scikit-learn and SageMaker script mode, on your local machine using SageMaker local mode.
  - [**scikit-learn Bring Your Own Model**](scikit_learn_bring_your_own_model_local_serving):  This example shows how to serve your pre-trained scikit-learn model with SageMaker, on your local machine using SageMaker local mode.
  - [**Gensim Word2Vec Bring Your Own Model**](gensim_with_word2vec_model_artifacts_local_serving):  This example shows how to serve your pre-trained Word2Vec model, trained using BlazingText algorithm with SageMaker, and gensim for inference, on your local machine using SageMaker local mode.
  - [**CatBoost with scikit-learn Script Mode Training and Serving**](catboost_scikit_learn_script_mode_local_training_and_serving):  This example shows how to train and serve a CatBoost model with scikit-learn and SageMaker script mode, on your local machine using SageMaker local mode.
  - [**Delta Sharing scikit-learn Script Mode Training and Serving**](delta_sharing_scikit_learn_local_training_and_serving):  This example shows how to train a scikit-learn model on the boston-housing dataset fetched from Delta Lake using Delta Sharing, and then serve your model with scikit-learn and SageMaker script mode, on your local machine using SageMaker local mode.

- **XGBoost resources:**  

  - [**XGBoost Script Mode Training and Serving**](xgboost_script_mode_local_training_and_serving):  This example shows how to train and serve your model with XGBoost and SageMaker script mode, on your local machine using SageMaker local mode.

- **TensorFlow resources:**  

  - [**TensorFlow Script Mode Training and Serving**](tensorflow_script_mode_local_training_and_serving):  This example shows how to train and serve your model with TensorFlow and SageMaker script mode, on your local machine using SageMaker local mode.
  - [**TensorFlow Script Mode Debug Training Script**](tensorflow_script_mode_debug_local_training):  This example shows how to debug your training script running inside a prebuilt SageMaker Docker image for TensorFlow, on your local machine using SageMaker local mode.
  - [**TensorFlow Script Mode Deploy a Trained Model and inference on file from S3**](tensorflow_script_mode_local_model_inference):  This example shows how to deploy a trained model to a SageMaker endpoint, on your local machine using SageMaker local mode, and inference with a file in S3 instead of http payload for the SageMaker Endpoint.
  - [**TensorFlow Script Mode Training and Batch Transform**](tensorflow_script_mode_california_housing_local_training_and_batch_transform):  This example shows how to train your model and run Batch Transform job with TensorFlow and SageMaker script mode, on your local machine using SageMaker local mode.
  - [**Extending SageMaker TensorFlow Deep Learning Container Image**](tensorflow_extend_dlc_california_housing_local_training):  In this example we show how to package a TensorFlow container, extending the SageMaker TensorFlow training container, with a Python example which works with the California Housing dataset. By extending the SageMaker TensorFlow container we can utilize the existing training solution made to work on SageMaker, leveraging SageMaker TensorFlow `Estimator` object, with `entry_point` parameter, specifying your local Python source file which should be executed as the entry point to training. To make it work, we replace the `framework_version` and `py_version` parameters, with `image_uri` of the Docker Image we have created. This sample code can run on your local machine using SageMaker local mode.

- **PyTorch resources:**  

  - [**PyTorch Script Mode Training and Serving**](pytorch_script_mode_local_training_and_serving):  This example shows how to train and serve your model with PyTorch and SageMaker script mode, on your local machine using SageMaker local mode.
  - [**PyTorch Script Mode Deploy a Trained Model**](pytorch_script_mode_local_model_inference):  This example shows how to deploy a trained model to a SageMaker endpoint, on your local machine using SageMaker local mode, and serve your model with the SageMaker Endpoint.
  - [**Deploy a pre-trained PyTorch HeBERT model from Hugging Face on SageMaker Endpoint**](huggingface_hebert_sentiment_analysis_local_serving):  This example shows how to deploy a pre-trained PyTorch HeBERT model from Hugging Face, on SageMaker Endpoint, on your local machine using SageMaker local mode.
  - [**Deploy a pre-trained PyTorch YOLOv5 model on SageMaker Endpoint**](pytorch_yolov5_local_model_inference):  This example shows how to deploy a pre-trained PyTorch YOLOv5 model on SageMaker Endpoint, on your local machine using SageMaker local mode.
  
- **Bring Your Own Container Training resources:**  

  - [**Bring Your Own Container TensorFlow Algorithm - Train/Serve**](tensorflow_bring_your_own_california_housing_local_training_and_serving):  This example provides a detailed walkthrough on how to package a Tensorflow 2.5.0 algorithm for training and production-ready hosting. We have included also a Python file for local training and serving that can run on your local computer, for faster development.
  - [**Bring Your Own Container TensorFlow Algorithm - Train/Batch Transform**](tensorflow_bring_your_own_california_housing_local_training_and_batch_transform):  This example provides a detailed walkthrough on how to package a Tensorflow 2.5.0 algorithm for training, and then run a Batch Transform job on a CSV file. We have included also a Python file for local training and serving that can run on your local computer, for faster development.
  - [**Bring Your Own Container TensorFlow Algorithm - Serve without TensorFlow Serving**](tensorflow_bring_your_own_california_housing_local_serving_without_tfs):  This example provides a detailed walkthrough on how to package a Tensorflow 2.5.0 algorithm for production-ready hosting without TensorFlow Serving. We have included also a Python file for local serving that can run on your local computer, for faster development.
  - [**Bring Your Own Container CatBoost Algorithm**](catboost_bring_your_own_container_local_training_and_serving):  This example provides a detailed walkthrough on how to package a CatBoost algorithm for training and production-ready hosting. We have included also a Python file for local training and serving that can run on your local computer, for faster development.    
  - [**Bring Your Own Container LightGBM Algorithm**](lightgbm_bring_your_own_container_local_training_and_serving):  This example provides a detailed walkthrough on how to package a LightGBM algorithm for training and production-ready hosting. We have included also a Python file for local training and serving that can run on your local computer, for faster development.
  - [**Bring Your Own Container Prophet Algorithm**](prophet_bring_your_own_container_local_training_and_serving):  This example provides a detailed walkthrough on how to package a Prophet algorithm for training and production-ready hosting. We have included also a Python file for local training and serving that can run on your local computer, for faster development.
  - [**Bring Your Own Container HDBSCAN Algorithm**](hdbscan_bring_your_own_container_local_training):  This example provides a detailed walkthrough on how to package a HDBSCAN algorithm for training. We have included also a Python file for local training that can run on your local computer, for faster development.
  - [**Bring Your Own Container scikit-learn Algorithm and deploy a pre-trained Model**](scikit_learn_bring_your_own_container_and_own_model_local_serving):  This example provides a detailed walkthrough on how to package a scikit-learn algorithm for serving with a pre-trained model. We have included also a Python file for local serving that can run on your local computer, for faster development.
  - [**Delta Lake Bring Your Own Container CatBoost Algorithm**](delta_lake_bring_your_own_container_local_training_and_serving):  This example provides a detailed walkthrough on how to package a CatBoost algorithm for training on data fetched from Delta Lake, directly from S3, and then serve your model with the Docker image you have built. We have included also a Python file for local training and serving that can run on your local computer, for faster development.  

- **Built-in scikit-learn Processing resources:**  

  - [**Built-in scikit-learn Processing Job**](scikit_learn_local_processing):  This example provides a detailed walkthrough on how to use the built-in scikit-learn Docker image for processing jobs. We have included also a Python file for processing jobs that can run on your local computer, for faster development.

- **Bring Your Own Container Processing resources:**  

  - [**Bring Your Own Container scikit-learn Processing Job**](scikit_learn_bring_your_own_container_local_processing):  This example provides a detailed walk-through on how to package a scikit-learn Docker image for processing jobs. We have included also a Python file for processing jobs that can run on your local computer, for faster development.
  - [**Delta Sharing Bring Your Own Container Processing Job**](delta_sharing_bring_your_own_container_local_processing):  This example provides a detailed walk-through on how to package a scikit-learn Docker image for processing job that fetch data from a table on Delta Lake using Delta Sharing, and aggregate total COVID-19 cases per country. We have included also a Python file for processing jobs that can run on your local computer, for faster development.
  - [**Dask Bring Your Own Container Processing Job**](dask_bring_your_own_container_local_processing):  This example provides a detailed walk-through on how to package a Dask Docker image for processing job that fetch JSON file from a website, and outputs the filenames found. We have included also a Python file for processing jobs that can run on your local computer, for faster development.

- **Graviton resources - will work only on MacBook M1/ARM/Apple Silicon:**  

  - [**Deploy a pre-trained TensorFlow model on SageMaker Graviton Endpoint**](tensorflow_graviton_script_mode_local_model_inference):  This example shows how to deploy a pre-trained TensorFlow model on SageMaker Graviton Endpoint, using your local machine using SageMaker local mode.
  - [**Deploy a pre-trained PyTorch CIFAR-10 model on SageMaker Graviton Endpoint**](pytorch_graviton_script_mode_local_model_inference):  This example shows how to deploy a pre-trained PyTorch CIFAR-10 model on SageMaker Graviton Endpoint, using your local machine using SageMaker local mode.
  - [**Bring Your Own Container scikit-learn Algorithm - Train/Serve**](scikit_learn_graviton_bring_your_own_container_local_training_and_serving):  This example provides a detailed walkthrough on how to package a scikit-learn algorithm for training and production-ready hosting. We have included also a Python file for local training and serving that can run on your local M1 MacBook computer, for faster development.
  - [**Bring Your Own Container TensorFlow Algorithm - Train**](tensorflow_graviton_bring_your_own_california_housing_local_training):  This example provides a detailed walkthrough on how to package a TensorFlow algorithm for training. We have included also a Python file for local training that can run on your local M1 MacBook computer, for faster development.
  - [**Bring Your Own Container TensorFlow Algorithm - SageMaker Training Toolkit**](tensorflow_graviton_bring_your_own_california_housing_local_training_toolkit):  This example provides a detailed walkthrough on how to package a TensorFlow algorithm for training using the SageMaker Training Toolkit. We have included also a Python file for local training that can run on your local M1 MacBook computer, for faster development.

**Note**: Those examples were tested on macOS and Linux.

### Prerequisites

1. [Create an AWS account](https://portal.aws.amazon.com/gp/aws/developer/registration/index.html) if you do not already have one and login.

2. Install [Docker Desktop for Mac](https://hub.docker.com/editions/community/docker-ce-desktop-mac)

3. [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-mac.html#cliv2-mac-install-gui) and [Configure AWS credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-config).

4. Make sure you have Python3 installed `python --version`

5. Make sure you have pip installed `pip --version`


### Installation Instructions for PyCharm
We assume the root folder of the project is `~/Documents/dev/`

1. Open Terminal and navigate to the project root folder: `cd ~/Documents/dev/`

2. Create a directory for the GitHub projects: `mkdir GitHub`

3. Navigate to `~/Documents/dev/GitHub/`: `cd GitHub`

4. Clone the repo onto your local development machine using `git clone https://github.com/aws-samples/amazon-sagemaker-local-mode`

5. Open PyCharm

6. Open `amazon-sagemaker-local-mode` project from `~/Documents/dev/GitHub/amazon-sagemaker-local-mode/`

7. Now you will add a new virtual environment and install the required Python dependencies to run the demos.  

8. Navigate to PyCharm -> Preferences -> Python Interpreter, and click "Add"
![Add new Interpreter](img/python_interpreter_initial_add_venv.png)

9. Add a new Virtual environment by specifying the location of the virtual environment to be created: `/Users/<MY USER>/Documents/dev/sagemaker-python-sdk`
![Add new venv](img/python_interpreter_save_new_venv.png)

10. Click OK

11. On the Python Interpreter Screen, click OK.
![Final add venv](img/python_interpreter_final_add_venv.png)

12. Open terminal (inside PyCharm) and install the requirements: `python -m pip install -r requirements.txt`
![install requirements](img/install_requirements_txt.png)

13. Once completed, navigate to `tensorflow_script_mode_california_housing_local_training_and_serving` folder, and double click on `tensorflow_script_mode_california_housing_local_training_and_serving.py` file
![open tf example](img/open_tf_training_and_serving.png)

14. Right click -> Run `tensorflow_script_mode_california_housing_local_training_and_serving.py`
![run tf example](img/run_tf_training_and_serving.png)

15. The container image download might take a few minutes to complete, but eventually you will View the output
![view tf output](img/output_tf_training_and_serving.png)


## Questions?

Please contact [@e_sela](https://twitter.com/e_sela) or raise an issue on this repo.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
