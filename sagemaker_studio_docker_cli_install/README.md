The instructions to install Docker CLI and Docker Compose plugin based on SageMaker Studio Environment are documented below. Follow specific instructions based on applicable [Studio Application Type](https://docs.aws.amazon.com/sagemaker/latest/dg/machine-learning-environments.html) / [Images](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-images.html#notebooks-available-images-supported). These instructions adhere to [Studio platforms requirements](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-local.html#studio-updated-local-docker) for enabling Local Mode/Docker Access.

* [**SageMaker Distribution Docker CLI Install Directions**](sagemaker-distribution-docker-cli-install.sh):  This script provides instructions for Docker CLI Install in Studio JupyterLab/Studio Code Editor and Studio Classic [SageMaker Distribution Images](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-distribution.html) which are Ubuntu-Jammy based. Do `cat /etc/os-release` to verify the OS in App Image terminal.
  * Applicable Studio AppType/Images:
    * JupyterLab
    * Code Editor
    * Amazon SageMaker Studio Classic [Kernel Gateway Applications]
      * Applicable Images: SageMaker Distribution v0 CPU, SageMaker Distribution v0 GPU, SageMaker Distribution v1 CPU, SageMaker Distribution v1 GPU.
* [**SageMaker Classic - Debian-Bullseye Docker CLI Install Directions**](sagemaker-debian-bullseye-cli-install.sh):  This script provides instructions for Docker CLI Install for Studio Classic SageMaker Images which are Debian-Bullseye based. Do `cat /etc/os-release` to verify the OS in App Image terminal.
  * Applicable Studio AppTypes/Images:
    * Amazon SageMaker Studio Classic [Kernel Gateway Applications]
      * Applicable Images: Base Python 3.0, Base Python 2.0, Data Science 3.0, Data Science 2.0, SparkAnalytics 2.0, SparkAnalytics 1.0.
* [**SageMaker Classic - Ubuntu-Focal Docker CLI Install Directions**](sagemaker-ubuntu-focal-docker-cli-install.sh):  This script provides instructions for Docker CLI Install for Studio Classic SageMaker Images which are Ubuntu-Focal based. Do `cat /etc/os-release` to verify the OS in App Image terminal.
  * Applicable Studio AppTypes/Images:
    * Amazon SageMaker Studio Classic [Kernel Gateway Applications]
      * Applicable Images: All currently supported Pytorch/Tensorflow Framework based Studio Images [here](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-images.html#notebooks-available-images-supported).
