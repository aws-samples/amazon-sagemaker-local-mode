The instructions to install Docker CLI and Docker Compose plugin based on SageMaker Studio Environment are documented below. Follow specific instructions based on applicable [Studio Application Type](https://docs.aws.amazon.com/sagemaker/latest/dg/machine-learning-environments.html) / [Images](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-images.html#notebooks-available-images-supported). These instructions adhere to [Studio platforms requirements](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-local.html#studio-updated-local-docker) for enabling Local Mode/Docker Access.


* [**SageMaker Distribution Docker CLI Install Directions**](sagemaker-distribution-docker-cli-install.sh):  This script provides instructions for Docker CLI Install in Studio JupyterLab/Studio Code Editor and Studio Classic [SageMaker Distribution Images](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-distribution.html) which are Ubuntu-Jammy based. Do `cat /etc/os-release` to verify the OS in App Image terminal.
  * Applicable Studio AppType/Images:
    * JupyterLab
    * Code Editor
    * Amazon SageMaker Studio Classic [Kernel Gateway Applications]
      * Applicable Images: SageMaker Distribution.
* [**SageMaker Classic - Ubuntu-Jammy Docker CLI Install Directions**](sagemaker-ubuntu-jammy-docker-cli-install.sh):  This script provides instructions for Docker CLI Install for Studio Classic SageMaker Images which are Ubuntu-Jammy based. Do `cat /etc/os-release` to verify the OS in App Image terminal.
  * Applicable Studio AppTypes/Images:
    * Amazon SageMaker Studio Classic [Kernel Gateway Applications]
      * Applicable Images: Data Science, SparkAnalytics.
* [**SageMaker Classic - Debian-Bullseye Docker CLI Install Directions**](sagemaker-debian-bullseye-cli-install.sh):  This script provides instructions for Docker CLI Install for Studio Classic SageMaker Images which are Debian-Bullseye based. Do `cat /etc/os-release` to verify the OS in App Image terminal.
  * Applicable Studio AppTypes/Images:
    * Amazon SageMaker Studio Classic [Kernel Gateway Applications]
      * Applicable Images: Base Python.
* [**SageMaker Classic - Ubuntu-Focal Docker CLI Install Directions**](sagemaker-ubuntu-focal-docker-cli-install.sh):  This script provides instructions for Docker CLI Install for Studio Classic SageMaker Images which are Ubuntu-Focal based. Do `cat /etc/os-release` to verify the OS in App Image terminal.
  * Applicable Studio AppTypes/Images:
    * Amazon SageMaker Studio Classic [Kernel Gateway Applications]
      * Applicable Images: All currently supported Pytorch/Tensorflow Framework based Studio Images [here](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-images.html#notebooks-available-images-supported).

<div style="background-color: #ffffcc; border-left: 6px solid #ffeb3b; margin-bottom: 15px; padding: 4px 12px;">
  <p style="color: #000000; margin: 0;"><strong>Note:</strong> For Studio use-cases requiring Docker Build Feature, Docker's Buildkit support should be disabled. Disable using: `export DOCKER_BUILDKIT=0`. This feature is not allowed in Studio as it requires privileged mode execution. </p>
</div>