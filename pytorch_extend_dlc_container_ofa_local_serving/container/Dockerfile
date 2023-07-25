ARG REGION=us-east-1

# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-inference:2.0.1-cpu-py310-ubuntu20.04-sagemaker

RUN git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
RUN pip install OFA/transformers/

ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
