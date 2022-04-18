# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# For more information on creating a Dockerfile
# https://docs.docker.com/compose/gettingstarted/#step-2-create-a-dockerfile
ARG REGION=us-east-1

# SageMaker TensorFlow image
FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/tensorflow-training:2.8-cpu-py39

# We will use the following packages in the training script.
RUN pip3 install nltk gensim

CMD ["/bin/bash"]