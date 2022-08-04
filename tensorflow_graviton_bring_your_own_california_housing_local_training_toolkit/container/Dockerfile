# Build a Graviton image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM arm64v8/ubuntu

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
         gcc \
         linux-headers-generic \
         libc-dev

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-aarch64.sh
RUN chmod a+x Miniconda3-py38_4.9.2-Linux-aarch64.sh
RUN bash Miniconda3-py38_4.9.2-Linux-aarch64.sh -b
ENV PATH /root/miniconda3/bin/:$PATH

COPY ml-dependencies.yml ./
RUN conda env create -f ml-dependencies.yml

ENV PATH /root/miniconda3/envs/ml-dependencies/bin:$PATH


