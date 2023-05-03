FROM ubuntu:22.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
         gcc \
         linux-headers-generic \
         libc-dev

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
RUN chmod a+x Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
RUN bash Miniconda3-py38_23.3.1-0-Linux-x86_64.sh -b
ENV PATH /root/miniconda3/bin/:$PATH

COPY ml-dependencies.yml ./
RUN conda env create -f ml-dependencies.yml

ENV PATH /root/miniconda3/envs/ml-dependencies/bin:$PATH


