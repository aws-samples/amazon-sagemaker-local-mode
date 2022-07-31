# Build a Graviton image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM arm64v8/ubuntu

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-aarch64.sh
RUN chmod a+x Miniconda3-py39_4.10.3-Linux-aarch64.sh
RUN bash Miniconda3-py39_4.10.3-Linux-aarch64.sh -b
ENV PATH /root/miniconda3/bin/:$PATH

COPY ml-dependencies.yml ./
RUN conda env create -f ml-dependencies.yml

ENV PATH /root/miniconda3/envs/ml-dependencies/bin:$PATH

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY linear_regressor /opt/program
WORKDIR /opt/program

