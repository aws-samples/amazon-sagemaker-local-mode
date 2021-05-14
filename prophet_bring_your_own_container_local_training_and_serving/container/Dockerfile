# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:18.04

MAINTAINER Amazon AI <sage-learner@amazon.com>

RUN apt-get -y update

RUN apt-get install -y --no-install-recommends \
         wget \
         curl \
         build-essential libssl-dev libffi-dev \
         libxml2-dev libxslt1-dev zlib1g-dev \
         nginx \
         ca-certificates


RUN apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip


RUN pip --no-cache-dir install \
        numpy \
        scipy \
        sklearn \
        pandas \
        flask \
        gevent \
        gunicorn \
        pystan \
        lunarcalendar \
        convertdate \
        holidays \
        tqdm

RUN pip --no-cache-dir install \
        fbprophet==0.7.1
        
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY prophet /opt/program
WORKDIR /opt/program

