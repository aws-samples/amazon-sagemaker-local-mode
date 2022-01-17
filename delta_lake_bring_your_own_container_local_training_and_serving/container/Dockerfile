# Build an image that can do training and inference in SageMaker

FROM openjdk:8-jre-slim

RUN apt-get update
RUN apt-get install -y python3 python3-setuptools python3-pip python-dev python3-dev

RUN apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates

RUN pip3 install catboost pandas flask gevent gunicorn pyspark==3.2.0 delta-spark

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY catboost_regressor /opt/program
WORKDIR /opt/program

