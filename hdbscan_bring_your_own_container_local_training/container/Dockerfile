FROM python:slim

MAINTAINER Amazon AI <sage-learner@amazon.com>

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev

RUN pip3 install --no-cache-dir -U \
    numpy \
    pandas \
    hdbscan==0.8.27

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY hdbscan /opt/program
WORKDIR /opt/
ENTRYPOINT ["train.py"]
