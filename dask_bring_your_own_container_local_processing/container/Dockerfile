FROM continuumio/miniconda3:4.7.12

ENV PYTHONHASHSEED 0
ENV PYTHONIOENCODING UTF-8

# Install required Python packages fo Dask
RUN conda install --yes dask distributed dask-ml boto3

# Install additional Python packages
RUN conda install aiohttp boto3

# Dumb init
RUN wget -O /usr/local/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64
RUN chmod +x /usr/local/bin/dumb-init

RUN mkdir /opt/app /etc/dask
COPY dask_config/dask.yaml /etc/dask/

# Set up bootstrapping program and Dask configuration
COPY program /opt/program
RUN chmod +x /opt/program/bootstrap.py

ENTRYPOINT ["/opt/program/bootstrap.py"]