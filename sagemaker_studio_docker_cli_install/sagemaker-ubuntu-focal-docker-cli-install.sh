#!/bin/bash

apt-get update
apt-get install ca-certificates curl gnupg -y
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update

# install docker cli and docker compose plugin
apt-get install docker-ce-cli docker-compose-plugin -y

# disabling docker build kit, docker's build kit feature is not allowed in Studio
export DOCKER_BUILDKIT=0

# validate the Docker Client is able to access Docker Server at [unix:///docker/proxy.sock]
if [ -z "${DOCKER_HOST}" ]; then
  export DOCKER_HOST="unix:///docker/proxy.sock"
fi

# validate the Docker Client is able to access Docker Server at [unix:///docker/proxy.sock]
docker version