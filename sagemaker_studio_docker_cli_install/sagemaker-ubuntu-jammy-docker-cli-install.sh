#!/bin/bash

# for a bit of documentation, that script is meant for jammy jellyfish,
# if you want to use another version, set the VERSION_CODENAME environment
# variable when running for another version, also it defaults the DOCKER_HOST
# to the location of the socket but if sagemaker does evolve, you can again
# just set that environment variable

apt-get update
apt-get install ca-certificates curl gnupg -y
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "${VERSION_CODENAME:-jammy}")" stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update

# pick the latest patch from:
# apt-cache madison docker-ce | awk '{ print $3 }' | grep -i 20.10
VERSION_STRING=5:20.10.24~3-0~ubuntu-${VERSION_CODENAME:-jammy}
apt-get install docker-ce-cli=$VERSION_STRING docker-compose-plugin -y

# validate the Docker Client is able to access Docker Server at [unix:///docker/proxy.sock]

if [ -z "${DOCKER_HOST}" ]; then
  export DOCKER_HOST="unix:///docker/proxy.sock"
fi

docker version
