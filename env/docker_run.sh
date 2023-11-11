#!/bin/bash

DOCKER_IMAGE_NAME=pytorch_training:v1
DOCKER_CONTAINER_NAME=pt
CURRENT_WORKING_DIR=$(pwd)/

if [[ "$(docker images -q $DOCKER_IMAGE_NAME 2> /dev/null)" == ""  ]]
then
    docker build -t $DOCKER_IMAGE_NAME -f "${CURRENT_WORKING_DIR}env/Dockerfile" .
fi

docker run -it \
           --name $DOCKER_CONTAINER_NAME \
           --mount type=bind,src=$CURRENT_WORKING_DIR,dst=/work/ \
           $DOCKER_IMAGE_NAME
