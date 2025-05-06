#!/bin/bash

# Exit script if any command fails
set -e

DOCKER_IMAGE="comisr:latest"
CONTAINER_NAME="comisr-container"

# Check if the container already exists
if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
    echo "Removing existing container: $CONTAINER_NAME"
    docker rm -f $CONTAINER_NAME
fi

# Run the container with GPU support
echo "Starting container: $CONTAINER_NAME with GPU support"
docker run --gpus all -it --rm \
    --name $CONTAINER_NAME \
    -v $(pwd):/app \
    -v /tmp/data:/app/data \
    -v /tmp/outputs:/app/outputs \
    $DOCKER_IMAGE

echo "Container has been stopped" 