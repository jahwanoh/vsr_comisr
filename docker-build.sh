#!/bin/bash

# Exit script if any command fails
set -e

DOCKER_IMAGE="comisr:latest"
DOCKER_BUILDKIT=1

echo "Building Docker image: $DOCKER_IMAGE"
docker build -t $DOCKER_IMAGE .

echo "Build completed successfully!"
echo "Run './docker-run.sh' to start the container." 