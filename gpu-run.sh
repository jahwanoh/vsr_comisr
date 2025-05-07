#!/bin/bash

# Run the Docker container with GPU support
docker run --gpus all -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model:/app/model \
  video-pipeline-sr-gpu \
  /app/docker-sr.sh 