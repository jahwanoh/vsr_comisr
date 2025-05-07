#!/bin/bash

# Run the Docker container with compatible TensorFlow
docker run --gpus all -it \
  -v $(pwd):/comisr -v $(pwd)/data:/app/data \
  -v $(pwd)/model:/app/model \
  video-pipeline-sr-compatible 