#!/bin/bash

# Build the Docker image with compatible TensorFlow
docker build -t video-pipeline-sr-compatible -f Dockerfile.compatible .

echo "Docker image built successfully"
echo "Run the container with:"
echo "docker run --gpus all -it -v $(pwd)/data:/app/data -v $(pwd)/model:/app/model video-pipeline-sr-compatible" 