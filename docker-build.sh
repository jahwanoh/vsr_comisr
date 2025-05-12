#!/bin/bash

# Build the Docker image with proper CUDA support
docker build -t comisr -f Dockerfile .

echo "Docker image built successfully"
echo "Run the container with:"
echo "docker run --gpus all -it -v $(pwd)/data:/app/data -v $(pwd)/model:/app/model video-pipeline-sr-gpu" 