#!/bin/bash

set -e  # Exit on error

# Print colorful messages
print_green() {
  echo -e "\033[0;32m$1\033[0m"
}

print_yellow() {
  echo -e "\033[0;33m$1\033[0m"
}

print_red() {
  echo -e "\033[0;31m$1\033[0m"
}

# Create necessary directories
mkdir -p data model

# Fix module structure if needed
if [ ! -d "comisr" ] && [ -d "lib" ]; then
  print_yellow "Creating comisr module structure..."
  mkdir -p comisr/lib
  cp lib/*.py comisr/lib/
  
  # Create __init__.py files if they don't exist
  if [ ! -f "comisr/__init__.py" ]; then
    echo "# COMISR package" > comisr/__init__.py
    print_green "Created comisr/__init__.py"
  fi
  
  if [ ! -f "comisr/lib/__init__.py" ]; then
    echo "# COMISR lib package" > comisr/lib/__init__.py
    print_green "Created comisr/lib/__init__.py"
  fi
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
  print_red "Docker is not installed. Please install Docker to continue."
  exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
  print_yellow "Docker is not running. Please start Docker and run this script again."
  exit 1
fi

# Check if the video exists
if [ ! -f "data/match_178203_half_res.mp4" ]; then
  print_yellow "Input video not found. Attempting to copy from Docker container..."
  
  # Try to copy from the Docker container
  if docker ps -a | grep -q video-pipeline-sr; then
    docker cp video-pipeline-sr:/app/data/match_178203_half_res.mp4 ./data/ || \
      print_red "Failed to copy video file. Please add it manually to the data directory."
  else
    print_yellow "Docker container 'video-pipeline-sr' not found."
    print_red "Please add the input video manually to the data directory."
  fi
  
  # Check again if we have the video
  if [ ! -f "data/match_178203_half_res.mp4" ]; then
    print_red "Input video not found in data directory. Cannot proceed."
    exit 1
  fi
fi

# Check if we have model files
if [ ! -f "model/model.ckpt.index" ]; then
  print_yellow "Model checkpoint files not found. Attempting to copy from Docker container..."
  
  # Try to copy from the Docker container
  if docker ps -a | grep -q video-pipeline-sr; then
    docker cp video-pipeline-sr:/app/model/model.ckpt.data-00000-of-00001 ./model/ || \
      print_red "Failed to copy model data file. Please add it manually to the model directory."
    docker cp video-pipeline-sr:/app/model/model.ckpt.index ./model/ || \
      print_red "Failed to copy model index file. Please add it manually to the model directory."
    docker cp video-pipeline-sr:/app/model/model.ckpt.meta ./model/ || \
      print_red "Failed to copy model meta file. Please add it manually to the model directory."
  else
    print_yellow "Docker container 'video-pipeline-sr' not found."
    print_red "Please add the model files manually to the model directory."
  fi
  
  # Check again if we have the model files
  if [ ! -f "model/model.ckpt.index" ]; then
    print_red "Model checkpoint files not found in model directory. Cannot proceed."
    exit 1
  fi
fi

# Try to run with the modified script first
print_green "Attempting to run super-resolution using the modified script..."

# Set PYTHONPATH to include current directory
export PYTHONPATH=.:$PYTHONPATH

if python video_inference.py \
     --input_video="data/match_178203_half_res.mp4" \
     --output_video="data/match_178203_super_res.mp4" \
     --checkpoint_path="model/model.ckpt" \
     --vsr_scale=4 \
     --num_resblock=10 \
     --use_ema=true; then
  
  print_green "Super-resolution completed successfully!"
  print_green "Output video saved to: data/match_178203_super_res.mp4"
  exit 0
fi

# If we got here, the script failed. Try using Docker.
print_yellow "Direct run failed. Attempting to run with compatible Docker container..."

# Check if the compatible Docker image exists
if ! docker images | grep -q video-pipeline-sr-compatible; then
  print_yellow "Building compatible Docker image..."
  
  # Build the compatible Docker image
  if ! docker build -t video-pipeline-sr-compatible -f Dockerfile.compatible .; then
    print_red "Failed to build compatible Docker image."
    exit 1
  fi
fi

# Run the super-resolution using Docker
print_green "Running super-resolution in Docker container..."
docker run --gpus all --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/model:/app/model" \
  video-pipeline-sr-compatible \
  /app/docker-sr.sh

# Check if the output was generated
if [ -f "data/match_178203_super_res.mp4" ]; then
  print_green "Super-resolution completed successfully using Docker!"
  print_green "Output video saved to: data/match_178203_super_res.mp4"
else
  print_red "Failed to generate output video even with Docker."
  print_red "Please check the error messages and try again."
  exit 1
fi 