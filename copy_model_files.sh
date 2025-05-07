#!/bin/bash

# Make sure directories exist
mkdir -p data model

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running. Please start Docker and try again."
  exit 1
fi

# Check if the container exists
if ! docker ps -a | grep -q video-pipeline-sr; then
  echo "Error: Docker container 'video-pipeline-sr' not found."
  echo "Please make sure the container exists and is running."
  exit 1
fi

# Function to copy a file from Docker with a progress indicator
copy_file() {
  local src=$1
  local dest=$2
  local name=$3
  
  echo -n "Copying $name... "
  if docker cp video-pipeline-sr:"$src" "$dest" > /dev/null 2>&1; then
    echo "Done!"
    return 0
  else
    echo "Failed!"
    return 1
  fi
}

# Copy model files from Docker container
echo "Copying model files from Docker container..."
copy_file "/app/model/model.ckpt.data-00000-of-00001" "./model/" "model checkpoint data" || \
  echo "Warning: Could not copy model checkpoint data file."
copy_file "/app/model/model.ckpt.index" "./model/" "model checkpoint index" || \
  echo "Warning: Could not copy model checkpoint index file."
copy_file "/app/model/model.ckpt.meta" "./model/" "model checkpoint meta" || \
  echo "Warning: Could not copy model checkpoint meta file."

# Copy input video (if it exists in the container)
echo "Copying input video from Docker container..."
copy_file "/app/data/match_178203_half_res.mp4" "./data/" "input video" || \
  echo "Warning: Could not copy input video. Please provide it manually."

# Verify files were copied successfully
echo ""
echo "Checking copied files:"
if [ -f "model/model.ckpt.data-00000-of-00001" ] && [ -f "model/model.ckpt.index" ] && [ -f "model/model.ckpt.meta" ]; then
  echo "✓ Model checkpoint files successfully copied."
else
  echo "✗ Some model checkpoint files are missing. Please copy them manually."
fi

if [ -f "data/match_178203_half_res.mp4" ]; then
  echo "✓ Input video file successfully copied."
else
  echo "✗ Input video file is missing. Please copy it manually."
fi

echo ""
echo "Done!" 