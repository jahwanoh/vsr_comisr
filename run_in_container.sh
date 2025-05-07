#!/bin/bash

# This script is designed to be run inside the video-pipeline-sr container

# Fix module structure
if [ ! -d "/app/comisr" ] && [ -d "/app/lib" ]; then
  echo "Creating comisr module structure..."
  mkdir -p /app/comisr/lib
  cp /app/lib/*.py /app/comisr/lib/
  
  # Create __init__.py files
  echo "# COMISR package" > /app/comisr/__init__.py
  echo "# COMISR lib package" > /app/comisr/lib/__init__.py
  
  echo "Module structure created successfully."
fi

# Set PYTHONPATH
export PYTHONPATH=/app:$PYTHONPATH

# Run the super-resolution process
python /app/video_inference.py \
  --input_video="/app/data/match_178203_half_res.mp4" \
  --output_video="/app/data/match_178203_super_res.mp4" \
  --checkpoint_path="/app/model/model.ckpt" \
  --vsr_scale=4 \
  --num_resblock=10 \
  --use_ema=true

# Check if the process was successful
if [ $? -eq 0 ]; then
  echo ""
  echo "Super-resolution completed successfully!"
  echo "Output video saved to: /app/data/match_178203_super_res.mp4"
else
  echo ""
  echo "Super-resolution failed. Please check the error messages above."
fi 