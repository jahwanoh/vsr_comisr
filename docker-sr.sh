#!/bin/bash

# This script is meant to be run inside the Docker container

# Check if the checkpoint files exist
if [ ! -f "/app/model/model.ckpt.index" ]; then
  echo "Error: Checkpoint files not found in /app/model/"
  echo "Please ensure you have mounted the model directory correctly"
  exit 1
fi

# Check if the input video exists
if [ ! -f "/app/data/match_178203_half_res.mp4" ]; then
  echo "Error: Input video not found in /app/data/"
  echo "Please ensure you have mounted the data directory correctly"
  exit 1
fi

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
  echo "You can find this file in the 'data' directory on your host machine"
else
  echo ""
  echo "Super-resolution failed. Please check the error messages above."
fi 