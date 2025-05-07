#!/bin/bash

# Create directories if they don't exist
mkdir -p data model

# Copy the model checkpoint files from the Docker container if needed
# Uncomment the following lines if you need to copy the model files from Docker
# NOTE: You should have the Docker container running for this to work
# docker cp video-pipeline-sr:/app/model/model.ckpt.data-00000-of-00001 ./model/
# docker cp video-pipeline-sr:/app/model/model.ckpt.index ./model/
# docker cp video-pipeline-sr:/app/model/model.ckpt.meta ./model/

# Parameters
INPUT_VIDEO="data/match_178203_half_res.mp4"
OUTPUT_VIDEO="data/match_178203_super_res.mp4"
CHECKPOINT="model/model.ckpt"

# Run the super-resolution script
python video_inference.py \
  --input_video="${INPUT_VIDEO}" \
  --output_video="${OUTPUT_VIDEO}" \
  --checkpoint_path="${CHECKPOINT}" \
  --vsr_scale=4 \
  --num_resblock=10 \
  --use_ema=true 