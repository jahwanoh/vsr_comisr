#!/bin/bash

# This script runs COMISR super-resolution with CUDA 12.x
set -e  # Exit on error

# Setup directories
mkdir -p data model

# Fix module structure if needed
if [ ! -d "comisr" ] && [ -d "lib" ]; then
  echo "Creating comisr module structure..."
  mkdir -p comisr/lib
  cp lib/*.py comisr/lib/
  echo "# COMISR package" > comisr/__init__.py
  echo "# COMISR lib package" > comisr/lib/__init__.py
fi

# Parameters
INPUT_VIDEO="data/match_178203_half_res.mp4"
OUTPUT_VIDEO="data/match_178203_super_res.mp4"
CHECKPOINT="model/model.ckpt"
VSR_SCALE=4
NUM_RESBLOCK=10
START_FRAME=0
END_FRAME=-1
FPS=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input=*)
      INPUT_VIDEO="${1#*=}"
      shift
      ;;
    --output=*)
      OUTPUT_VIDEO="${1#*=}"
      shift
      ;;
    --checkpoint=*)
      CHECKPOINT="${1#*=}"
      shift
      ;;
    --scale=*)
      VSR_SCALE="${1#*=}"
      shift
      ;;
    --blocks=*)
      NUM_RESBLOCK="${1#*=}"
      shift
      ;;
    --start=*)
      START_FRAME="${1#*=}"
      shift
      ;;
    --end=*)
      END_FRAME="${1#*=}"
      shift
      ;;
    --fps=*)
      FPS="${1#*=}"
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --input=FILE     Input video file (default: $INPUT_VIDEO)"
      echo "  --output=FILE    Output video file (default: $OUTPUT_VIDEO)"
      echo "  --checkpoint=DIR Checkpoint directory (default: $CHECKPOINT)"
      echo "  --scale=N        Super resolution scale (default: $VSR_SCALE)"
      echo "  --blocks=N       Number of resblocks (default: $NUM_RESBLOCK)"
      echo "  --start=N        Start frame (default: $START_FRAME)"
      echo "  --end=N          End frame (default: $END_FRAME)"
      echo "  --fps=N          Output FPS (default: same as input)"
      echo "  --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check prerequisites
if ! command -v python3 &> /dev/null; then
  echo "Error: python3 not found. Please install Python 3."
  exit 1
fi

# Set environment variables for GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PYTHONPATH=.:$PYTHONPATH

# Check GPU detection
echo "Checking GPU detection..."
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Run super-resolution
echo "Starting super-resolution with CUDA 12.x GPU acceleration..."
echo "Input: $INPUT_VIDEO"
echo "Output: $OUTPUT_VIDEO"
echo "Scale: ${VSR_SCALE}x"

python3 video_inference.py \
  --input_video="${INPUT_VIDEO}" \
  --output_video="${OUTPUT_VIDEO}" \
  --checkpoint_path="${CHECKPOINT}" \
  --vsr_scale="${VSR_SCALE}" \
  --num_resblock="${NUM_RESBLOCK}" \
  --use_ema=true \
  --start_frame="${START_FRAME}" \
  --end_frame="${END_FRAME}" \
  --fps="${FPS}"

# Check result
if [ -f "$OUTPUT_VIDEO" ]; then
  echo "Super-resolution completed successfully!"
  echo "Output saved to: $OUTPUT_VIDEO"
else
  echo "Error: Output file not created."
  echo "Try running ./cuda12-fix.sh to fix CUDA 12.x compatibility issues."
fi 