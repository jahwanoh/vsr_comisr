#!/bin/bash

# Create directories if they don't exist
mkdir -p data model

# Fix module structure if needed
if [ ! -d "comisr" ] && [ -d "lib" ]; then
  echo "Creating comisr module structure..."
  mkdir -p comisr/lib
  cp lib/*.py comisr/lib/
  
  # Create __init__.py files if they don't exist
  if [ ! -f "comisr/__init__.py" ]; then
    echo "# COMISR package" > comisr/__init__.py
  fi
  
  if [ ! -f "comisr/lib/__init__.py" ]; then
    echo "# COMISR lib package" > comisr/lib/__init__.py
  fi
fi

# Parameters
INPUT_VIDEO="/mnt/data/long/pano_center_600.mp4"
# INPUT_VIDEO="data/match_178203.mp4"
# OUTPUT_VIDEO="data/match_178203_from_full_x4_sr_comisr.mp4"
OUTPUT_VIDEO="/mnt/output/comisr/pano_center_600x4_comisr.mp4"
CHECKPOINT="/mnt/model/model.ckpt"
VSR_SCALE=4
NUM_RESBLOCK=10
USE_EMA=true
START_FRAME=0
END_FRAME=-1
FPS=0
USE_CPU=false
GPU_MEMORY_FRACTION=1.0

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
    --use_cpu=*)
      USE_CPU="${1#*=}"
      shift
      ;;
    --gpu_memory_fraction=*)
      GPU_MEMORY_FRACTION="${1#*=}"
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --input=FILE             Input video file (default: $INPUT_VIDEO)"
      echo "  --output=FILE            Output video file (default: $OUTPUT_VIDEO)"
      echo "  --checkpoint=DIR         Checkpoint directory (default: $CHECKPOINT)"
      echo "  --scale=N                Super resolution scale (default: $VSR_SCALE)"
      echo "  --blocks=N               Number of resblocks (default: $NUM_RESBLOCK)"
      echo "  --start=N                Start frame (default: $START_FRAME)"
      echo "  --end=N                  End frame (default: $END_FRAME)"
      echo "  --fps=N                  Output FPS (default: same as input)"
      echo "  --use_cpu=BOOL           Use CPU instead of GPU (default: $USE_CPU)"
      echo "  --gpu_memory_fraction=N  Fraction of GPU memory to use (0.0-1.0, default: $GPU_MEMORY_FRACTION)"
      echo "  --help                   Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if input video exists
if [ ! -f "$INPUT_VIDEO" ]; then
  echo "Error: Input video file not found: $INPUT_VIDEO"
  echo "Please make sure the input video exists or specify a different file with --input="
  exit 1
fi

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT}.index" ]; then
  echo "Error: Checkpoint files not found at $CHECKPOINT"
  echo "Please make sure the checkpoint files are available or specify a different checkpoint with --checkpoint="
  exit 1
fi

# Print configuration
echo "Starting super-resolution with the following settings:"
echo "Input:              $INPUT_VIDEO"
echo "Output:             $OUTPUT_VIDEO"
echo "Checkpoint:         $CHECKPOINT"
echo "Scale:              $VSR_SCALE"
echo "Resblocks:          $NUM_RESBLOCK"
echo "Start:              $START_FRAME"
echo "End:                $END_FRAME"
echo "FPS:                $FPS"
echo "Use CPU:            $USE_CPU"
echo "GPU Memory Fraction: $GPU_MEMORY_FRACTION"
echo ""

# Set PYTHONPATH to include current directory
export PYTHONPATH=.:$PYTHONPATH

# Set TensorFlow memory management environment variables
if [ "$USE_CPU" = "true" ]; then
  export CUDA_VISIBLE_DEVICES="-1"
  echo "Running on CPU only mode"
fi

# Set GPU memory fraction
export TF_GPU_MEMORY_FRACTION=$GPU_MEMORY_FRACTION
echo "GPU memory fraction set to: $GPU_MEMORY_FRACTION"

# Force TensorFlow to allocate memory dynamically
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Echo the command that will be executed
echo "Executing command:"
echo "python3 video_inference.py \\
  --input_video=\"${INPUT_VIDEO}\" \\
  --output_video=\"${OUTPUT_VIDEO}\" \\
  --checkpoint_path=\"${CHECKPOINT}\" \\
  --vsr_scale=\"${VSR_SCALE}\" \\
  --num_resblock=\"${NUM_RESBLOCK}\" \\
  --use_ema=\"${USE_EMA}\" \\
  --start_frame=\"${START_FRAME}\" \\
  --end_frame=\"${END_FRAME}\" \\
  --fps=\"${FPS}\""
echo ""

# Run the super-resolution script
python3 video_inference.py \
  --input_video="${INPUT_VIDEO}" \
  --output_video="${OUTPUT_VIDEO}" \
  --checkpoint_path="${CHECKPOINT}" \
  --vsr_scale="${VSR_SCALE}" \
  --num_resblock="${NUM_RESBLOCK}" \
  --use_ema="${USE_EMA}" \
  --start_frame="${START_FRAME}" \
  --end_frame="${END_FRAME}" \
  --fps="${FPS}"

gsutil cp data/match_178203_from_full_x2_sr_comisr.mp4 gs://bepro-dev/research/SRGAN/

# Check the exit code
if [ $? -eq 0 ]; then
  echo ""
  echo "Super-resolution completed successfully!"
  echo "Output video saved to: $OUTPUT_VIDEO"
else
  echo ""
  echo "Super-resolution failed. Please check the error messages above."
  echo "If you're experiencing OOM (Out of Memory) errors, try the following:"
  echo "  1. Limit GPU memory:       ./run_sr.sh --gpu_memory_fraction=0.7"
  echo "  2. Process fewer frames:   ./run_sr.sh --start=0 --end=100" 
  echo "  3. Use CPU (much slower):  ./run_sr.sh --use_cpu=true"
  echo "  4. Reduce input video resolution before processing"
  echo ""
  echo "If you're experiencing TensorFlow compatibility issues, consider using the compatible Docker container:"
  echo "  1. Build the container:  ./docker-build-compatible.sh"
  echo "  2. Run the container:    ./docker-run-compatible.sh"
  echo "  3. Inside the container: ./run_sr.sh"
fi
