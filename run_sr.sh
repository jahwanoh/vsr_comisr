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
VSR_SCALE=4
NUM_RESBLOCK=10
USE_EMA=true
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
echo "Input:      $INPUT_VIDEO"
echo "Output:     $OUTPUT_VIDEO"
echo "Checkpoint: $CHECKPOINT"
echo "Scale:      $VSR_SCALE"
echo "Resblocks:  $NUM_RESBLOCK"
echo "Start:      $START_FRAME"
echo "End:        $END_FRAME"
echo "FPS:        $FPS"
echo ""

# Run the super-resolution script
python video_inference.py \
  --input_video="${INPUT_VIDEO}" \
  --output_video="${OUTPUT_VIDEO}" \
  --checkpoint_path="${CHECKPOINT}" \
  --vsr_scale="${VSR_SCALE}" \
  --num_resblock="${NUM_RESBLOCK}" \
  --use_ema="${USE_EMA}" \
  --start_frame="${START_FRAME}" \
  --end_frame="${END_FRAME}" \
  --fps="${FPS}"

# Check the exit code
if [ $? -eq 0 ]; then
  echo ""
  echo "Super-resolution completed successfully!"
  echo "Output video saved to: $OUTPUT_VIDEO"
else
  echo ""
  echo "Super-resolution failed. Please check the error messages above."
  echo "If you're experiencing TensorFlow compatibility issues, consider using the compatible Docker container:"
  echo "  1. Build the container:  ./docker-build-compatible.sh"
  echo "  2. Run the container:    ./docker-run-compatible.sh"
  echo "  3. Inside the container: ./run_sr.sh"
fi 