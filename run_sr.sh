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
INPUT_VIDEO="/mnt/data/pano_cerberus_center_15.mp4"
# INPUT_VIDEO="data/match_178203.mp4"
# OUTPUT_VIDEO="data/match_178203_from_full_x4_sr_comisr.mp4"
OUTPUT_VIDEO="/mnt/output/comisr/pano_cerberus_center_15x4_comisr.mp4"
CHECKPOINT="/mnt/model/model.ckpt"
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

# Set PYTHONPATH to include current directory
export PYTHONPATH=.:$PYTHONPATH

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
  echo "If you're experiencing TensorFlow compatibility issues, consider using the compatible Docker container:"
  echo "  1. Build the container:  ./docker-build-compatible.sh"
  echo "  2. Run the container:    ./docker-run-compatible.sh"
  echo "  3. Inside the container: ./run_sr.sh"
fi root@bepro-MS-7B51:/home/research/vsr_comisr# ll
total 11556
drwxr-xr-x 8 root root     4096 Mai 12 19:32 ./
drwxr-xr-x 8 root root     4096 Mai 12 18:15 ../
-rw-r--r-- 1 root root 11674797 Mai  9 20:24 2105.01237v2.pdf
drwxr-xr-x 4 root root     4096 Mai 12 12:03 comisr/
-rwxr-xr-x 1 root root     3023 Mai  9 20:24 cuda12-fix.sh*
drwxr-xr-x 5 root root     4096 Mai  9 20:34 data/
drwxr-xr-x 2 root root     4096 Mai 12 15:45 .devcontainer/
-rwxr-xr-x 1 root root      271 Mai  9 20:24 docker-build.sh*
-rw-r--r-- 1 root root     1748 Mai  9 21:36 Dockerfile
-rwxr-xr-x 1 root root      608 Mai  9 20:24 docker-run.sh*
-rwxr-xr-x 1 root root     1996 Mai  9 20:24 docker-sr.sh*
drwxr-xr-x 8 root root     4096 Mai 12 19:33 .git/
-rw-r--r-- 1 root root       24 Mai  9 20:24 .gitignore
-rwxr-xr-x 1 root root      300 Mai  9 20:24 gpu-build.sh*
-rwxr-xr-x 1 root root      195 Mai  9 20:24 gpu-run.sh*
-rw-r--r-- 1 root root    13596 Mai  9 20:24 inference_and_eval.py
-rw-r--r-- 1 root root     3414 Mai  9 20:24 metrics.py
drwxr-xr-x 2 root root     4096 Mai 12 17:33 model/
-rw-r--r-- 1 root root       43 Mai  9 20:24 model_pull.sh
-rw-r--r-- 1 root root     4833 Mai  9 20:24 README.md
-rw-r--r-- 1 root root      480 Mai  9 21:35 requirements.txt
-rwxr-xr-x 1 root root     3222 Mai  9 20:24 run_cuda12.sh*
-rwxr-xr-x 1 root root     4015 Mai  9 20:24 run_sr.sh*
-rw-r--r-- 1 root root    19982 Mai  9 21:35 train_comisr.py
-rw-r--r-- 1 root root    20834 Mai  9 21:35 video_inference.py
drwxr-xr-x 2 root root     4096 Mai 12 15:45 .vscode/
root@bepro-MS-7B51:/home/research/vsr_comisr# vi run_sr.sh 

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

# Set PYTHONPATH to include current directory
export PYTHONPATH=.:$PYTHONPATH

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
  echo "If you're experiencing TensorFlow compatibility issues, consider using the compatible Docker container:"
  echo "  1. Build the container:  ./docker-build-compatible.sh"
  echo "  2. Run the container:    ./docker-run-compatible.sh"
  echo "  3. Inside the container: ./run_sr.sh"
fi
