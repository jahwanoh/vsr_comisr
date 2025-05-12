# COMISR Video Super-Resolution with CUDA 12.x Support

This repository contains scripts and Dockerfiles to run COMISR (COherent MultI-frame Video Super-Resolution) with CUDA 12.x GPU acceleration.

## Overview

COMISR is a video super-resolution model developed by Google Research that performs 4x upscaling of videos while maintaining temporal coherence across frames.

This repository has been modified to ensure compatibility with CUDA 12.x and modern TensorFlow 2.15.0.

## Quick Start

### Option 1: Fix existing Docker container with CUDA 12.x

If you're already in a Docker container with CUDA 12.x:

```bash
# Copy the fix script to your container
chmod +x cuda12-fix.sh
./cuda12-fix.sh

# Run super-resolution
./run_sr.sh
```

### Option 2: Build a new Docker container with CUDA 12.x support

If you want to create a fresh Docker container:

```bash
# Build the Docker image
./gpu-build.sh

# Run the container with the super-resolution script
./gpu-run.sh
```

### Option 3: Manual installation for CUDA 12.x

If you need to manually install the required CUDA libraries:

```bash
sudo ./install-cuda-libs.sh

# Run super-resolution with GPU
./run-with-gpu.sh
```

## Files

- `video_inference.py` - Main script for video super-resolution
- `run_sr.sh` - Script to run super-resolution with parameters
- `cuda12-fix.sh` - Script to fix GPU compatibility issues in existing environments
- `install-cuda-libs.sh` - Script to install CUDA libraries
- `run-with-gpu.sh` - Script to run with explicit GPU configuration
- `Dockerfile.cuda` - Dockerfile with CUDA 12.x support
- `gpu-build.sh` - Script to build the Docker image
- `gpu-run.sh` - Script to run the Docker container
- `docker-sr.sh` - Script to run inside the Docker container

## Requirements

The updated dependencies compatible with CUDA 12.x and TensorFlow 2.15.0 are in `requirements.txt`.

## Usage

1. Place your input video in the `data/` directory (default: `data/match_178203_half_res.mp4`)
2. Place model checkpoint files in the `model/` directory
3. Run the appropriate script based on your environment
4. Find the super-resolution output at `data/match_178203_super_res.mp4`

## Parameters

You can customize the super-resolution process by modifying `run_sr.sh`:

- `--vsr_scale`: Super resolution scale (default: 4)
- `--num_resblock`: Number of residual blocks (default: 10)
- `--start_frame`: Starting frame for processing (default: 0)
- `--end_frame`: Ending frame (-1 for all frames)
- `--fps`: Output FPS (0 to use same as input)

## Troubleshooting

If you encounter GPU detection issues:

1. Verify TensorFlow can see your GPU:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

2. Try running `cuda12-fix.sh` which includes diagnostics and fixes
3. Make sure CUDA 12.x drivers are properly installed
4. Check that TensorFlow 2.15.0 is installed (required for CUDA 12.x)

## License

This project includes code from Google Research under the Apache License 2.0.

# COMISR:Compression-Informed Video Super-Resolution

This repo contains the testing code for the paper in the ICCV 2021.
["COMISR: Compression-Informed Video Super-Resolution"](https://arxiv.org/abs/2105.01237)

*Disclaimer: This is not an official Google product.*

![COMISR sample](resources/comisr.png)

## Pre-requisite

Install dependencies:
```
pip3 install -r requirements.txt
```

The vid4 testing data can be downloaded from: *gs://gresearch/comisr/data/*
[gcloud sdk](https://cloud.google.com/sdk/docs/install)

The folder path should be similar to:\
.../testdata/lr_crf25/calendar\
.../testdata/lr_crf25/city\
.../testdata/lr_crf25/foliage\
.../testdata/lr_crf25/walk

.../testdata/hr/calendar\
.../testdata/hr/city\
.../testdata/hr/foliage\
.../testdata/hr/walk

## Creating compressed frames
We use [ffmpeg](https://www.ffmpeg.org/) to compress video frames. Below is one sample CLI usage.

Suppose you have a sequence of frames in im%2d.png format, e.g. calendar from vid4.

```shell
ffmpeg -framerate 10 -i im%2d.png -c:v libx264 -crf 0 lossless.mp4 \
&& ffmpeg -i lossless.mp4 -vcodec libx264 -crf 25 crf25.mp4 \
&& ffmpeg -ss 00:00:00 -t 00:00:10 -i crf25.mp4 -r 10 crf25_%2d.png
```

## Pre-trained Model
The pre-trained model can be downloaded from: *gs://gresearch/comisr/model/*


## Usage
```shell
docker run --rm --gpus all -v /home/research/data:/mnt/data -v /home/research/model/comisr:/mnt/model -v /home/research/output:/mnt/output comisr -c "bash ./run_sr.sh"

```

python3 /app/video_inference.py \
  --input_video="/app/data/match_178203_half_res.mp4" \
  --output_video="/app/data/match_178203_super_res.mp4" \
  --checkpoint_path="/app/model/model.ckpt" \
  --vsr_scale=4 \
  --num_resblock=10 \
  --use_ema=true

```shell
python inference_and_eval.py \
--checkpoint_path=/tmp/model.ckpt \
--input_lr_dir=/tmp/lr_4x_crf25 \
--targets=/tmp/hr \
--output_dir=/tmp/output_dir
```

## Citation
If you find this code is useful for your publication, please cite the original paper:
```
@inproceedings{yli_comisr_iccv2021,
  title = {COMISR: Compression-Informed Video Super-Resolution},
  author = {Yinxiao Li and Pengchong Jin and Feng Yang and Ce Liu and Ming-Hsuan Yang and Peyman Milanfar},
  booktitle = {ICCV},
  year = {2021}
}
```


