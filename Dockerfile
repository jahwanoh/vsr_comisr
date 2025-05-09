FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    ffmpeg \
    git \
    wget \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as default
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Create working directory
WORKDIR /comisr_app

# Copy project structure
# Assuming your project root has a 'comisr' subdir for the library
# and scripts like video_inference.py are at the root or in a 'scripts' subdir.
# Adjust paths as needed.
COPY ./comisr /comisr_app/comisr
COPY ./video_inference.py /comisr_app/
# COPY ./train_comisr.py /comisr_app/ # If you have the training script
COPY ./requirements.txt /comisr_app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# For tensorflow_addons, sometimes it needs to be built or find specific headers.
# The pip install should handle it with the devel image.

# Set up environment variables
ENV PYTHONPATH=/comisr_app # So `from comisr.lib import ...` works
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# LD_LIBRARY_PATH should be correctly set by nvidia/cuda image

# Expose a port if needed for any service (not typical for inference script)
# EXPOSE 8888

# Set entrypoint or command
# For development, an entrypoint to bash is fine.
# For running a specific script:
# ENTRYPOINT ["python", "video_inference.py"]
# CMD ["--input_video", "sample.mp4", "--output_video", "output.mp4", "--checkpoint_path", "/checkpoints/model.ckpt-XXX"]
ENTRYPOINT ["/bin/bash"]