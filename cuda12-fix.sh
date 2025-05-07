#!/bin/bash

# This script fixes TensorFlow GPU issues for CUDA 12.3 in your Docker environment
echo "=== COMISR CUDA 12.3 Fix Tool ==="
echo "This script will configure TensorFlow to work with your CUDA 12.3 installation."

# Create proper module structure
echo "Setting up module structure..."
if [ ! -d "/app/comisr" ]; then
  mkdir -p /app/comisr/lib
  cp /app/lib/*.py /app/comisr/lib/ 2>/dev/null
  echo "# COMISR package" > /app/comisr/__init__.py
  echo "# COMISR lib package" > /app/comisr/lib/__init__.py
  echo "Module structure created."
else
  echo "Module structure already exists."
fi

# Display current TensorFlow version
echo "Current TensorFlow setup:"
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" 2>/dev/null || echo "TensorFlow not properly installed"

# Create a requirements file for CUDA 12.3
echo "Creating requirements file for CUDA 12.3..."
cat > /tmp/requirements_cuda12.txt << EOF
absl-py>=1.0.0
numpy>=1.22.0,<1.24.0
scipy>=1.9.0
pandas>=2.0.0
keras>=2.15.0
opencv-python>=4.6.0
tensorflow==2.15.0
tensorflow-addons==0.22.0
scikit-image>=0.19.0
matplotlib>=3.5.0
pillow>=9.0.0
h5py>=3.6.0
protobuf>=3.19.0,<4.0.0
tqdm>=4.64.0
ffmpeg-python>=0.2.0
EOF

# Install upgraded packages
echo "Installing compatible packages..."
pip3 install --upgrade pip 
pip3 install -r /tmp/requirements_cuda12.txt

# Set up proper environment variables
echo "Setting up environment variables..."
echo 'export TF_FORCE_GPU_ALLOW_GROWTH=true' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc
echo 'export PYTHONPATH=/app:$PYTHONPATH' >> ~/.bashrc

# Apply to current session
export TF_FORCE_GPU_ALLOW_GROWTH=true
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PYTHONPATH=/app:$PYTHONPATH

# Create GPU test script
echo "Creating GPU test script..."
cat > /tmp/test_gpu.py << 'EOF'
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

if len(tf.config.list_physical_devices('GPU')) > 0:
    print("\nTensorFlow is properly configured to use the GPU!")
    
    # Create a simple test to verify GPU works
    print("Running a small test...")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("Matrix multiplication result:", c.numpy())
    print("GPU test successful!")
else:
    print("\nNo GPU detected by TensorFlow!")
    print("Please check your CUDA installation.")
EOF

# Run the test
echo "Testing GPU detection..."
python3 /tmp/test_gpu.py

echo ""
echo "Setup completed. If GPU detection was successful, you can now run the super-resolution:"
echo "   ./run_sr.sh"
echo ""
echo "If GPU detection failed, run this script to complete installation of CUDA libraries:"
echo "   sudo ./install-cuda-libs.sh" 