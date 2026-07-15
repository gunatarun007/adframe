#!/bin/bash

# Setup script for Virtual Product Placement (VPP) SaaS Pipeline
# Tailored for Ubuntu/Linux container environments (e.g. RunPod) with NVIDIA GPUs (L40S)
set -e

echo "=== [1/5] Installing system dependencies for OpenCV and video handling ==="
if [ -x "$(command -v apt-get)" ]; then
    sudo apt-get update && sudo apt-get install -y \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
        wget \
        git-lfs
else
    echo "Warning: apt-get not found. Skipping system package installation. Ensure ffmpeg and OpenGL libraries are installed."
fi

echo "=== [2/5] Installing core Python packages (PyTorch, Hugging Face suite, OpenCV) ==="
# Install PyTorch with CUDA support (targeting L40S)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install diffusers from source for latest Wan 2.1 / VACE support
pip install git+https://github.com/huggingface/diffusers.git

# Install other requirements
pip install \
    transformers \
    accelerate \
    opencv-python \
    pillow \
    numpy \
    scipy \
    decord \
    einops \
    ninja \
    huggingface_hub \
    bitsandbytes

echo "=== [3/5] Cloning and installing Meta's SAM 3 package ==="
if [ -d "sam3" ]; then
    echo "sam3 directory already exists. Pulling latest code..."
    cd sam3
    git pull
    cd ..
else
    git clone https://github.com/facebookresearch/sam3.git
fi

# Install SAM 3 in editable mode
cd sam3
pip install -e .
cd ..

echo "=== [4/5] Downloading SAM 3 / 3.1 Model Checkpoint ==="
mkdir -p models

# Download SAM 3.1 multiplex checkpoint from open community mirror to avoid gated HF authentication block
echo "Downloading sam3.1_multiplex.pt from HF..."
huggingface-cli download research21/sam3.1 sam3.1_multiplex.pt --local-dir models --local-dir-use-symlinks False

echo "=== [5/5] Setup completed successfully! ==="
echo "You can now run the pipeline using:"
echo "python main.py --video <path_to_video> --brand <path_to_brand_asset> --prompt <target_surface>"
