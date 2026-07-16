#!/bin/bash
# =============================================================================
# AdFrame 2.0 Environment Setup
# =============================================================================
# Extends adframe 1.0 setup.sh to add Qwen2.5-VL and FLUX.1-dev dependencies.
# Designed for RunPod L40S GPU container (Ubuntu Linux, CUDA 12.1).
# =============================================================================
set -e

echo "=== [1/6] System dependencies ==="
if [ -x "$(command -v apt-get)" ]; then
    apt-get update && apt-get install -y \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
        wget \
        git-lfs \
        --no-install-recommends
else
    echo "Warning: apt-get not found. Ensure ffmpeg and OpenGL libraries are installed."
fi

echo "=== [2/6] Core Python packages (PyTorch + CUDA 12.1) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== [3/6] Diffusers (latest from source for Wan VACE + FLUX) ==="
pip install git+https://github.com/huggingface/diffusers.git

echo "=== [4/6] Transformers, accelerate, and VL utilities ==="
pip install \
    transformers \
    accelerate \
    bitsandbytes \
    opencv-python \
    pillow \
    numpy \
    scipy \
    einops \
    ninja \
    huggingface_hub \
    sentencepiece \
    qwen-vl-utils \
    decord

echo "=== [5/6] Redirect HuggingFace cache to /tmp (avoids RunPod disk quota) ==="
export HF_HOME=/tmp/huggingface
export HF_HUB_CACHE=/tmp/huggingface/hub
export TMPDIR=/tmp
mkdir -p /tmp/huggingface/hub

echo "=== [6/6] Setup complete ==="
echo "Run the AdFrame 2.0 pipeline with:"
echo ""
echo "  cd /workspace/adframe"
echo "  export HF_HOME=/tmp/huggingface"
echo "  export HF_HUB_CACHE=/tmp/huggingface/hub"
echo "  export TMPDIR=/tmp"
echo "  python adframe2.0/run.py \\"
echo "    --video /workspace/demo.mp4 \\"
echo "    --brand /workspace/demo_brand.jpg \\"
echo "    --image_backend flux \\"
echo "    --wan_model Wan-AI/Wan2.1-VACE-1.3B-diffusers \\"
echo "    --seed 42"
