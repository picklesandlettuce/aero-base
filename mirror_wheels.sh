#!/bin/bash
set -euo pipefail

REPO_URL="https://pypi.jetson-ai-lab.io/jp6/cu126"
DEST_DIR="./wheels"
TRUSTED_HOST="jetson-ai-lab.io"

mkdir -p "$DEST_DIR"

echo "Mirroring Jetson AI Lab repository to $DEST_DIR"
echo "Repository: $REPO_URL"
echo "=" * 50

PACKAGES=(
    "bitsandbytes==0.48.0.dev0"
    "causal-conv1d==1.5.2"
    "ctranslate2==4.6.0"
    "cuda-python==12.6.2.post1"
    "diffusers==0.36.0.dev0"
    "flash-attn==2.8.4"
    "flashinfer-python==0.3.1"
    "flex-prefill==0.1.0"
    "genai-bench==0.0.2"
    "hloc==1.5"
    "kaolin==0.18.0"
    "llama-cpp-python==0.3.16"
    "mamba-ssm==2.2.5"
    "minference==0.1.6.0"
    "mistral-common==1.8.4"
    "nerfacc==0.5.3"
    "nerfview==0.1.4"
    "nvidia-cudnn-frontend==1.14.0"
    "nvidia-cutlass==4.0.0.0"
    "onnxruntime-gpu==1.23.0"
    "opencv-contrib-python==4.12.0"
    "pyceres==2.5"
    "pycolmap==3.13.0.dev0"
    "pycuda==2025.1.1"
    "pycute==4.1.0"
    "pymeshlab==2025.7"
    "pytorch3d==0.7.8"
    "sgl-kernel==0.3.9.post2"
    "sglang==0.5.2"
    "tinycudann==2.0"
    "torch==2.8.0"
    "torch-memory-saver==0.0.9"
    "torch-tensorrt==2.8.0+cu126"
    "torchao==0.14.0"
    "torchaudio==2.8.0"
    "torchcodec==0.6.0"
    "torchvision==0.23.0"
    "triton==3.4.0"
    "unsloth==2025.7.9"
    "usd-core==25.11"
    "vllm==0.10.2+cu126"
    "xformers==0.0.33+ac00641.d20250830"
    "xgrammar==0.1.23"
)

# Function to download a package
download_package() {
    local package="$1"
    echo "Downloading: $package"
    
    pip download \
        --index-url "$REPO_URL" \
        --trusted-host "$TRUSTED_HOST" \
        --dest "$DEST_DIR" \
        --no-deps \
        "$package" || echo "Failed to download: $package"
}

# Download all packages
for package in "${PACKAGES[@]}"; do
    download_package "$package"
done

echo ""
echo "Download complete!"
echo "Wheels saved to: $DEST_DIR"
echo "Total files downloaded:"
ls -1 "$DEST_DIR" | wc -l
