#!/bin/bash
set -eou pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Instructions derived from
# https://github.com/dusty-nv/jetson-containers/tree/770099e94317753a7bfbf1ca1bd4aa41508c334b/packages/cuda/tensorrt

# Archive source
# wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/tars/TensorRT-10.3.0.26.l4t.aarch64-gnu.cuda-12.6.tar.gz

tar -xvf TensorRT-10.3.0.26.l4t.aarch64-gnu.cuda-12.6.tar.gz -C /usr/src
mv /usr/src/TensorRT-* /usr/src/tensorrt

# Install libraries to system location
cp -r /usr/src/tensorrt/lib/* /usr/lib/$(uname -m)-linux-gnu/

# Install headers to system location  
cp -r /usr/src/tensorrt/include/* /usr/include/$(uname -m)-linux-gnu/

# Install binaries with proper permissions
install -m 755 /usr/src/tensorrt/targets/$(uname -m)-linux-gnu/bin/* /usr/local/bin/

# Wheel is installed after Python install
# PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
# pip3 install /usr/src/tensorrt/python/tensorrt-*-cp${PY_VERSION}-*.whl