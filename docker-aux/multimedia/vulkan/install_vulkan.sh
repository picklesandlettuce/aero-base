#!/bin/bash
set -eou pipefail
workdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $workdir

# Instructions derived from
# https://github.com/dusty-nv/jetson-containers/tree/770099e94317753a7bfbf1ca1bd4aa41508c334b/packages/multimedia/vulkan

# Archive source
# wget https://apt.jetson-ai-lab.io/multiarch/vulkan-sdk-1.4.321.0.tar.gz


apt-get update && apt-get install -y --no-install-recommends \
    gcc-12 \
    g++-12 \
    ninja-build \
    bison \
    ocaml-core \
    xz-utils \
    pkg-config \
    libglm-dev \
    libxcb-dri3-0 \
    libxcb-present0 \
    libpciaccess0 \
    libpng-dev \
    libxcb-keysyms1-dev \
    libxcb-dri3-dev \
    libxcb-glx0-dev \
    libxcb-glx0 \
    libx11-dev \
    libwayland-dev \
    libxrandr-dev \
    libxcb-randr0-dev \
    libxcb-ewmh-dev \
    libx11-xcb-dev \
    liblz4-dev \
    libzstd-dev \
    libxml2-dev \
    wayland-protocols \
    mesa-vulkan-drivers \
    vulkan-tools \
    libxcb-cursor-dev \
    libxcb-xinput0 \
    libxcb-xinerama0 \
    git python-is-python3 python3-jsonschema \
    clang-format qtbase5-dev qt6-base-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean


tar -xzvf vulkan-sdk-1.4.321.0.tar.gz -C /usr/local
ldconfig