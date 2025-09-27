#!/bin/bash
set -eou pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Instructions derived from
# https://github.com/dusty-nv/jetson-containers/tree/770099e94317753a7bfbf1ca1bd4aa41508c334b/packages/cuda

# Archive source
# wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-tegra-repo-ubuntu2204-12-6-local_12.6.3-1_arm64.deb

cp cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
chmod 644 /etc/apt/preferences.d/cuda-repository-pin-600
dpkg -i cuda-tegra-repo-ubuntu2204-12-6-local_12.6.3-1_arm64.deb
cp /var/cuda-*-local/cuda-*-keyring.gpg /usr/share/keyrings/


# Install cuda-toolkit and cuda-libraries
apt-get update
apt-get install -y cuda-toolkit-12-6 cuda-libraries-12-6
rm -rf /var/lib/apt/lists/*
apt-get clean


# Install cuda-compat using dpkg to avoid dependency on nvidia-l4t-core
# Ref: https://gitlab.com/nvidia/container-images/l4t-jetpack/-/blob/master/Dockerfile.jetpack_6?ref_type=heads#L89
apt-get update && apt-get download cuda-compat-12-6 \
    && dpkg-deb -R ./cuda-compat-12-6_*_arm64.deb ./cuda-compat \
    && cp -r ./cuda-compat/usr/local/* /usr/local/ \
    && rm -rf ./cuda-compat-12-6_*_arm64.deb ./cuda-compat


dpkg --list | grep cuda

# Purge the local installer to reduce image size
dpkg -P cuda-tegra-repo-ubuntu2204-12-6-local
