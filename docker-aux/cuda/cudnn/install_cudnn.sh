#!/bin/bash
set -eou pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Instructions derived from
# https://github.com/dusty-nv/jetson-containers/tree/770099e94317753a7bfbf1ca1bd4aa41508c334b/packages/cuda/cudnn

# Archive source
# wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb

dpkg -i cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb
cp /var/cudnn-local-tegra-repo-ubuntu2204-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get install -y cudnn

rm -rf /var/lib/apt/lists/*
apt-get clean
dpkg --list | grep cudnn
dpkg -P cudnn-local-tegra-repo-ubuntu2204-9.3.0
