#!/bin/bash
set -eou pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Instructions derived from
# https://github.com/dusty-nv/jetson-containers/tree/770099e94317753a7bfbf1ca1bd4aa41508c334b/packages/cuda/cudss

# Archive source
# wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb

dpkg -i cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
cp /var/cudss-local-tegra-repo-ubuntu2204-0.6.0/cudss-local-tegra-72E455D5-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cudss

apt-get remove --purge -y cudss-local-tegra-repo-ubuntu2204-0.6.0
rm -rf /var/cudss-local-tegra-repo-ubuntu2204-0.6.0

rm -rf /var/lib/apt/lists/*
apt-get clean