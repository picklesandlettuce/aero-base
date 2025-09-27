#!/bin/bash
set -eou pipefail
set -x
workdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $workdir

# Instructions derived from
# https://github.com/dusty-nv/jetson-containers/tree/770099e94317753a7bfbf1ca1bd4aa41508c334b/packages/multimedia/ffmpeg

# Archive source
# wget https://apt.jetson-ai-lab.io/jp6/cu126/ffmpeg-7.1.1.tar.gz


apt-get update
apt-get install -y --no-install-recommends \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    libvpx-dev \
    libx264-dev \
    libx265-dev \
    libopus-dev \
    libdav1d-dev
  apt-get clean
  rm -rf /var/lib/apt/lists/*

tar -xzvf ffmpeg-7.1.1.tar.gz -C /usr/local
ldconfig

