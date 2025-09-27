#!/bin/bash
set -eou pipefail
set -x
workdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $workdir

# Instructions derived from
# https://github.com/dusty-nv/jetson-containers/tree/770099e94317753a7bfbf1ca1bd4aa41508c334b/packages/multimedia/video-codec-sdk

# Archive source
# wget https://apt.jetson-ai-lab.io/multiarch/Video_Codec_SDK_13.0.19.zip

NV_CODEC_VERSION=13.0.19
ZIP="Video_Codec_SDK_$NV_CODEC_VERSION.zip"
URL="https://apt.jetson-ai-lab.io/multiarch/$ZIP"


if [ ! -f $ZIP ]; then
  wget $URL
fi

NV_CODEC_ROOT=/opt/nvidia/video-codec-sdk
mkdir -p $NV_CODEC_ROOT
unzip $ZIP -d $NV_CODEC_ROOT
mv $NV_CODEC_ROOT/Video_Codec_SDK_*/* $NV_CODEC_ROOT/
rmdir $NV_CODEC_ROOT/Video_Codec_SDK_*

# Copy libraries to system locations
cp $NV_CODEC_ROOT/Lib/linux/stubs/$(uname -m)/*.so /usr/local/lib
cp $NV_CODEC_ROOT/Lib/linux/stubs/$(uname -m)/*.so $CUDA_HOME/lib64

# Copy headers to system locations
cp $NV_CODEC_ROOT/Interface/* /usr/local/include
cp $NV_CODEC_ROOT/Interface/* $CUDA_HOME/include
