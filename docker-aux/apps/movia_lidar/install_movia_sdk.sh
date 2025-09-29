#!/bin/bash
set -eou pipefail
set -x
workdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $workdir

SDK_ZIP="MVIS_SDK8.9.1.zip"
TARGET=/opt/mvis_sdk

unzip -q $SDK_ZIP -d $TARGET
cd $TARGET
# Build the SDK, echo to auto-confirm the prompt
echo | ./build.sh -a -c Release -j 4