#!/bin/bash
set -eou pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Instructions derived from
# https://github.com/dusty-nv/jetson-containers/tree/770099e94317753a7bfbf1ca1bd4aa41508c334b/packages/cuda/nccl

# Archive source
# wget https://apt.jetson-ai-lab.io/jp6/cu126/nccl-2.27.7.tar.gz

tar -xzvf nccl-2.27.7.tar.gz -C /usr/local
ldconfig