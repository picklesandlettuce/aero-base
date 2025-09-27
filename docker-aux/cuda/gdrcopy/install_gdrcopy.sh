#!/bin/bash
set -eou pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Instructions derived from
# https://github.com/dusty-nv/jetson-containers/tree/770099e94317753a7bfbf1ca1bd4aa41508c334b/packages/cuda/gdrcopy

# Archive source
# wget https://apt.jetson-ai-lab.io/jp6/cu126/gdrcopy-2.5.1.tar.gz

tar -xzvf gdrcopy-2.5.1.tar.gz -C /usr/local
ldconfig