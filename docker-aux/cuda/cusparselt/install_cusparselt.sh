#!/bin/bash
set -eou pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Instructions derived from
# https://github.com/dusty-nv/jetson-containers/tree/770099e94317753a7bfbf1ca1bd4aa41508c334b/packages/cuda/cusparselt

# Archive source
# wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.8.0.4_cuda12-archive.tar.xz

CUDA_MAJOR="12"
ARCHIVE="libcusparse_lt-linux-aarch64-0.8.0.4_cuda12-archive.tar.xz"
WORK="/tmp/cusparselt"
EXTRACT="${WORK}/extract"
mkdir -p "$WORK" "$EXTRACT"
# wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate -O libcusparse_lt-linux-aarch64-0.8.0.4_cuda12-archive.tar.xz https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.8.0.4_cuda12-archive.tar.xz

echo "Extracting $ARCHIVE ..."
tar -xJf "$SCRIPT_DIR/$ARCHIVE" --strip-components=1 -C "$EXTRACT"
cd "$EXTRACT"

# Install headers under a CUDA-majored include prefix (JP6: /usr/include/libcusparseLt/12)
HDR_DST="/usr/include/libcusparseLt/${CUDA_MAJOR}"
install -d "$HDR_DST"
install -m 0644 include/cusparseLt*.h "$HDR_DST/"

# Install libs into CUDA Tegra target lib dir
LIB_DST="/usr/local/cuda/targets/aarch64-linux/lib"
install -d "$LIB_DST"
install -m 0644 lib/libcusparseLt.so* "$LIB_DST/"

# Ensure the dynamic linker can find them
echo "$LIB_DST" >/etc/ld.so.conf.d/cusparselt.conf
ldconfig

echo "Installed headers to $HDR_DST"
echo "Installed libraries to $LIB_DST"

cuobjdump --list-elf "$LIB_DST"/libcusparseLt.so* 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u || true
cuobjdump --dump-ptx "$LIB_DST"/libcusparseLt.so* 2>/dev/null | grep -oE 'compute_[0-9]+' | sort -u || true


rm -rf /var/lib/apt/lists/*
apt-get clean
cd /
rm -rf $WORK
ldconfig