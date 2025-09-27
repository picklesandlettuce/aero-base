ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

# CUDA Toolkit Version 12.6 Update 3
# Installs cuda-toolkit, cuda-compat
RUN --mount=type=bind,source=docker-aux/cuda/cuda,target=/tmp/docker-aux/cuda \
    /tmp/docker-aux/cuda/install_cuda.sh

LABEL cuda.version="12.6.3"

# CUDNN Version 9.3.0
RUN --mount=type=bind,source=docker-aux/cuda/cudnn,target=/tmp/docker-aux/cudnn \
    /tmp/docker-aux/cudnn/install_cudnn.sh
LABEL cudnn.version="9.3.0"

# TensorRT Version 10.3
RUN --mount=type=bind,source=docker-aux/tensorrt,target=/tmp/docker-aux/tensorrt \
    /tmp/docker-aux/tensorrt/install_tensorrt.sh
LABEL tensorrt.version="10.3"

# Set CUDA Environment Variables
# https://github.com/dusty-nv/jetson-containers/blob/770099e94317753a7bfbf1ca1bd4aa41508c334b/packages/cuda/cuda/Dockerfile#L23-L59
ENV CUDA_HOME="/usr/local/cuda" \
    CUDA_ARCH="tegra-aarch64" \
    CUDA_INSTALLED_VERSION="126" \
    CUDA_ARCH_LIST="87"

ENV NVCC_PATH="$CUDA_HOME/bin/nvcc"

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    CUDAARCHS=${CUDA_ARCH_LIST} \
    CUDA_ARCHITECTURES=${CUDA_ARCH_LIST} \
    CUDA_INSTALLED_VERSION=${CUDA_INSTALLED_VERSION} \
    CUDNN_LIB_PATH="/usr/lib/aarch64-linux-gnu" \
    CUDNN_LIB_INCLUDE_PATH="/usr/include" \
    CMAKE_CUDA_COMPILER=${NVCC_PATH} \
    CUDA_NVCC_EXECUTABLE=${NVCC_PATH} \
    CUDACXX=${NVCC_PATH} \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CUDA_BIN_PATH="${CUDA_HOME}/bin" \
    CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/compat:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    LDFLAGS="-L/usr/local/cuda/lib64${LDFLAGS:+ ${LDFLAGS}}" \
    DEBIAN_FRONTEND=noninteractive
ENV PATH="${PATH}:${CUDA_HOME}/bin" \
    LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}" \
    LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
    LDFLAGS="${LDFLAGS:+${LDFLAGS} }-L/usr/local/cuda/lib64/stubs -Wl,-rpath,/usr/local/cuda/lib64" \
    CPLUS_INCLUDE_PATH="/usr/local/cuda/include:/usr/local/cuda/include/cccl:/usr/include/aarch64-linux-gnu:/usr/include/libcusparseLt/12${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}" \
    C_INCLUDE_PATH="/usr/local/cuda/include:/usr/local/cuda/include/cccl:/usr/include/aarch64-linux-gnu:/usr/include/libcusparseLt/12${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so || true \
 && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so.1 || true


# Install all additional cuda libraries compatible with CUDA 12.6 and tegra-aarch64

# NCCL Version 2.27.7
# NCCL Allows for multi-gpu orchestration.  Potential to share GPU Bound workloads between both Orin Modules in the custom carrier over Ethernet
RUN --mount=type=bind,source=docker-aux/cuda/nccl,target=/tmp/docker-aux/nccl \
    /tmp/docker-aux/nccl/install_nccl.sh
LABEL nccl.version="2.27.7"

# gdrcopy Version 2.5.1
# gdrcopy enables efficent transfer of data from CPU to GPU
RUN --mount=type=bind,source=docker-aux/cuda/gdrcopy,target=/tmp/docker-aux/gdrcopy \
    /tmp/docker-aux/gdrcopy/install_gdrcopy.sh
LABEL gdrcopy.version="2.5.1"

# cuSPARSE Version 0.8.0.4
# cuSPARSE contains a set of GPU-accelerated basic linear algebra subroutines used for handling sparse matrices
RUN --mount=type=bind,source=docker-aux/cuda/cusparselt,target=/tmp/docker-aux/cusparselt \
    /tmp/docker-aux/cusparselt/install_cusparselt.sh
LABEL cusparselt.version="0.8.0.4"

# cuDSS Version 0.6.0
# cuDSS is a GPU-accelerated Direct Sparse Solver library for solving linear systems with very sparse matrices
RUN --mount=type=bind,source=docker-aux/cuda/cudss,target=/tmp/docker-aux/cudss \
    /tmp/docker-aux/cudss/install_cudss.sh
LABEL cudss.version="0.6.0"

