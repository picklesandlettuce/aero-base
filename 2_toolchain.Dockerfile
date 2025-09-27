ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install basics for Python, wait for ml layer to install libraries for runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install rust/cargo
ENV PATH="/root/.cargo/bin:${PATH}"
RUN --mount=type=bind,source=docker-aux/toolchain/install_rust.sh,target=/tmp/install_rust.sh  \
    /tmp/install_rust.sh


# Install protobuf cpp
ENV PROTOBUF_VERSION="3.20.3"
RUN --mount=type=bind,source=docker-aux/toolchain/install_protobuf.sh,target=/tmp/install_protobuf.sh  \
    /tmp/install_protobuf.sh
LABEL protobuf.version="3.20.3"


# Install Bazel
# https://github.com/bazelbuild/bazelisk
RUN BAZELISK_RELEASE=$(wget -qO- https://api.github.com/repos/bazelbuild/bazelisk/releases/latest | grep -Po '"tag_name": "\K.*?(?=")') && \
    BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/$BAZELISK_RELEASE/bazelisk-linux-arm64" && \
    echo "BAZELISK_RELEASE=$BAZELISK_RELEASE" && echo "BAZELISK_URL=$BAZELISK_URL" && \
    wget $WGET_FLAGS $BAZELISK_URL -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel && \
    bazel --version

# Install llvm20
ENV LLVM_VERSION="20"
ENV PATH="/root/.cargo/bin:${PATH}"
RUN --mount=type=bind,source=docker-aux/toolchain/install_llvm.sh,target=/tmp/install_llvm.sh  \
    /tmp/install_llvm.sh
LABEL llvm.version="20"