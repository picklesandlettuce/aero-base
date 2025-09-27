ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    docker.io \
    iperf3 \
    iproute2 \
    iptables \
    iputils-ping \
    tcpdump \
    ccache \
    unzip \
    smartmontools \
    usbutils \
    fio \
    net-tools \
    nmap \
    jq \
    # openjdk is needed for casc2
    openjdk-21-jre \
    openjdk-21-jre-headless \
    # other pkgs needed for casc2
    libboost-dev \
    libboost-iostreams-dev \
    libboost-chrono1.74-dev \
    libboost-date-time1.74-dev \
    libboost-filesystem1.74-dev \
    libboost-program-options1.74-dev \
    libboost-regex1.74-dev \
    libboost-system1.74-dev \
    libboost-test1.74-dev \
    libboost-thread1.74-dev \
    libxerces-c-dev \
    libczmq-dev \
    liblog4cxx-dev \
    libcgal-dev \
    librdkafka-dev \
    libgsoap-dev \
    gsoap \
    python3-tk \
    gdal-bin \
    libgdal-dev \
    v4l-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# Install additional pip packages
RUN uv pip install --system py-cpuinfo \
                # for casc2
                skyfield \
                scipy \
                pandas \
                matplotlib \
                # for multi-modal project
                mcp \
                fastmcp \
                uvicorn \
                google-adk \
                # for AFRL
                onnx \
                quadprog