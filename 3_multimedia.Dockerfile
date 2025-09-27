ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install FFmpeg
RUN --mount=type=bind,source=docker-aux/multimedia/ffmpeg,target=/tmp/ffmpeg  \
    /tmp/ffmpeg/install_ffmpeg.sh

# Install OpenGL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglu1-mesa-dev \
        libglx-mesa0 \
        libegl-dev \
        libxrender1 \
        libglfw3-dev \
        libglew-dev \
        glew-utils \
        libopenblas0 \
        mesa-common-dev \
        freeglut3-dev  \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install GStreamer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        lsb-release \
        libunwind-dev \
        libgstreamer1.0-dev \
        gstreamer1.0-tools \
        gstreamer1.0-libav \
        gstreamer1.0-rtsp \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-plugins-rtp \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-good1.0-dev \
        libgstrtspserver-1.0-0 \
        gstreamer1.0-nice \
        libnss-mdns \
        avahi-utils \
        python3-gi \
        python3-gst-1.0 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Install vulkan
RUN --mount=type=bind,source=docker-aux/multimedia/vulkan,target=/tmp/vulkan  \
    /tmp/vulkan/install_vulkan.sh

# Install vc-sdk
RUN --mount=type=bind,source=docker-aux/multimedia/vc-sdk,target=/tmp/vc-sdk  \
    /tmp/vc-sdk/install_vc-sdk.sh

# Install opencv
RUN --mount=type=bind,source=docker-aux/multimedia/opencv,target=/tmp/opencv  \
    /tmp/opencv/install_opencv.sh


