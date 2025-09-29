FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANGUAGE=en_US:en \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8


# Update locale and unminimize the image
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            locales \
            locales-all \
            tzdata && \
        locale-gen en_US $LANG && \
        update-locale LC_ALL=$LC_ALL LANG=$LANG && \
        locale && \
    apt-get install -y --no-upgrade --no-install-recommends man manpages manpages-posix && \
    yes | unminimize && \
    rm -rf /usr/share/locale/*/LC_MESSAGES/*.mo
RUN echo "path-exclude=/usr/share/locale/*/LC_MESSAGES/*.mo" >> /etc/dpkg/dpkg.cfg.d/excludes

# Install core utilities and development tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ubuntu-minimal \
        coreutils \
        build-essential \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        lsb-release \
        pkg-config \
        gnupg \
        git \
        git-lfs \
        gdb \
        wget \
        wget2 \
        curl \
        zip \
        unzip \
        libc6-dev \
        linux-libc-dev \
        libstdc++-11-dev \
        libnuma-dev \
        libibverbs-dev \
        time \
        nano \
        sshpass \
        ssh-client \
        devscripts \
        debhelper \
        fakeroot \
        dkms \
        autoconf \
        automake \
        libtool \
        vim \
        iperf3 \
        iproute2 \
        iptables \
        iputils-ping \
        tcpdump \
        ccache \
        smartmontools \
        usbutils \
        fio \
        net-tools \
        nmap \
        jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    gcc --version && \
    g++ --version
