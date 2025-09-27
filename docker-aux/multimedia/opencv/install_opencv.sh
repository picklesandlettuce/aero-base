#!/bin/bash
set -eou pipefail
set -x
workdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $workdir

# Instructions derived from
# https://github.com/dusty-nv/jetson-containers/tree/770099e94317753a7bfbf1ca1bd4aa41508c334b/packages/cv/opencv

# Archive source
# wget https://apt.jetson-ai-lab.io/jp6/cu126/OpenCV-4.12.0.tar.gz


apt-get update
apt-get purge -y 'libopencv*-dev' 'libopencv-dev' || true
mkdir -p /tmp/opencv-extract
tar -xvzf OpenCV-4.12.0.tar.gz -C /tmp/opencv-extract
cd /tmp/opencv-extract
dpkg -i --force-depends *.deb
apt-get update
apt-get install -y -f --no-install-recommends
dpkg -i *.deb
rm -rf /var/lib/apt/lists/*
apt-get clean


# manage some install paths
PYTHON3_VERSION=`python3 -c 'import sys; version=sys.version_info[:3]; print("{0}.{1}".format(*version))'`

if [ "$(uname -m)" = "aarch64" ]; then
	local_include_path="/usr/local/include/opencv4"
	local_python_path="/usr/local/lib/python${PYTHON3_VERSION}/dist-packages/cv2"

	if [ -d "$local_include_path" ]; then
		echo "$local_include_path already exists, replacing..."
		rm -rf $local_include_path
	fi

	if [ -d "$local_python_path" ]; then
		echo "$local_python_path already exists, replacing..."
		rm -rf $local_python_path
	fi

	ln -sfnv /usr/include/opencv4 $local_include_path
	ln -sfnv /usr/lib/python${PYTHON3_VERSION}/dist-packages/cv2 $local_python_path
fi

cd /
rm -rf /tmp/opencv-extract