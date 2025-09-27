#!/usr/bin/env bash
set -ex
pip3 install --no-index --find-links=/opt/wheels --force-reinstall "cmake==$CMAKE_VERSION" 

cmake --version
which cmake

update-alternatives --force --install /usr/bin/cmake cmake "$(which cmake)" 100

ls -ll /usr/bin/cmake*

# Held packages were changed and -y was used without --allow-change-held-packages
#apt-get update
#apt-mark hold cmake
#apt-get clean
#rm -rf /var/lib/apt/lists/*