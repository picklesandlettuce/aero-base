ARG BASE_IMAGE
FROM ${BASE_IMAGE}


# Install hard pinned wheels for critical ML libraries so that they do not get clobbered by future pip installs
ENV WHEEL_CACHE=/opt/wheels
ENV PIP_NO_CACHE_DIR=1
# Now, install TensorRT wheel from the tarball install done in previous layers
# And forcefully install jetson optimized wheels for core machine learning libraries
# which were mirrored from https://developer.download.nvidia.com/compute/redist/jp/v60 (tensorflow)
# and https://pypi.jetson-ai-lab.io/jp6/cu126

# We need both pip install -r requirements.txt and the force installs here because pip's resolver does not always pick the right versions
# even if they are available locally in the wheel cache

# Also the tensorrt wheel is installed manually first because it is a dependency of other jetson optimized wheels
RUN mkdir /opt/wheels && cp /usr/src/tensorrt/python/tensorrt-*-cp310-*.whl /opt/wheels/ && pip3 install /opt/wheels/*.whl

COPY wheels /opt/wheels
RUN ls /opt/wheels && \
    pip install \
        /opt/wheels/tensorflow-*.whl \
        /opt/wheels/torch-*.whl \
        /opt/wheels/torchvision-*.whl \
        /opt/wheels/torchaudio-*.whl \
        /opt/wheels/torch_tensorrt-*.whl \
        /opt/wheels/torchao-*.whl \
        /opt/wheels/torchcodec-*.whl \
        /opt/wheels/flash_attn-*.whl \
        /opt/wheels/xformers-*.whl \
        /opt/wheels/triton-*.whl \
        /opt/wheels/bitsandbytes-*.whl \
        /opt/wheels/pycuda-*.whl \
        /opt/wheels/onnxruntime_gpu-*.whl \
        /opt/wheels/pytorch3d-*.whl \
        /opt/wheels/kaolin-*.whl \
        /opt/wheels/tinycudann-*.whl \
        /opt/wheels/causal_conv1d-*.whl \
        /opt/wheels/llama_cpp_python-*.whl \
        /opt/wheels/ctranslate2-*.whl \
        /opt/wheels/cuda_python-*.whl \
        /opt/wheels/nvidia_cudnn_frontend-*.whl \
        /opt/wheels/nvidia_cutlass-*.whl \
        /opt/wheels/pycolmap-*.whl \
        /opt/wheels/pyceres-*.whl \
        /opt/wheels/nerfacc-*.whl \
        /opt/wheels/usd*.whl

# Now, install all other pinnned pip packages using the requirements.txt file
COPY requirements.txt /opt/core_requirements.txt
RUN pip download --find-links="${WHEEL_CACHE}" -r /opt/core_requirements.txt -d /opt/wheels && \
    pip install --no-index --find-links=/opt/wheels -r /opt/core_requirements.txt

# Add a global constraint on this requirements.txt to avoid
# future pip package installs from breaking core machine learning libraries

# This means that any future pip installs will take into account any core machine learning library versions
# and their dependencies, and will not be able to upgrade/downgrade them to incompatible versions

# If a collaborator *absolutely* needs a package that breaks the core libraries,
# they will have to make a venv for their use case like "python3 -m venv --system-site-packages myenv" and install packages there
# with the core libraries in this image, but having this global constraints file ensures that we don't accidentally break the core libraries
# with future pip installs
RUN mkdir -p /etc/pip && \
    cat <<EOF > /etc/pip.conf
[global]
constraint = /opt/core_requirements.txt
find-links = /opt/wheels
EOF