ARG BASE_IMAGE
FROM ${BASE_IMAGE}


# Install hard pinned wheels for critical ML libraries so that they do not get clobbered by future pip installs
ENV WHEEL_CACHE=/opt/wheels

RUN mkdir /opt/wheels && cp /usr/src/tensorrt/python/tensorrt-*-cp310-*.whl /opt/wheels/ && pip3 install /opt/wheels/*.whl
RUN --mount=type=bind,source=wheels,target=/opt/wheels,readonly \
    ls /opt/wheels && \
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

# Copy wheels refrenced in the requirements.txt from the bind mount to /opt/wheels for reinstalls later
RUN --mount=type=bind,source=wheels,target=/tmp/wheels,readonly \
    --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt,readonly \
    bash -c "set -euxo pipefail \
      && mkdir -p \"${WHEEL_CACHE:?}\" \
      && shopt -s nullglob \
      && for w in /usr/src/tensorrt/python/tensorrt-*-cp*.whl; do mv \"\$w\" \"${WHEEL_CACHE}/\"; done \
      && shopt -u nullglob \
      && while IFS= read -r line; do \
           pkg=\$(echo \"\$line\" | sed 's/#.*//' | awk '{print \$1}' | cut -d\"[\" -f1 | sed 's/[<>=!~].*//'); \
           [ -z \"\$pkg\" ] && continue; \
           for pat in \"\${pkg,,}-\" \"\${pkg//-/_}-\"; do \
             for w in /tmp/wheels/\${pat}*.whl; do [ -e \"\$w\" ] && cp -n \"\$w\" \"${WHEEL_CACHE}/\"; done; \
           done; \
         done < /tmp/requirements.txt \
      && cp -n /tmp/wheels/usd*.whl \"${WHEEL_CACHE}/\" 2>/dev/null || true" \
    && pip install --find-links="${WHEEL_CACHE}" -r /tmp/requirements.txt


COPY requirements.txt /opt/ml_requirements.txt

# Add a global constraint on this requirements.txt to avoid
# future pip package installs from breaking core machine learning libraries
RUN mkdir -p /etc/pip && \
    printf "[global]\nconstraint = /opt/ml_requirements.txt\n" > /etc/pip.conf