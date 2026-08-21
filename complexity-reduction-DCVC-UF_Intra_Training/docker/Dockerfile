# syntax=docker/dockerfile:1

# DCVC-UF image-model training environment.
#
# Pinned to the stack the throughput table in train_image_allqp_OpenImage.sh was
# measured on (RTX PRO 6000 Blackwell, sm_120):
#     python 3.10 + torch 2.10.0+cu128 + triton 3.6.0
# torch.compile --compile full --compile_mode max-autotune is the part that is
# sensitive to these versions, so bump them (via the ARGs below) only on purpose.
#
# Why a `devel` CUDA base rather than `runtime`:
#   * the torch wheel ships its own cudart / cudnn / cublas, so the base CUDA is
#     not what runs the model -- only the driver comes from the host, injected
#     by the NVIDIA Container Toolkit;
#   * but torch.compile's inductor shells out to a C++ compiler for the CPU
#     wrapper, and the optional C++/CUDA extensions (src/cpp, src/layers/
#     extensions/inference) need nvcc. `devel` covers both.
#   12.8.1-devel-ubuntu22.04 is also already present on this host, so the build
#   pulls nothing.
#
# The repository itself is NOT copied into the image -- docker-compose.yml
# bind-mounts it at /workspace/DCVC-UF, so editing training code takes effect
# without a rebuild. Only the pip manifest is copied in.

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
# Owner of $HOME inside the image. Keep in sync with the `user:` in
# docker-compose.yml (defaults to this host's uid/gid 1006).
ARG UID=1006
ARG GID=1006

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        tmux \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && python -m pip install --upgrade pip setuptools wheel

# triton comes along as a torch dependency; it is what --compile needs.
RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION}

COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN python -m pip install -r /tmp/requirements-docker.txt \
    && python -c "import torch, torchvision, triton; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('torchvision', torchvision.__version__); \
print('triton', triton.__version__)"

# $HOME for the run-time user. compose mounts a named volume here so the triton
# / inductor autotune caches survive container restarts -- with max-autotune the
# first steps of a fresh cache cost several minutes of compilation, twice (the
# 256x256 graph, then the 512x512 one after epoch 90).
RUN mkdir -p /home/trainer && chown ${UID}:${GID} /home/trainer
ENV HOME=/home/trainer

COPY docker/entrypoint.sh docker/prepare_datasets.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/prepare_datasets.sh \
    && chmod g=u,o=u /etc/passwd /etc/group

# Bind-mount target for this repository (see docker-compose.yml). The training
# scripts address the datasets as ../datasets, so the two have to be siblings
# under /workspace exactly as they are siblings under /home/<user> on the host.
# Everything else (SAVE_DIR, reference checkpoint, RD cache) lives under this
# repo's own checkpoints/.
WORKDIR /workspace/DCVC-UF

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
