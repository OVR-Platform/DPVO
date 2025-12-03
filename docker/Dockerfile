# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    unzip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY . /app

RUN chmod +x docker/entrypoint.sh

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 && \
    python -m pip install --no-cache-dir torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.3.1+cu121.html && \
    python -m pip install --no-cache-dir tensorboard numba tqdm einops pypose kornia numpy==1.26.4 plyfile evo opencv-python yacs && \
    python -m pip install --no-cache-dir --no-build-isolation .

RUN chmod +x download_models_and_data.sh && \
    ./download_models_and_data.sh && \
    rm -f models.zip movies.zip

ENTRYPOINT ["/app/docker/entrypoint.sh"]
