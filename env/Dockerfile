FROM ubuntu:20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python-is-python3 \
    cmake \
    build-essential \
    libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
    torch \
    torchvision \
    && pip install \
    opencv-python \
    matplotlib

WORKDIR /work
