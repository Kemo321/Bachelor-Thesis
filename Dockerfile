FROM nvcr.io/nvidia/pytorch:26.03-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libpugixml-dev \
    libopencv-dev \
    git \
    wget \
    unzip \
    clang-tidy \
    cppcheck \
    doxygen \
    graphviz \
    ninja-build \
    ccache \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

ENV CCACHE_DIR=/ccache

WORKDIR /app

CMD ["/bin/bash"]