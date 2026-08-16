FROM nvcr.io/nvidia/pytorch:26.03-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    CCACHE_DIR=/ccache \
    CCACHE_MAXSIZE=10G \
    CCACHE_COMPRESS=1 \
    CCACHE_COMPILERCHECK=content \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CMAKE_CUDA_COMPILER_LAUNCHER=ccache

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        ccache \
        git \
        pkg-config \
        libpugixml-dev \
        libopencv-dev \
        clang-tidy \
        cppcheck \
        doxygen \
        graphviz \
    && rm -rf /var/lib/apt/lists/* \
    && ccache --set-config=max_size=10G \
    && ccache --set-config=compression=true

WORKDIR /app

CMD ["/bin/bash"]
