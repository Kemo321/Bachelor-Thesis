cmake_minimum_required(VERSION 3.18)
project(MiniC_DL LANGUAGES CXX CUDA)

# SIMD flags (AVX2)
if(NOT MSVC)
    add_compile_options(-mavx2 -mfma -O3)
endif()

# CUDA architecture (adjust to your GPU, e.g. 75 for RTX 2060)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add GoogleTest from extern
add_subdirectory(extern/googletest)

# Add subfolders
add_subdirectory(src)
add_subdirectory(tests)