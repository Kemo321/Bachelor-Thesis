cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project(MiniC_DL LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(TORCH_CUDA_ARCH_LIST "10.0" CACHE STRING "CUDA archs" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "100" CACHE STRING "CMake CUDA archs" FORCE)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDAToolkit_ROOT /usr/local/cuda)
find_package(CUDAToolkit REQUIRED)

execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
message(STATUS "Szukanie LibTorch w środowisku: ${TORCH_CMAKE_PATH}")

set(CAFFE2_USE_KINETO OFF CACHE BOOL "" FORCE)
set(kineto_LIBRARIES "")

find_package(Torch REQUIRED)
if(Torch_FOUND)
    message(STATUS "Sukces! LibTorch znaleziony.")
    include_directories(${TORCH_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Nie znaleziono LibTorch!")
endif()

add_subdirectory(extern/googletest EXCLUDE_FROM_ALL)

set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(extern/highway EXCLUDE_FROM_ALL)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(extern/benchmark EXCLUDE_FROM_ALL)

set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/cutlass")

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)

option(USE_SANITIZERS "Enable ASan and UBSan" ON)
if(USE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(
    DeepLearnLib PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address
                        -fsanitize=undefined -fno-omit-frame-pointer>)

  target_link_options(
    DeepLearnLib PUBLIC $<$<LINK_LANGUAGE:CXX>:-fsanitize=address
    -fsanitize=undefined>)

  message(STATUS "Sanitizers enabled for DeepLearnLib")
endif()
