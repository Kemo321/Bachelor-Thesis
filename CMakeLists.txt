cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(MiniC_DL LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

add_subdirectory(extern/googletest)

set(HWY_ENABLE_TESTS
    OFF
    CACHE BOOL "Disable Highway tests")
set(HWY_ENABLE_CONTRIB
    OFF
    CACHE BOOL "Disable Highway contrib")
set(HWY_ENABLE_EXAMPLES
    OFF
    CACHE BOOL "Disable Highway examples")
add_subdirectory(extern/highway)

set(BENCHMARK_ENABLE_TESTING
    OFF
    CACHE BOOL "")
add_subdirectory(extern/benchmark)

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
