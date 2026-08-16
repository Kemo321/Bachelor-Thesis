cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project(MiniC_DL LANGUAGES CXX CUDA)

option(USE_CUDA "Enable CUDA" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}")
  file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _cuda_root)
  set(CUDA_TOOLKIT_ROOT_DIR "${_cuda_root}")
  set(CUDAToolkit_ROOT "${_cuda_root}")
elseif(EXISTS "/usr/local/cuda")
  set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
  set(CUDAToolkit_ROOT /usr/local/cuda)
endif()

find_package(CUDAToolkit REQUIRED)

# CUDA 13.x toolkits may ship without an imported CUDA::cudnn target.
if(NOT TARGET CUDA::cudnn)
  if(NOT CUDNN_ROOT)
    if(DEFINED ENV{CUDNN_ROOT})
      set(CUDNN_ROOT "$ENV{CUDNN_ROOT}")
    endif()
  endif()
  if(CUDNN_ROOT)
    file(TO_CMAKE_PATH "${CUDNN_ROOT}" CUDNN_ROOT)
  endif()
  find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS ${CUDNN_ROOT} ${CUDNN_ROOT}/include
                  "$ENV{CUDA_PATH}/include"
                  /usr/include
                  /usr/local/cuda/include
            PATH_SUFFIXES include)
  find_library(CUDNN_LIBRARY NAMES cudnn cudnn64_9 cudnn64_8
               HINTS ${CUDNN_LIBRARY_DIR} ${CUDNN_ROOT} ${CUDNN_ROOT}/lib ${CUDNN_ROOT}/lib/x64
                     "$ENV{CUDA_PATH}/lib/x64"
                     /usr/lib/x86_64-linux-gnu
                     /usr/local/cuda/lib64
               PATH_SUFFIXES lib lib/x64 lib64)
  if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    add_library(CUDA::cudnn SHARED IMPORTED)
    if(WIN32)
      set_target_properties(CUDA::cudnn PROPERTIES
        IMPORTED_IMPLIB "${CUDNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}")
      file(GLOB _cudnn_extra_libs "${CUDNN_LIBRARY_DIR}/cudnn_*.lib")
      foreach(_extra_lib IN LISTS _cudnn_extra_libs)
        get_filename_component(_extra_name "${_extra_lib}" NAME_WE)
        if(NOT TARGET CUDA::${_extra_name})
          add_library(CUDA::${_extra_name} SHARED IMPORTED)
          set_target_properties(CUDA::${_extra_name} PROPERTIES IMPORTED_IMPLIB "${_extra_lib}")
          target_link_libraries(CUDA::cudnn INTERFACE CUDA::${_extra_name})
        endif()
      endforeach()
    else()
      set_target_properties(CUDA::cudnn PROPERTIES
        IMPORTED_LOCATION "${CUDNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}")
    endif()
    message(STATUS "Using cuDNN: ${CUDNN_LIBRARY}")
  else()
    message(FATAL_ERROR "cuDNN was not found. Set CUDNN_ROOT / CUDNN_LIBRARY_DIR.")
  endif()
endif()

# Prefer the host GPU (`native`, CMake 3.24+ / recent nvcc). CI runners and older
# CMake get a portable fallback list instead of a hardcoded 75/120.
set(_deeplearn_cuda_arch_fallback "70;75;80;86;89;90")
if(NOT CMAKE_CUDA_ARCHITECTURES)
  if(DEFINED ENV{GITHUB_ACTIONS} OR DEFINED ENV{CI})
    set(CMAKE_CUDA_ARCHITECTURES "${_deeplearn_cuda_arch_fallback}" CACHE STRING "CMake CUDA architectures")
  elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CMake CUDA architectures")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "${_deeplearn_cuda_arch_fallback}" CACHE STRING "CMake CUDA architectures")
  endif()
endif()

if(NOT DEFINED TORCH_CUDA_ARCH_LIST OR TORCH_CUDA_ARCH_LIST STREQUAL "")
  if(DEFINED ENV{TORCH_CUDA_ARCH_LIST} AND NOT "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
    set(TORCH_CUDA_ARCH_LIST "$ENV{TORCH_CUDA_ARCH_LIST}" CACHE STRING "LibTorch CUDA architectures")
  elseif(DEFINED ENV{GITHUB_ACTIONS} OR DEFINED ENV{CI})
    set(TORCH_CUDA_ARCH_LIST "7.5" CACHE STRING "LibTorch CUDA architectures")
  else()
    set(TORCH_CUDA_ARCH_LIST "Common" CACHE STRING "LibTorch CUDA architectures")
  endif()
endif()

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
endif()

# OpenCV 4.10 Windows packs ship vc16/vc17 only. VS 2026 is MSVC 19.50+ and
# is otherwise skipped by OpenCVConfig.cmake's runtime mapping.
if(MSVC AND MSVC_VERSION GREATER_EQUAL 1950 AND NOT DEFINED OpenCV_RUNTIME)
  set(OpenCV_ARCH x64)
  set(OpenCV_RUNTIME vc16)
endif()

find_package(OpenCV QUIET)
if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
  message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
  message(WARNING "OpenCV not found. Building the core GPU library and unit tests only; dataset/utils/benchmarks are skipped.")
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  set(_python_exe "${Python3_EXECUTABLE}")
else()
  set(_python_exe python3)
endif()

execute_process(
    COMMAND "${_python_exe}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_CMAKE_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
    message(STATUS "Szukanie LibTorch (Python): ${TORCH_CMAKE_PATH}")
else()
    file(GLOB FALLBACK_TORCH_PATH "/usr/local/lib/python3.*/dist-packages/torch/share/cmake/Torch")
    list(APPEND CMAKE_PREFIX_PATH ${FALLBACK_TORCH_PATH})
    message(STATUS "Szukanie LibTorch (Fallback ścieżka): ${FALLBACK_TORCH_PATH}")
endif()

set(CAFFE2_USE_KINETO OFF CACHE BOOL "" FORCE)
set(kineto_LIBRARIES "")

set(DEEPLEARN_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "LibTorch found for baseline targets.")
else()
    message(STATUS "LibTorch not found; torch baseline targets will be skipped.")
endif()

# LibTorch's CUDA cmake overwrites CMAKE_CUDA_ARCHITECTURES. Restore ours
# (`native` locally, fallback list or an explicit -D in CI).
set(CMAKE_CUDA_ARCHITECTURES "${DEEPLEARN_CUDA_ARCHITECTURES}" CACHE STRING "CMake CUDA architectures" FORCE)
message(STATUS "DeepLearnLib CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "LibTorch TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")

add_subdirectory(extern/googletest EXCLUDE_FROM_ALL)

set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(extern/highway EXCLUDE_FROM_ALL)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(extern/benchmark EXCLUDE_FROM_ALL)

set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/cutlass")

find_package(pugixml QUIET)
if(NOT pugixml_FOUND)
  add_subdirectory(extern/pugixml EXCLUDE_FROM_ALL)
endif()

include(FetchContent)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(nlohmann_json)

set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.12.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(spdlog)

file(COPY "${CMAKE_SOURCE_DIR}/config" DESTINATION "${CMAKE_BINARY_DIR}")

add_subdirectory(src)
if(Torch_FOUND)
  add_subdirectory(torch_baseline)
endif()
add_subdirectory(tests)
if(OpenCV_FOUND)
  add_subdirectory(benchmarks)
endif()

# Tabular demo is non-vision: keep it available when OpenCV (and thus benchmarks/) is missing.
if(NOT TARGET tabular_demo)
  add_executable(tabular_demo benchmarks/tabular_demo.cpp)
  target_link_libraries(tabular_demo PRIVATE DeepLearnLib nlohmann_json::nlohmann_json spdlog::spdlog)
  target_include_directories(tabular_demo PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/benchmarks")
  file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" _deeplearn_source_dir)
  target_compile_definitions(tabular_demo PRIVATE DEEPLEARN_SOURCE_DIR="${_deeplearn_source_dir}")
  set_target_properties(tabular_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
endif()

option(USE_SANITIZERS "Enable ASan and UBSan" ON)
if(USE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT MSVC)
  target_compile_options(
    DeepLearnLib PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address
                        -fsanitize=undefined -fno-omit-frame-pointer>)

  target_link_options(
    DeepLearnLib PUBLIC $<$<LINK_LANGUAGE:CXX>:-fsanitize=address
    -fsanitize=undefined>)

  message(STATUS "Sanitizers enabled for DeepLearnLib")
endif()
