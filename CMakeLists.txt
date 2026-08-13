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
            PATH_SUFFIXES include)
  find_library(CUDNN_LIBRARY NAMES cudnn cudnn64_9 cudnn64_8
               HINTS ${CUDNN_LIBRARY_DIR} ${CUDNN_ROOT} ${CUDNN_ROOT}/lib ${CUDNN_ROOT}/lib/x64
                     "$ENV{CUDA_PATH}/lib/x64"
               PATH_SUFFIXES lib lib/x64)
  if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    add_library(CUDA::cudnn SHARED IMPORTED)
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
    message(STATUS "Using cuDNN: ${CUDNN_LIBRARY}")
  else()
    message(FATAL_ERROR "cuDNN was not found. Set CUDNN_ROOT / CUDNN_LIBRARY_DIR.")
  endif()
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CMake CUDA archs")
endif()
if(NOT DEFINED TORCH_CUDA_ARCH_LIST OR TORCH_CUDA_ARCH_LIST STREQUAL "" OR TORCH_CUDA_ARCH_LIST STREQUAL "native")
  set(TORCH_CUDA_ARCH_LIST "12.0" CACHE STRING "CUDA archs" FORCE)
endif()

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
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

find_package(pugixml QUIET)
if(NOT pugixml_FOUND)
  add_subdirectory(extern/pugixml EXCLUDE_FROM_ALL)
endif()

add_subdirectory(src)
add_subdirectory(tests)
if(OpenCV_FOUND)
  add_subdirectory(benchmarks)
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
