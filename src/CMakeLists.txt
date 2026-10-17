set(DEEPLEARN_SOURCES
    core/Tensor.cpp
    layers/Conv2d.cpp
    layers/MaxPool2d.cpp
    layers/LeakyReLU.cpp
    layers/FullyConnected.cpp
    layers/Gru.cpp
    layers/Flatten.cpp
    core/Network.cpp
    utils/Profiler.cpp
    losses/Losses.cpp
    loaders/CSVLoader.cpp
    loaders/Ecg200Loader.cpp
    loaders/PackedImageLoader.cpp
    utils/Logger.cpp
    core/Precision.cpp
)

if(OpenCV_FOUND)
  list(APPEND DEEPLEARN_SOURCES loaders/ClassificationLoader.cpp)
endif()

add_library(DeepLearnLib ${DEEPLEARN_SOURCES})

if(DEBUG_NUMERICS)
  target_compile_definitions(DeepLearnLib PUBLIC DEBUG_NUMERICS)
  message(STATUS "DEBUG_NUMERICS enabled: NaN/Inf checks after layer passes")
endif()

set(DEEPLEARN_CUDA_SOURCES
    core/Tensor.cpp
    layers/Conv2d.cpp
    layers/LeakyReLU.cpp
    layers/FullyConnected.cpp
    layers/Gru.cpp
    core/Network.cpp
    utils/Profiler.cpp
    losses/Losses.cpp
)

foreach(_cuda_src IN LISTS DEEPLEARN_CUDA_SOURCES)
  set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/${_cuda_src}" PROPERTIES LANGUAGE CUDA)
endforeach()

set_target_properties(
  DeepLearnLib
  PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
             CUDA_STANDARD 17
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON)

target_include_directories(
  DeepLearnLib
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
         $<INSTALL_INTERFACE:include>)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(DeepLearnLib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
endif()

# spdlog's bundled fmt is not a valid C++17 constexpr under nvcc's host GCC
# (CI Debug treats it as an error). Keep spdlog out of CUDA TUs; this flag
# covers any remaining host-side include of the headers.
target_compile_options(DeepLearnLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177,1675>)
if(NOT MSVC)
  # nvcc's host GCC rejects spdlog's fmt constexpr. cl.exe does not accept this flag.
  target_compile_options(DeepLearnLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-invalid-constexpr>)
endif()

if(MSVC)
  target_compile_options(
    DeepLearnLib
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj;/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler;-Xcompiler=/Zc:preprocessor,-bigobj>)
endif()

find_package(Threads REQUIRED)

target_link_libraries(
  DeepLearnLib
  PUBLIC CUDA::cudart
         CUDA::cublas
         CUDA::cudnn
         spdlog::spdlog
         Threads::Threads)

# GRU lives in cuDNN's adv library. The main cudnn import does not always export it.
foreach(_cudnn_adv_name cudnn_adv cudnn_adv_train cudnn_adv_infer)
  string(TOUPPER "${_cudnn_adv_name}" _cudnn_adv_var)
  find_library(
    CUDNN_${_cudnn_adv_var}_LIBRARY
    NAMES ${_cudnn_adv_name}
    HINTS ${CUDNN_ROOT}
          ${CUDNN_LIBRARY_DIR}
          $ENV{CUDNN_ROOT}
          $ENV{CUDA_PATH}
          /usr/lib/x86_64-linux-gnu
          /usr/local/cuda/lib64
    PATH_SUFFIXES lib lib/x64 lib64)
  if(CUDNN_${_cudnn_adv_var}_LIBRARY)
    target_link_libraries(DeepLearnLib PUBLIC ${CUDNN_${_cudnn_adv_var}_LIBRARY})
    message(STATUS "Linking ${_cudnn_adv_name}: ${CUDNN_${_cudnn_adv_var}_LIBRARY}")
  endif()
endforeach()

if(TARGET TBB::tbb)
  target_link_libraries(DeepLearnLib PUBLIC TBB::tbb)
endif()

if(TARGET CUDA::nvtx3)
  target_link_libraries(DeepLearnLib PUBLIC CUDA::nvtx3)
elseif(TARGET CUDA::nvToolsExt)
  target_link_libraries(DeepLearnLib PUBLIC CUDA::nvToolsExt)
endif()

if(OpenCV_FOUND)
  target_link_libraries(DeepLearnLib PUBLIC ${OpenCV_LIBS})
endif()
