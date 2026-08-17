set(DEEPLEARN_SOURCES
    Tensor.cpp
    Conv2d.cpp
    MaxPool2d.cpp
    BatchNorm2d.cpp
    LeakyReLU.cpp
    Dropout.cpp
    FullyConnected.cpp
    Flatten.cpp
    Network.cpp
    YOLOLoss.cpp
    Profiler.cpp
    mAP.cpp
    Losses.cpp
    Softmax.cpp
    CSVLoader.cpp
    Logger.cpp
    Precision.cpp
    FusedCBR2d.cpp
)

if(OpenCV_FOUND)
  list(APPEND DEEPLEARN_SOURCES dataset.cpp utils.cpp ClassificationLoader.cpp)
endif()

add_library(DeepLearnLib ${DEEPLEARN_SOURCES})

if(DEBUG_NUMERICS)
  target_compile_definitions(DeepLearnLib PUBLIC DEBUG_NUMERICS)
  message(STATUS "DEBUG_NUMERICS enabled: NaN/Inf checks after layer passes")
endif()

set(DEEPLEARN_CUDA_SOURCES
    Tensor.cpp
    Conv2d.cpp
    BatchNorm2d.cpp
    LeakyReLU.cpp
    Dropout.cpp
    FullyConnected.cpp
    Network.cpp
    YOLOLoss.cpp
    Profiler.cpp
    Losses.cpp
    Softmax.cpp
    FusedCBR2d.cpp
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
         $<INSTALL_INTERFACE:include>
         "${CMAKE_SOURCE_DIR}/extern/stb"
         ${CUTLASS_DIR}/include
         ${CUTLASS_DIR}/tools/util/include)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(DeepLearnLib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
endif()

# spdlog's bundled fmt is not a valid C++17 constexpr under nvcc's host GCC
# (CI Debug treats it as an error). Keep spdlog out of CUDA TUs; this flag
# covers any remaining host-side include of the headers.
target_compile_options(
  DeepLearnLib
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177,1675>
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-invalid-constexpr>)

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
         hwy
         spdlog::spdlog
         Threads::Threads)

if(TARGET TBB::tbb)
  target_link_libraries(DeepLearnLib PUBLIC TBB::tbb)
endif()

if(TARGET CUDA::nvtx3)
  target_link_libraries(DeepLearnLib PUBLIC CUDA::nvtx3)
elseif(TARGET CUDA::nvToolsExt)
  target_link_libraries(DeepLearnLib PUBLIC CUDA::nvToolsExt)
endif()

if(OpenCV_FOUND)
  target_link_libraries(DeepLearnLib PUBLIC ${OpenCV_LIBS} pugixml)
endif()
