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
    YOLO.cpp
    YOLOLoss.cpp
    Profiler.cpp
    mAP.cpp
    Losses.cpp
    Softmax.cpp
    CSVLoader.cpp
    SimpleCNN.cpp
    Logger.cpp
)

if(OpenCV_FOUND)
  list(APPEND DEEPLEARN_SOURCES dataset.cpp utils.cpp ClassificationLoader.cpp)
endif()

add_library(DeepLearnLib ${DEEPLEARN_SOURCES})

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

if(MSVC)
  target_compile_options(
    DeepLearnLib
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj;/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler;-Xcompiler=/Zc:preprocessor,-bigobj>)
endif()

target_link_libraries(
  DeepLearnLib
  PUBLIC CUDA::cudart
         CUDA::cublas
         CUDA::cudnn
         hwy
         spdlog::spdlog)

if(TARGET CUDA::nvtx3)
  target_link_libraries(DeepLearnLib PUBLIC CUDA::nvtx3)
elseif(TARGET CUDA::nvToolsExt)
  target_link_libraries(DeepLearnLib PUBLIC CUDA::nvToolsExt)
endif()

if(OpenCV_FOUND)
  target_link_libraries(DeepLearnLib PUBLIC ${OpenCV_LIBS} pugixml)
endif()
