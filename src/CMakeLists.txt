file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.cu")

if(NOT OpenCV_FOUND)
  list(FILTER SOURCES EXCLUDE REGEX ".*/dataset\\.cpp$")
  list(FILTER SOURCES EXCLUDE REGEX ".*/utils\\.cpp$")
endif()

add_library(DeepLearnLib ${SOURCES})

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/Tensor.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/Conv2d.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/BatchNorm2d.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/LeakyReLU.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/Dropout.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/FullyConnected.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/Network.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/YOLOLoss.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(
  DeepLearnLib
  PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
             CUDA_STANDARD 17
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON)

target_include_directories(DeepLearnLib PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include> 
    "${CMAKE_SOURCE_DIR}/extern/stb"
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(DeepLearnLib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
endif()

if(MSVC)
  target_compile_options(DeepLearnLib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj;/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler;-Xcompiler=/Zc:preprocessor,-bigobj>
  )
endif()

set(_deeplearn_libs
    torch
    hwy
    CUDA::cudart
    CUDA::cublas
    CUDA::cudnn
    ${TORCH_LIBRARIES}
)
if(OpenCV_FOUND)
  list(APPEND _deeplearn_libs ${OpenCV_LIBS})
endif()

target_link_libraries(DeepLearnLib PUBLIC ${_deeplearn_libs})
