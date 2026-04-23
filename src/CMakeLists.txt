set(Torch_DIR "${CMAKE_SOURCE_DIR}/extern/pytorch/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB_RECURSE SOURCES "*.cpp" "*.cu")

add_library(DeepLearnLib ${SOURCES})

set_target_properties(
  DeepLearnLib
  PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
             CUDA_STANDARD 17
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON)

target_include_directories(
  DeepLearnLib
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
         $<INSTALL_INTERFACE:include> ${CUTLASS_DIR}/include
         ${CUTLASS_DIR}/tools/util/include)

target_link_libraries(DeepLearnLib PUBLIC hwy CUDA::cudart "${TORCH_LIBRARIES}")

if(NOT MSVC)
  target_compile_options(DeepLearnLib PRIVATE -mavx2 -mfma -O3)
endif()
