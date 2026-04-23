file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp" "*.cu")

add_library(DeepLearnLib ${SOURCES})

set_target_properties(DeepLearnLib PROPERTIES 
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
)

target_include_directories(DeepLearnLib PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include> 
    "${CMAKE_SOURCE_DIR}/extern/stb"
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)

target_compile_options(DeepLearnLib PRIVATE -mavx2 -mfma)

target_link_libraries(DeepLearnLib PUBLIC 
    torch
    hwy 
    CUDA::cudart
)