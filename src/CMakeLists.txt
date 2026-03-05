file(GLOB_RECURSE SOURCES "*.cpp" "*.cu")

# Create the library
add_library(DeepLearnLib ${SOURCES})

# Set the paths to the header files
target_include_directories(DeepLearnLib PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
    # CUTLASS headers
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)

# Linking libraries
# hwy is the name of the Highway library target
target_link_libraries(DeepLearnLib PUBLIC hwy)

# Flags for SIMD (AVX2/AVX512)
if(NOT MSVC)
    target_compile_options(DeepLearnLib PRIVATE -mavx2 -mfma -O3)
endif()