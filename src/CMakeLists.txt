# Create a library from C++ and CUDA files
add_library(DeepLearnLib
    tensor.cpp
    cuda_kernels.cu
)

# Define header paths
target_include_directories(DeepLearnLib PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

# If you use special CUDA libraries (e.g., cuBLAS)
target_link_libraries(DeepLearnLib PRIVATE cudart)