find_package(pugixml REQUIRED)
find_package(Threads REQUIRED)

set(BENCH_TARGETS 
    train_pipeline_torch 
    train_pipeline_custom
    short_pipeline_torch
    short_pipeline_custom
    bench_torch 
    bench_custom
)

add_executable(train_pipeline_torch train_pipeline_torch.cpp)
add_executable(train_pipeline_custom train_pipeline_custom.cpp)
add_executable(bench_torch bench_torch.cpp)
add_executable(bench_custom bench_custom.cpp)
add_executable(short_pipeline_torch short_pipeline_torch.cpp)
add_executable(short_pipeline_custom short_pipeline_custom.cpp)

foreach(target ${BENCH_TARGETS})
    target_link_libraries(${target} PRIVATE 
        DeepLearnLib 
        ${TORCH_LIBRARIES}
        Threads::Threads 
        pugixml
    )

    target_include_directories(${target} PRIVATE 
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/extern/stb"
    )
    
    target_include_directories(${target} SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    
    target_compile_definitions(${target} PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    set_target_properties(${target} PROPERTIES 
        CXX_STANDARD 17 
        CXX_STANDARD_REQUIRED ON
    )
endforeach()

target_link_libraries(bench_torch PRIVATE 
    benchmark::benchmark 
    benchmark::benchmark_main
)

target_link_libraries(bench_custom PRIVATE 
    benchmark::benchmark 
    benchmark::benchmark_main
)

if(TORCH_CUDA_AVAILABLE)
    target_link_libraries(train_pipeline_custom PRIVATE torch_cuda)
    target_link_libraries(bench_custom PRIVATE torch_cuda)
endif()