# benchmarks/CMakeLists.txt

find_package(pugixml REQUIRED)
find_package(Threads REQUIRED)

# -------------------------------------------------------------------------
# 1. Cel: train_pipeline (Pętla treningowa do dokumentacji)
# -------------------------------------------------------------------------
add_executable(train_pipeline train_pipeline.cpp)

target_link_libraries(train_pipeline PRIVATE 
    DeepLearnLib 
    torch 
    torch_cuda 
    c10 
    Threads::Threads 
    pugixml
)

target_include_directories(train_pipeline PRIVATE 
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/extern/stb"
)
target_include_directories(train_pipeline SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(train_pipeline PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
set_target_properties(train_pipeline PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------------------------
# 2. Cel: bench_torch (Benchmark wydajnościowy)
# -------------------------------------------------------------------------
add_executable(bench_torch bench_torch.cpp)

target_link_libraries(bench_torch PRIVATE 
    DeepLearnLib 
    torch 
    torch_cuda 
    benchmark::benchmark 
    benchmark::benchmark_main
    Threads::Threads 
    pugixml
)

target_include_directories(bench_torch PRIVATE 
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/extern/stb"
)
target_include_directories(bench_torch SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(bench_torch PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
set_target_properties(bench_torch PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)