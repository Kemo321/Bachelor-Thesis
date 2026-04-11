set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(bench_tensor bench_tensor.cpp)

target_link_libraries(bench_tensor PRIVATE DeepLearnLib benchmark::benchmark)

target_compile_definitions(bench_tensor PRIVATE BENCHMARK_STATIC_DEFINE)

set_property(TARGET bench_tensor PROPERTY CXX_STANDARD 17)
