set(Torch_DIR "${CMAKE_SOURCE_DIR}/extern/pytorch/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(pugixml REQUIRED)

add_executable(bench_torch bench_torch.cpp)

target_link_libraries(
  bench_torch
  PRIVATE ${TORCH_LIBRARIES} benchmark::benchmark benchmark::benchmark_main
          Threads::Threads pugixml::pugixml)

set_target_properties(bench_torch PROPERTIES CXX_STANDARD 17
                                             CXX_STANDARD_REQUIRED ON)

target_include_directories(bench_torch SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(bench_torch PRIVATE "${CMAKE_SOURCE_DIR}/extern/stb")
target_compile_definitions(bench_torch PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
