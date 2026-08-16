set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT TARGET pugixml)
  find_package(pugixml REQUIRED)
endif()
find_package(Threads REQUIRED)

function(dllib_configure_bench_target target)
  target_include_directories(
    ${target}
    PRIVATE "${CMAKE_SOURCE_DIR}/src"
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/benchmarks"
            "${CMAKE_SOURCE_DIR}/extern/stb")
  if(NOT MSVC)
    target_compile_definitions(${target} PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
  endif()
  file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" _deeplearn_source_dir)
  target_compile_definitions(${target} PRIVATE DEEPLEARN_SOURCE_DIR="${_deeplearn_source_dir}")
  set_target_properties(${target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
endfunction()

set(CUSTOM_BENCH_TARGETS
    train_pipeline_custom
    short_pipeline_custom
    bench_custom
    train_pipeline_custom_BCCD
    train_pipeline_custom_synthetic
    overfit_test
    tabular_demo)

add_executable(train_pipeline_custom train_pipeline_custom.cpp)
add_executable(short_pipeline_custom short_pipeline_custom.cpp)
add_executable(bench_custom bench_custom.cpp)
add_executable(train_pipeline_custom_BCCD train_pipeline_custom_BCCD.cpp)
add_executable(train_pipeline_custom_synthetic train_pipeline_custom_synthetic.cpp)
add_executable(overfit_test overfit_test.cpp)
add_executable(tabular_demo tabular_demo.cpp)

foreach(target ${CUSTOM_BENCH_TARGETS})
  target_link_libraries(
    ${target}
    PRIVATE DeepLearnLib Threads::Threads pugixml ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
  dllib_configure_bench_target(${target})
endforeach()

target_link_libraries(bench_custom PRIVATE benchmark::benchmark benchmark::benchmark_main)

set(TORCH_BENCH_TARGETS
    train_pipeline_torch
    short_pipeline_torch
    bench_torch
    train_pipeline_torch_BCCD
    train_pipeline_torch_synthetic
    inference
    inference_bccd
    inference_synthetic)

if(TARGET TorchBaseline)
  add_executable(train_pipeline_torch train_pipeline_torch.cpp)
  add_executable(short_pipeline_torch short_pipeline_torch.cpp)
  add_executable(bench_torch bench_torch.cpp)
  add_executable(train_pipeline_torch_BCCD train_pipeline_torch_BCCD.cpp)
  add_executable(train_pipeline_torch_synthetic train_pipeline_torch_synthetic.cpp)
  add_executable(inference inference.cpp)
  add_executable(inference_bccd inference_bccd.cpp)
  add_executable(inference_synthetic inference_synthetic.cpp)

  foreach(target ${TORCH_BENCH_TARGETS})
    target_link_libraries(
      ${target}
      PRIVATE DeepLearnLib
              TorchBaseline
              ${TORCH_LIBRARIES}
              Threads::Threads
              pugixml
              ${OpenCV_LIBS}
              nlohmann_json::nlohmann_json)
    target_include_directories(${target} SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    dllib_configure_bench_target(${target})
    if(TORCH_CUDA_AVAILABLE)
      target_link_libraries(${target} PRIVATE torch_cuda)
    endif()
  endforeach()

  target_link_libraries(bench_torch PRIVATE benchmark::benchmark benchmark::benchmark_main)
else()
  message(STATUS "LibTorch baseline not available; skipping torch benchmark targets")
endif()
