set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

function(dllib_configure_bench_target target)
  target_include_directories(
    ${target}
    PRIVATE "${CMAKE_SOURCE_DIR}/src"
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/benchmarks")
  if(NOT MSVC)
    target_compile_definitions(${target} PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
  endif()
  file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" _deeplearn_source_dir)
  target_compile_definitions(${target} PRIVATE DEEPLEARN_SOURCE_DIR="${_deeplearn_source_dir}")
  target_link_libraries(${target} PRIVATE spdlog::spdlog)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
endfunction()

# Application models live in models/ and link via DeepLearnModels.
# They are not compiled into DeepLearnLib.
set(CUSTOM_BENCH_TARGETS
    train_cifar_custom
    train_mnist_custom
    train_tabular_custom
    train_gru_custom
    bench_forward_custom)

add_executable(train_cifar_custom train/train_cifar_custom.cpp)
add_executable(train_mnist_custom train/train_mnist_custom.cpp)
add_executable(train_tabular_custom train/train_tabular_custom.cpp)
add_executable(train_gru_custom train/train_gru_custom.cpp)
add_executable(bench_forward_custom forward/bench_forward_custom.cpp)

foreach(target ${CUSTOM_BENCH_TARGETS})
  target_link_libraries(
    ${target}
    PRIVATE DeepLearnLib DeepLearnModels Threads::Threads ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
  dllib_configure_bench_target(${target})
endforeach()

# Matching Torch binaries. Micro-ops stay Torch-gated because they compare against LibTorch.
set(TORCH_BENCH_TARGETS
    train_cifar_torch
    train_mnist_torch
    train_tabular_torch
    train_gru_torch
    bench_forward_torch
    bench_micro_ops)

if(Torch_FOUND)
  add_executable(train_cifar_torch train/train_cifar_torch.cpp)
  add_executable(train_mnist_torch train/train_mnist_torch.cpp)
  add_executable(train_tabular_torch train/train_tabular_torch.cpp)
  add_executable(train_gru_torch train/train_gru_torch.cpp)
  add_executable(bench_forward_torch forward/bench_forward_torch.cpp)
  add_executable(bench_micro_ops micro/bench_micro_ops.cpp micro/bench_micro_more.cpp)

  foreach(target ${TORCH_BENCH_TARGETS})
    target_link_libraries(
      ${target}
      PRIVATE DeepLearnLib
              DeepLearnModels
              ${TORCH_LIBRARIES}
              Threads::Threads
              ${OpenCV_LIBS}
              nlohmann_json::nlohmann_json)
    target_include_directories(${target} SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    dllib_configure_bench_target(${target})
    if(TORCH_CUDA_AVAILABLE)
      target_link_libraries(${target} PRIVATE torch_cuda)
    endif()
  endforeach()

  target_link_libraries(bench_micro_ops PRIVATE benchmark::benchmark benchmark::benchmark_main)
else()
  message(STATUS "LibTorch baseline not available; skipping torch benchmark targets")
endif()
