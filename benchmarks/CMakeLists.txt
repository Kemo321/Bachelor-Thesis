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
            "${CMAKE_SOURCE_DIR}/benchmarks/models"
            "${CMAKE_SOURCE_DIR}/extern/stb")
  if(NOT MSVC)
    target_compile_definitions(${target} PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
  endif()
  file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" _deeplearn_source_dir)
  target_compile_definitions(${target} PRIVATE DEEPLEARN_SOURCE_DIR="${_deeplearn_source_dir}")
  target_link_libraries(${target} PRIVATE spdlog::spdlog)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
endfunction()

# Application models (YOLO, SimpleCNN) live in models/ and link via DeepLearnModels.
# They are not compiled into DeepLearnLib.
set(CUSTOM_BENCH_TARGETS
    train_voc_custom
    train_bccd_custom
    train_synthetic_custom
    train_cifar_custom
    train_tabular_custom
    overfit_voc_custom
    short_voc_custom
    inference_voc_custom
    inference_bccd_custom
    inference_synthetic_custom
    bench_voc_custom)

add_executable(train_voc_custom train_voc_custom.cpp)
add_executable(train_bccd_custom train_bccd_custom.cpp)
add_executable(train_synthetic_custom train_synthetic_custom.cpp)
add_executable(train_cifar_custom train_cifar_custom.cpp)
add_executable(train_tabular_custom train_tabular_custom.cpp)
add_executable(overfit_voc_custom overfit_voc_custom.cpp)
add_executable(short_voc_custom short_voc_custom.cpp)
add_executable(inference_voc_custom inference_voc_custom.cpp)
add_executable(inference_bccd_custom inference_bccd_custom.cpp)
add_executable(inference_synthetic_custom inference_synthetic_custom.cpp)
add_executable(bench_voc_custom bench_voc_custom.cpp)

foreach(target ${CUSTOM_BENCH_TARGETS})
  target_link_libraries(
    ${target}
    PRIVATE DeepLearnLib DeepLearnModels Threads::Threads pugixml ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
  dllib_configure_bench_target(${target})
endforeach()

target_link_libraries(bench_voc_custom PRIVATE benchmark::benchmark benchmark::benchmark_main)

# Matching Torch binaries. Micro-ops stay Torch-gated because they compare against LibTorch.
set(TORCH_BENCH_TARGETS
    train_voc_torch
    train_bccd_torch
    train_synthetic_torch
    train_cifar_torch
    train_tabular_torch
    overfit_voc_torch
    short_voc_torch
    inference_voc_torch
    inference_bccd_torch
    inference_synthetic_torch
    bench_voc_torch
    bench_micro_ops)

if(TARGET TorchBaseline)
  add_executable(train_voc_torch train_voc_torch.cpp)
  add_executable(train_bccd_torch train_bccd_torch.cpp)
  add_executable(train_synthetic_torch train_synthetic_torch.cpp)
  add_executable(train_cifar_torch train_cifar_torch.cpp)
  add_executable(train_tabular_torch train_tabular_torch.cpp)
  add_executable(overfit_voc_torch overfit_voc_torch.cpp)
  add_executable(short_voc_torch short_voc_torch.cpp)
  add_executable(inference_voc_torch inference_voc_torch.cpp)
  add_executable(inference_bccd_torch inference_bccd_torch.cpp)
  add_executable(inference_synthetic_torch inference_synthetic_torch.cpp)
  add_executable(bench_voc_torch bench_voc_torch.cpp)
  add_executable(bench_micro_ops bench_micro_ops.cpp bench_micro_more.cpp)

  foreach(target ${TORCH_BENCH_TARGETS})
    target_link_libraries(
      ${target}
      PRIVATE DeepLearnLib
              DeepLearnModels
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

  target_link_libraries(bench_voc_torch PRIVATE benchmark::benchmark benchmark::benchmark_main)
  target_link_libraries(bench_micro_ops PRIVATE benchmark::benchmark benchmark::benchmark_main)
else()
  message(STATUS "LibTorch baseline not available; skipping torch benchmark targets")
endif()
