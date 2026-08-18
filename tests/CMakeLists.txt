find_package(CUDAToolkit QUIET)

add_executable(
  dllib_tests
  test_main.cpp
  test_tensor.cpp
  test_conv2d.cpp
  test_maxpool2d.cpp
  test_batchnorm2d.cpp
  test_fullyconnected.cpp
  test_activations.cpp
  test_flatten.cpp
  test_fused_cbr2d.cpp
  test_loss.cpp
  test_precision.cpp
  test_numerics.cpp
  test_yolo.cpp
  test_network.cpp
  test_generic.cpp
  test_profiler_map.cpp
  test_simplecnn.cpp
  test_packed_image_loader.cpp
  test_darknet.cpp)

if(OpenCV_FOUND)
  target_sources(dllib_tests PRIVATE test_classification_loader.cpp)
  target_include_directories(dllib_tests PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(dllib_tests PRIVATE ${OpenCV_LIBS})
endif()

target_include_directories(dllib_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/benchmarks/models")

target_link_libraries(dllib_tests PRIVATE DeepLearnLib DeepLearnModels GTest::gtest spdlog::spdlog)

if(CUDAToolkit_FOUND)
  target_link_libraries(dllib_tests PRIVATE CUDA::cudart)
endif()

if(Torch_FOUND)
  target_sources(dllib_tests PRIVATE test_torch_equivalence.cpp)
  target_link_libraries(dllib_tests PRIVATE ${TORCH_LIBRARIES})
  target_include_directories(dllib_tests SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
endif()

if(MSVC)
  target_compile_options(dllib_tests PRIVATE /bigobj /Zc:preprocessor)
endif()

option(ENABLE_COVERAGE "Enable gcov/lcov coverage instrumentation" OFF)
if(ENABLE_COVERAGE AND NOT MSVC)
  message(STATUS "Code coverage enabled for DeepLearnLib and dllib_tests")
  target_compile_options(
    DeepLearnLib
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:--coverage>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=--coverage;-Xcompiler=-Wno-invalid-constexpr>)
  target_compile_options(dllib_tests PRIVATE --coverage)
  target_link_options(dllib_tests PRIVATE --coverage)
endif()

# Prefer the cuDNN DLLs we compiled against (pip nvidia-cudnn) over Torch's
# older copies on PATH. Windows loads DLLs from the executable directory first.
if(WIN32 AND CUDNN_ROOT)
  file(GLOB _cudnn_runtime_dlls "${CUDNN_ROOT}/bin/cudnn*.dll")
  if(_cudnn_runtime_dlls)
    add_custom_command(
      TARGET dllib_tests
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_cudnn_runtime_dlls}
              $<TARGET_FILE_DIR:dllib_tests>
      COMMENT "Copy matching cuDNN DLLs next to dllib_tests")
  endif()
endif()

include(GoogleTest)
gtest_discover_tests(dllib_tests DISCOVERY_MODE PRE_TEST)
