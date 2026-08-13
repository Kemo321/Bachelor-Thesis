find_package(CUDAToolkit QUIET)

add_executable(dllib_tests test_main.cpp test_tensor.cpp test_layers.cpp test_loss.cpp)

target_link_libraries(dllib_tests PRIVATE DeepLearnLib GTest::gtest)

if(CUDAToolkit_FOUND)
  target_link_libraries(dllib_tests PRIVATE CUDA::cudart)
endif()

if(MSVC)
  target_compile_options(dllib_tests PRIVATE /bigobj /Zc:preprocessor)
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
