find_package(CUDAToolkit QUIET)

add_executable(dllib_tests test_main.cpp test_tensor.cpp)

target_link_libraries(dllib_tests PRIVATE DeepLearnLib GTest::gtest_main)

if(CUDAToolkit_FOUND)
  target_link_libraries(dllib_tests PRIVATE CUDA::cudart)
endif()

include(GoogleTest)
gtest_discover_tests(dllib_tests)
