find_package(CUDAToolkit REQUIRED)

add_executable(dllib_tests test_main.cpp test_tensor.cpp)

target_link_libraries(dllib_tests PRIVATE DeepLearnLib GTest::gtest_main
                                          CUDA::cudart)

include(GoogleTest)
gtest_discover_tests(dllib_tests)
