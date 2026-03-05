add_executable(minic_tests test_main.cpp)

# Link your library and GTest
target_link_libraries(minic_tests 
    PRIVATE 
    DeepLearnLib 
    gtest_main
)

# Automatically add tests to "ctest"
include(GoogleTest)
gtest_discover_tests(minic_tests)