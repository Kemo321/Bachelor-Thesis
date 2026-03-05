add_executable(minic_tests test_main.cpp)

# Link your library and GoogleTest
target_link_libraries(minic_tests 
    PRIVATE 
    DeepLearnLib 
    gtest_main
)

# Register tests in the CTest system
include(GoogleTest)
gtest_discover_tests(minic_tests)