cmake_minimum_required(VERSION 3.10)
project(DeepLearnLib VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include the library and tests
add_subdirectory(src)
add_subdirectory(tests)
