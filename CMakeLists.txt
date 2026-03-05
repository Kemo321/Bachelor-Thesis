cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(MiniC_DL LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89 90 100) 

add_subdirectory(extern/googletest)

set(HWY_ENABLE_TESTS OFF CACHE BOOL "Disable Highway tests")
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "Disable Highway contrib")
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Highway examples")
add_subdirectory(extern/highway)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
add_subdirectory(extern/benchmark)

set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/cutlass")

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)

option(USE_SANITIZERS "Enable ASan and UBSan" ON)
if(USE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(DeepLearnLib PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(DeepLearnLib PUBLIC -fsanitize=address,undefined)
    message(STATUS "Sanitizers enabled for DeepLearnLib")
endif()

find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
if(CLANG_TIDY_EXE)
    set(CLANG_TIDY_COMMAND 
        ${CLANG_TIDY_EXE}
        "-checks=-*,bugprone-*,performance-*,modernize-*,readability-*,cppcoreguidelines-*"
        "-warnings-as-errors=*"
    )
    set_target_properties(DeepLearnLib PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
endif()

find_program(CPPCHECK_EXE NAMES "cppcheck")
if(CPPCHECK_EXE)
    add_custom_target(cppcheck
        COMMAND ${CPPCHECK_EXE} --enable=all --suppress=missingIncludeSystem 
                --project=${CMAKE_BINARY_DIR}/compile_commands.json
        COMMENT "Running cppcheck analysis..."
    )
endif()

find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating documentation with Doxygen"
        VERBATIM)
endif()
