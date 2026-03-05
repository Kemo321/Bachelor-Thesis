cmake_minimum_required(VERSION 3.18)

# --- 1. C++ STANDARD SETTINGS BEFORE PROJECT() ---
# This helps CMake properly initialize CXX compiler features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(MiniC_DL LANGUAGES CXX CUDA)

# --- 2. CUDA SETTINGS ---
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add architecture 100 for RTX 5080 (Blackwell)
# 89 (Lovelace/40-series), 90 (Hopper), 100 (Blackwell)
set(CMAKE_CUDA_ARCHITECTURES 89 90 100) 

# --- 3. EXTERNAL LIBRARIES CONFIGURATION ---

# GoogleTest
add_subdirectory(extern/googletest)

# Highway (SIMD)
# Disable tests and problematic vqsort module (contrib)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "Disable Highway tests")
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "Disable Highway contrib (fixes vqsort C++17 error)")
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Highway examples")

add_subdirectory(extern/highway)

# CUTLASS (Header-only)
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/cutlass")

add_subdirectory(src)
add_subdirectory(tests)