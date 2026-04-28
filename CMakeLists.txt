cmake_minimum_required(VERSION 3.20)
project(qwen3-engine VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(QWEN3_BUILD_TESTS "Build tests" OFF)
option(QWEN3_BUILD_EXAMPLES "Build examples" OFF)
option(QWEN3_ENABLE_OPENMP "Enable OpenMP for CPU parallelism" ON)
option(QWEN3_ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(QWEN3_ENABLE_AVX512 "Enable AVX-512 optimizations" ON)
option(QWEN3_STATIC_LINK "Build static binaries" OFF)
option(GGML_USE_METAL "Enable Metal for GPU acceleration on Apple devices" OFF)
option(QINF_MOE_FALLBACK "Use ggml_mul_mat_id fallback for MoE expert dispatch (ON=fallback path, OFF=fused Metal kernel — Phase 4 only)" ON)

# Set standard output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Compiler-specific optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-O3 -march=native -mtune=native)
        add_compile_options(-fno-finite-math-only -funroll-loops)
    elseif(MSVC)
        add_compile_options(/O2 /arch:AVX2)
    endif()
endif()

# Find dependencies
find_package(Threads REQUIRED)

# OpenMP support
if(QWEN3_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(core PRIVATE OpenMP::OpenMP_CXX)
        add_compile_definitions(QWEN3_USE_OPENMP)
    endif()
endif()

# GGML dependency (from llama.cpp)
include(FetchContent)
set(FETCHCONTENT_QUIET ON)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
  ggml
  GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
  GIT_TAG        b8390
  SOURCE_SUBDIR  ggml
)

# Set GGML options to minimal configuration
set(GGML_METAL ${GGML_USE_METAL} CACHE BOOL "Enable Metal backend")
set(GGML_CUDA OFF CACHE BOOL "Disable CUDA for initial build")
set(GGML_VULKAN OFF CACHE BOOL "Disable Vulkan")
set(GGML_OPENCL OFF CACHE BOOL "Disable OpenCL")
set(GGML_SYCL OFF CACHE BOOL "Disable SYCL")
set(GGML_BLAS ON CACHE BOOL "Enable BLAS via Accelerate framework")
set(GGML_LLAMAFILE OFF CACHE BOOL "Disable llamafile")
set(GGML_HBM OFF CACHE BOOL "Disable HBM")
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

FetchContent_MakeAvailable(ggml)

# Apply compiler flags to ggml
target_compile_options(ggml PRIVATE -fno-finite-math-only -funroll-loops)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${ggml_SOURCE_DIR}/include)

# Core library components
add_subdirectory(src/core)
add_subdirectory(src/sampling)

# Main CLI executable
add_subdirectory(src/cli)

# HTTP Server
add_subdirectory(src/server)

# New modular structure (stubs — populated by Phase 1 extraction PRs 1.2–1.6)
add_subdirectory(src/layers)
add_subdirectory(src/state)
add_subdirectory(src/models)
add_subdirectory(src/loader)
add_subdirectory(src/quant)

# Optional components
if(QWEN3_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(QWEN3_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
install(TARGETS qwenium-cli
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY src/core/ 
    DESTINATION include/core
    FILES_MATCHING PATTERN "*.h"
)