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
        target_link_libraries(qinf-engine PRIVATE OpenMP::OpenMP_CXX)
        add_compile_definitions(QWEN3_USE_OPENMP)
    endif()
endif()

# GGML dependency (from llama.cpp)
include(FetchContent)
set(FETCHCONTENT_QUIET ON)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

#
# Vendored ggml is patched with our downstream-only fused ops via the
# patches/ directory. patches/apply-all.sh iterates patches/*.patch in
# lexical order, dry-run-checking each before applying so reconfigures are
# idempotent (FETCHCONTENT_UPDATES_DISCONNECTED already suppresses re-fetch;
# the dry-run check guards against double-apply in mixed-state trees).
#
# See: docs/plan-deltanet-fusion.md for context on why we patch downstream
# rather than fork or upstream.
#
file(GLOB QINF_GGML_PATCH_FILES "${CMAKE_SOURCE_DIR}/patches/*.patch")
if(QINF_GGML_PATCH_FILES)
    message(STATUS "qinf: ggml patches detected: ${QINF_GGML_PATCH_FILES}")
endif()

FetchContent_Declare(
  ggml
  GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
  GIT_TAG        b10582
  SOURCE_SUBDIR  ggml
  PATCH_COMMAND  bash "${CMAKE_SOURCE_DIR}/patches/apply-all.sh" "${CMAKE_SOURCE_DIR}/patches"
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
add_subdirectory(src/engine)
# Session snapshot seams: dependency-free; declared before its consumers
# (sampling, state) so their target_link_libraries reference resolves cleanly.
add_subdirectory(src/session)
add_subdirectory(src/sampling)

# Main CLI executable
add_subdirectory(src/cli)

# HTTP Server
add_subdirectory(src/server)

# Layer-type modules, state, recipes and the subsystems around them.
add_subdirectory(src/layers)
add_subdirectory(src/state)
add_subdirectory(src/graph_inputs)
add_subdirectory(src/models)
add_subdirectory(src/loader)
add_subdirectory(src/vision)

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

install(DIRECTORY src/engine/ 
    DESTINATION include/engine
    FILES_MATCHING PATTERN "*.h"
)