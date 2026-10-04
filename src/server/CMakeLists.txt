# CMakeLists.txt for HTTP inference server
# Add this to your existing CMake setup or integrate as needed

cmake_minimum_required(VERSION 3.16)

# =============================================================================
# Dependencies
# =============================================================================

# cpp-httplib (header-only)
# Option 1: FetchContent
include(FetchContent)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.15.3
)
FetchContent_MakeAvailable(httplib)

# nlohmann/json (header-only)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# =============================================================================
# Server Library (header-only, but we create an interface target)
# =============================================================================
add_library(inference_server INTERFACE)
target_include_directories(inference_server INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(inference_server INTERFACE
    Threads::Threads
)

# =============================================================================
# Server units (compiled)
# =============================================================================
# The reusable half of the server: the data: URI decoder for OpenAI-style
# image_url payloads. It was compiled directly into qwenium-server AND into test
# targets until 2026-08-30 — the same one-source-many-copies arrangement
# slot_snapshot had. The lens (server_lens) and the image verdict
# (image_verdict) lived here too until 2026-10-03; they moved to the
# qemmi-lens-server repo, which links this library for image_data_uri.
#
# http_server.cpp and server_vision.cpp stay in the executable: the first owns
# main(), and the second has no consumer other than the executable, so it needs
# no library.
#
add_library(qinf-server STATIC
    image_data_uri.cpp
)

target_include_directories(qinf-server PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(qinf-server PUBLIC
    nlohmann_json::nlohmann_json
)

set_target_properties(qinf-server PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# =============================================================================
# HTTP Server Executable
# =============================================================================
add_executable(qwenium-server
    http_server.cpp
    server_vision.cpp
)

target_link_libraries(qwenium-server PRIVATE
    qinf-server
    qinf-snapshot
    qinf-orchestration
    qinf-image
    inference_server
    httplib::httplib
    nlohmann_json::nlohmann_json
    qinf-models
    qinf-vision
    qinf-engine
    qinf-state
    sampling
    ggml
    Threads::Threads
)

target_compile_features(qwenium-server PRIVATE cxx_std_17)

# =============================================================================
# Install
# =============================================================================
install(TARGETS qwenium-server
    RUNTIME DESTINATION bin
)

install(FILES inference_server.h
    DESTINATION include/qwen3-server
)