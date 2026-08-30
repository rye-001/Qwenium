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
# The reusable halves of the server: the Lens extraction shape contract and the
# data: URI decoder for OpenAI-style image_url payloads. Both were compiled
# directly into http_server AND into three test targets until 2026-08-30 — the
# same one-source-many-copies arrangement slot_snapshot had.
#
# http_server.cpp and server_vision.cpp stay in the executable: the first owns
# main(), and the second has no consumer other than the executable, so it needs
# no library.
#
# Trade worth naming: image_data_uri.cpp is pure std, and test_image_data_uri
# previously linked nothing at all. Linking qinf-server pulls the model stack
# into that small test. Accepted for one home per source; the alternative was a
# second single-file library, which is a worse trade at this size.
add_library(qinf-server STATIC
    server_lens.cpp
    image_data_uri.cpp
)

target_include_directories(qinf-server PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(qinf-server PUBLIC
    qinf-models
    sampling
    nlohmann_json::nlohmann_json
    ggml
)

set_target_properties(qinf-server PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# =============================================================================
# HTTP Server Executable
# =============================================================================
add_executable(http_server
    http_server.cpp
    server_vision.cpp
)

target_link_libraries(http_server PRIVATE
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
    sampling
    ggml
    Threads::Threads
)

target_compile_features(http_server PRIVATE cxx_std_17)

# =============================================================================
# Install
# =============================================================================
install(TARGETS http_server
    RUNTIME DESTINATION bin
)

install(FILES inference_server.h
    DESTINATION include/qwen3-server
)