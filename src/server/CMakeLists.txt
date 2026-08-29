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
# HTTP Server Executable
# =============================================================================
add_executable(http_server
    http_server.cpp
    server_vision.cpp
    server_lens.cpp
    image_data_uri.cpp
    # Host-side image pipeline + multimodal orchestrator, compiled directly (these
    # live in src/image, src/session and src/engine but are built into consumers, not a lib —
    # same arrangement the CLI and multimodal-prefill-tests use). Enables image
    # input on /v1/chat/completions when the server is started with --mmproj.
    ${CMAKE_SOURCE_DIR}/src/image/image_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/image/image_prompt.cpp
    ${CMAKE_SOURCE_DIR}/src/engine/multimodal_prefill.cpp
    ${CMAKE_SOURCE_DIR}/src/session/image_embedding_cache.cpp
    ${CMAKE_SOURCE_DIR}/src/session/slot_snapshot.cpp
)

target_link_libraries(http_server PRIVATE
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