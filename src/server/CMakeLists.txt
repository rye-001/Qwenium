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
)

target_link_libraries(http_server PRIVATE
    inference_server
    httplib::httplib
    nlohmann_json::nlohmann_json
    qinf-models
    core
    sampling
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