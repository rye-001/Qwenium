# Qwen-3 Core Engine Library
add_library(qwen3-core STATIC
    qwen3-model.cpp
    gguf-loader.cpp
    forward-pass-base.cpp
    forward-pass.cpp
    forward-pass-qwen35.cpp
    forward-pass-factory.cpp
    tokenizer.cpp
)

target_include_directories(qwen3-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(qwen3-core PUBLIC
    ggml
    qwen3-kv-cache
)

# Enable position-independent code for static library
set_target_properties(qwen3-core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compiler-specific optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(qwen3-core PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -ffast-math>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>
    )
endif()