# KV Cache Management Library
add_library(qwen3-kv-cache STATIC
    simple-kv-cache.cpp
    ssm-state-cache.cpp
    turboquant.cpp
    compressed_kv_store.cpp
    snapkv-eviction.cpp
)

target_include_directories(qwen3-kv-cache PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>
)



# Enable position-independent code
set_target_properties(qwen3-kv-cache PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Memory optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(qwen3-kv-cache PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -funroll-loops>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>
    )
endif()