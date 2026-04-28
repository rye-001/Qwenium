# Sampling Library
add_library(sampling STATIC
    sampling.cpp
    grammar_vocab.cpp
    token-trie.cpp
)

target_include_directories(sampling PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(sampling PUBLIC
)

# Enable position-independent code for static library
set_target_properties(sampling PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compiler-specific optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(sampling PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -ffast-math>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>
    )
endif()