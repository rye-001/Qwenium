# Qwen-3 Core Engine Library
add_library(core STATIC
    model.cpp
)

target_include_directories(core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(core PUBLIC
    ggml
    qinf-layers
    qinf-state
    qinf-loader
)

# Enable position-independent code for static library
set_target_properties(core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compiler-specific optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(core PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -ffast-math>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>
    )
endif()