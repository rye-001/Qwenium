# Model recipes: Qwen2/3 and Qwen3.5.
# Each file composes layer modules (src/layers/) into a complete forward pass.
# Phase 1 (PR 1.6): mechanical move of Qwen3ForwardPass and Qwen35ForwardPass
# from src/core/ to here. Logic is unchanged.

add_library(qinf-models STATIC
    forward_pass_base.cpp
    qwen3.cpp
    qwen35.cpp
    qwen36.cpp
    gemma1.cpp
    gemma2.cpp
    model_registry.cpp
)

target_include_directories(qinf-models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(qinf-models PUBLIC
    ggml
    core
    qinf-layers
    qinf-state
)

set_target_properties(qinf-models PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
