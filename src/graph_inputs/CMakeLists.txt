# Typed graph inputs: self-describing contracts over graph slots.
# See docs/plan-typed-graph-inputs.md. Phase 1 — extraction only.

add_library(qinf-graph-inputs STATIC
    graph_input.cpp
    tokens_input.cpp
    positions_input.cpp
    attn_mask_input.cpp
    gather_indices_input.cpp
    sparse_head_input.cpp
    output_ids_input.cpp
)

target_include_directories(qinf-graph-inputs PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(qinf-graph-inputs PUBLIC ggml)

set_target_properties(qinf-graph-inputs PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
