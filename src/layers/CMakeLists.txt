# Layer-type modules: attention, ffn, deltanet, ssm, moe.
# ffn added in Phase 1 (PR 1.5); deltanet added in Phase 3 (PR 3.D.2-4).
# moe added in Phase 3 (PR 3.M.1-3); ssm added in Phase 3 if needed.

add_library(qinf-layers STATIC
    attention.cpp
    ffn.cpp
    deltanet.cpp
    moe.cpp
    moe_residency.cpp
    norm.cpp
    ple.cpp           # Per-Layer Embedding lookup + projection
    transformer_block.cpp
)

target_include_directories(qinf-layers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(qinf-layers PUBLIC
    ggml
    qinf-state
)

if(QINF_MOE_FALLBACK)
    target_compile_definitions(qinf-layers PUBLIC QINF_MOE_FALLBACK)
endif()

set_target_properties(qinf-layers PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
