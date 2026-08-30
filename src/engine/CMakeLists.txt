# The engine: the loaded model (weights, backend, scheduler) and the
# orchestration of one step over it. Renamed from `core` — a concept-free name
# that had accreted the load path, decode orchestration and four persistence
# services. The persistence services moved to src/session/; the GGUF value bag
# and the mmap wrapper moved to src/loader/.
add_library(qinf-engine STATIC
    model.cpp
    decode_graph_cache.cpp
    # These two live in src/session/ (they are session-persistence services) but
    # build here: qinf-session is deliberately dependency-free so it stays
    # unit-testable in isolation, and persistent_image_embedding_store needs
    # graph_inputs/ — which already links qinf-session, so folding it in would
    # be a cycle. Directory = concept; target = layering.
    ${CMAKE_SOURCE_DIR}/src/session/persistent_image_embedding_store.cpp
    ${CMAKE_SOURCE_DIR}/src/session/prefix_library.cpp
)

target_include_directories(qinf-engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(qinf-engine PUBLIC
    ggml
    qinf-layers
    qinf-state
    qinf-loader
    qinf-graph-inputs
)

# Enable position-independent code for static library
set_target_properties(qinf-engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compiler-specific optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(qinf-engine PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -ffast-math>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>
    )
endif()

# ── Orchestration: driving a built forward pass ─────────────────────────────
# decode_plan (which kind of step), decode_step (run one), multimodal_prefill
# (run an image prefill). All three take a ForwardPassBase, so they sit ABOVE
# qinf-models and cannot join qinf-engine — qinf-models links qinf-engine, and
# folding them in would close a cycle. Same rule the two session sources above
# follow: directory = concept, target = layering.
#
# Before 2026-08-30 these were compiled directly into cli/, server/ and four
# test targets.
add_library(qinf-orchestration STATIC
    decode_plan.cpp
    decode_step.cpp
    multimodal_prefill.cpp
)

target_include_directories(qinf-orchestration PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(qinf-orchestration PUBLIC
    qinf-models
    qinf-vision
    qinf-session
    sampling
    ggml
)

set_target_properties(qinf-orchestration PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
