# Per-sequence state: kv_cache_simple, recurrent/deltanet.

add_library(qinf-state STATIC
    kv_cache_simple.cpp
    ssm_state_cache.cpp
    recurrent_state.cpp
    deltanet_state.cpp
    token_sequence_section.cpp
)

target_include_directories(qinf-state PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>
)

# qinf-session: snapshot contract + IO. TokenSequenceSection serializes the
# per-sequence Control-lane token vector (L1 session snapshots).
target_link_libraries(qinf-state PUBLIC ggml qinf-session)

set_target_properties(qinf-state PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
