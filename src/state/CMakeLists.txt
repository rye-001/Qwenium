# Per-sequence state: kv_cache_simple, kv_cache_compressed, turboquant, snapkv.
# deltanet_state added in Phase 3 (PR 3.D.1).

add_library(qinf-state STATIC
    kv_cache_simple.cpp
    kv_cache_compressed.cpp
    turboquant.cpp
    snapkv.cpp
    ssm_state_cache.cpp
    recurrent_state.cpp
    deltanet_state.cpp
)

target_include_directories(qinf-state PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(qinf-state PUBLIC ggml)

set_target_properties(qinf-state PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
