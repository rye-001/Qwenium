# Portable session snapshot seams (docs/plan-session-snapshot.md, Phase 0).
# Pure byte plumbing + contract + coordinator + compat header. No ggml/model
# dependency so it stays unit-testable in isolation.

add_library(qinf-session STATIC
    snapshot_io.cpp
    compat_header.cpp
    session_manifest.cpp
)

target_include_directories(qinf-session PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(qinf-session PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
