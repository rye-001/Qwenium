# Portable session snapshot seams (docs/plan-session-snapshot.md, Phase 0).
# Pure byte plumbing + contract + coordinator + compat header. No ggml/model
# dependency so it stays unit-testable in isolation.

add_library(qinf-session STATIC
    snapshot_io.cpp
    compat_header.cpp
    session_manifest.cpp
    # Pure std — no ggml, no model, no vision — so it satisfies this library's
    # model-free contract. It was compiled directly into cli/, server/ and three
    # test targets until 2026-08-30.
    image_embedding_cache.cpp
)

target_include_directories(qinf-session PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(qinf-session PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Session state bound to a LIVE forward pass: capture/restore of one slot's KV
# and recurrent sections, plus the compat header derived from the caches that
# will receive them. Separate target from qinf-session on purpose — that one is
# deliberately model-free (pure byte plumbing, unit-testable without ggml), and
# this one needs the model, the recipes and the state types.
#
# It exists because slot_snapshot.cpp used to be compiled directly into every
# consumer (cli/, server/, and any test needing make_snapshot_header): one
# shared source with three private copies, and a fourth required every time
# something new wanted it.
add_library(qinf-snapshot STATIC
    slot_snapshot.cpp
)

target_include_directories(qinf-snapshot PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(qinf-snapshot PUBLIC
    qinf-session
    qinf-models
    qinf-state
    ggml
)

set_target_properties(qinf-snapshot PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
