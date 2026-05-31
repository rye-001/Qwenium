# tests/grammar/CMakeLists.txt
# Grammar regression tests (non-GTest, standalone executables)
# These test the GrammarVocab API from src/sampling/grammar_vocab.h

add_executable(grammar-test-dynamic-strings test_dynamic_strings.cpp)
target_link_libraries(grammar-test-dynamic-strings PRIVATE sampling)

add_executable(grammar-test-js-subset test_js_subset.cpp)
target_link_libraries(grammar-test-js-subset PRIVATE sampling)

add_executable(grammar-test-multitoken test_multitoken.cpp)
target_link_libraries(grammar-test-multitoken PRIVATE sampling)

add_executable(grammar-test-sampling-integration test_sampling_integration.cpp)
target_link_libraries(grammar-test-sampling-integration PRIVATE sampling)

add_executable(grammar-test-resolve-once test_resolve_once.cpp)
target_link_libraries(grammar-test-resolve-once PRIVATE sampling)

# Phase 1 correctness guard: corpus parse-check (standalone, no model).
add_executable(grammar-test-corpus-parse test_corpus_parse.cpp)
target_link_libraries(grammar-test-corpus-parse PRIVATE sampling)

# Diagnostic: byte-identical decode across sparse/dense paths.
# Heavy — loads a real model. Not a unit test; gated on QWENIUM_MODEL_PATH at runtime.
add_executable(grammar-test-sparse-differential
    test_sparse_differential.cpp
    ${CMAKE_SOURCE_DIR}/src/core/decode_step.cpp
    ${CMAKE_SOURCE_DIR}/src/core/decode_plan.cpp
)
target_link_libraries(grammar-test-sparse-differential PRIVATE
    qinf-models
    core
    sampling
    ggml
    Threads::Threads
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(grammar-test-sparse-differential PRIVATE OpenMP::OpenMP_CXX)
endif()
if(APPLE)
    target_link_libraries(grammar-test-sparse-differential PRIVATE "-framework Accelerate")
endif()

# Phase B forced-token elision differential (token-stable vs non-elided).
# Heavy — loads a real model; self-skips when QWENIUM_MODEL_PATH is absent.
add_executable(grammar-test-forced-elision
    test_forced_elision.cpp
    ${CMAKE_SOURCE_DIR}/src/core/decode_step.cpp
    ${CMAKE_SOURCE_DIR}/src/core/decode_plan.cpp
)
target_link_libraries(grammar-test-forced-elision PRIVATE
    qinf-models
    core
    sampling
    ggml
    Threads::Threads
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(grammar-test-forced-elision PRIVATE OpenMP::OpenMP_CXX)
endif()
if(APPLE)
    target_link_libraries(grammar-test-forced-elision PRIVATE "-framework Accelerate")
endif()
