# tests/CMakeLists.txt
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# Define the unit test executable
add_executable(unit-tests
    unit/test_tokenizer.cpp
    unit/test_vocab_file_loading.cpp
    unit/test_qwen35_metadata.cpp
    unit/test_qwen35_tensor_loading.cpp
    unit/test_ssm_state_cache.cpp
    unit/test_qwen35_tokenizer.cpp
    unit/test_qwen35_hybrid_cache.cpp
    unit/test_qwen35_forward_attn.cpp
    unit/test_qwen35_generation.cpp
    unit/test_grammar_vocab.cpp
    unit/test_turboquant_generation.cpp
)

# Link libraries for unit tests
target_link_libraries(unit-tests PRIVATE
    GTest::gtest_main
    qinf-models
    core
    sampling
    ggml
)

# Discover and add unit tests to CTest
include(GoogleTest)
gtest_discover_tests(unit-tests)

# Define the integration test executable
add_executable(integration-tests
    integration/test_e2e_generation.cpp
)

# Link libraries for integration tests
target_link_libraries(integration-tests PRIVATE
    GTest::gtest_main
    qinf-models
    core
    sampling
    ggml
)

# Discover and add integration tests to CTest
gtest_discover_tests(integration-tests)

# Define the standalone order management test (Non-GTest)
add_executable(order-mngmt-test
    integration/test_order_management_generation.cpp
)

# Link libraries
target_link_libraries(order-mngmt-test PRIVATE
    qinf-models
    core
    sampling
    ggml
)

# Define the standalone order management test (Non-GTest)
add_executable(parallel-order-mngmt-test
    integration/test_parallel_order_management.cpp
)

# Link libraries
target_link_libraries(parallel-order-mngmt-test PRIVATE
    qinf-models
    core
    sampling
    ggml
)

# TurboQuant unit tests (GTest, no model needed)
add_executable(turboquant-tests
    unit/test_turboquant.cpp
)

target_link_libraries(turboquant-tests PRIVATE
    GTest::gtest_main
    qinf-state
)

gtest_discover_tests(turboquant-tests)

# Compressed KV store unit tests (GTest, no model needed)
add_executable(compressed-kv-tests
    unit/test_compressed_kv_store.cpp
)

target_link_libraries(compressed-kv-tests PRIVATE
    GTest::gtest_main
    qinf-state
)

gtest_discover_tests(compressed-kv-tests)

# SnapKV CLI config tests (GTest, no model needed)
add_executable(snapkv-cli-tests
    unit/test_snapkv_cli.cpp
)

target_link_libraries(snapkv-cli-tests PRIVATE
    GTest::gtest_main
)

gtest_discover_tests(snapkv-cli-tests)

# SnapKV eviction + compaction unit tests (GTest, no model needed)
add_executable(snapkv-eviction-tests
    unit/test_snapkv_eviction.cpp
)

target_link_libraries(snapkv-eviction-tests PRIVATE
    GTest::gtest_main
    qinf-state
    ggml
)

gtest_discover_tests(snapkv-eviction-tests)

# SnapKV position tracking tests (GTest)
add_executable(snapkv-position-tests
    unit/test_snapkv_position.cpp
)

target_link_libraries(snapkv-position-tests PRIVATE
    GTest::gtest_main
    qinf-state
    core
    ggml
)

gtest_discover_tests(snapkv-position-tests)

# TokenTrie unit tests (GTest, no model needed)
add_executable(token-trie-tests
    unit/test_token_trie.cpp
)

target_link_libraries(token-trie-tests PRIVATE
    GTest::gtest_main
    sampling
)

gtest_discover_tests(token-trie-tests)

# Parallel grammar test (Non-GTest)
add_executable(parallel-grammar-test
    integration/test_parallel_grammar.cpp
)

target_link_libraries(parallel-grammar-test PRIVATE
    qinf-models
    core
    sampling
    ggml
)

# HTTP Server Integration Test (GTest)
add_executable(http-server-test
    integration/test_http_server.cpp
)

target_link_libraries(http-server-test PRIVATE
    inference_server
    httplib::httplib
    nlohmann_json::nlohmann_json
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

target_compile_features(http-server-test PRIVATE cxx_std_17)
gtest_discover_tests(http-server-test)

# Grammar regression tests (non-GTest standalone executables)
add_subdirectory(grammar)

# ── Phase 3 unit tests ────────────────────────────────────────────────────────

# DeltaNetState tests (PR 3.D.1) — no model file needed.
add_executable(deltanet-state-tests
    unit/test_deltanet_state.cpp
)

target_link_libraries(deltanet-state-tests PRIVATE
    GTest::gtest_main
    qinf-state
    ggml
)

target_compile_features(deltanet-state-tests PRIVATE cxx_std_17)
gtest_discover_tests(deltanet-state-tests)

# DeltaNetLayer tests (PR 3.D.2-4) — no model file needed.
add_executable(deltanet-tests
    unit/test_deltanet.cpp
)

target_link_libraries(deltanet-tests PRIVATE
    GTest::gtest_main
    qinf-layers
    qinf-state
    ggml
)

target_compile_features(deltanet-tests PRIVATE cxx_std_17)
gtest_discover_tests(deltanet-tests)

# MoELayer tests (PR 3.M.1-3) — no model file needed.
add_executable(moe-tests
    unit/test_moe.cpp
)

target_link_libraries(moe-tests PRIVATE
    GTest::gtest_main
    qinf-layers
    ggml
)

target_compile_features(moe-tests PRIVATE cxx_std_17)
gtest_discover_tests(moe-tests)

# MoE residency tracker tests (PR 3.M.4) — pure C++, no ggml compute.
add_executable(moe-residency-tests
    unit/test_moe_residency.cpp
)

target_link_libraries(moe-residency-tests PRIVATE
    GTest::gtest_main
    qinf-layers
)

target_compile_features(moe-residency-tests PRIVATE cxx_std_17)
gtest_discover_tests(moe-residency-tests)

# Qwen36ForwardPass smoke tests (PR 3.I.2).
# Model-file tests self-skip when QWEN36_MODEL_PATH is unset.
add_executable(forward-tests-qwen36
    unit/test_qwen36_forward.cpp
)

target_link_libraries(forward-tests-qwen36 PRIVATE
    GTest::gtest_main
    qinf-models
    core
    ggml
)

target_compile_features(forward-tests-qwen36 PRIVATE cxx_std_17)
gtest_discover_tests(forward-tests-qwen36)

# Qwen 3.6 hparam + inventory validation tests (PR 3.I.1).
# Pure-unit tests (no model file) run unconditionally.
# Model-file integration tests self-skip when QWEN_MODEL_PATH is unset.
add_executable(hparams-tests-qwen36
    unit/test_qwen36_hparams.cpp
)

target_link_libraries(hparams-tests-qwen36 PRIVATE
    GTest::gtest_main
    qinf-loader
    qinf-models
)

target_compile_features(hparams-tests-qwen36 PRIVATE cxx_std_17)
gtest_discover_tests(hparams-tests-qwen36)

# Qwen35Config unit tests — no model file needed.
# Covers: from_metadata happy path; each invariant violation → fail-loud message;
#         is_full_attention_layer / is_ssm_layer correctness against inline arithmetic.
add_executable(config-tests-qwen35
    unit/test_qwen35_config.cpp
)

target_link_libraries(config-tests-qwen35 PRIVATE
    GTest::gtest_main
    qinf-models
    core
)

target_compile_features(config-tests-qwen35 PRIVATE cxx_std_17)
gtest_discover_tests(config-tests-qwen35)

# Qwen35MoEConfig unit tests — no model file needed.
# Covers: from_metadata happy path; each invariant violation → fail-loud message.
add_executable(config-tests-qwen36
    unit/test_qwen36_config.cpp
)

target_link_libraries(config-tests-qwen36 PRIVATE
    GTest::gtest_main
    qinf-models
    core
)

target_compile_features(config-tests-qwen36 PRIVATE cxx_std_17)
gtest_discover_tests(config-tests-qwen36)

# Norm helper unit tests — no model file needed.
add_executable(norm-tests
    unit/test_norm.cpp
)

target_link_libraries(norm-tests PRIVATE
    GTest::gtest_main
    qinf-layers
    ggml
)

target_compile_features(norm-tests PRIVATE cxx_std_17)
gtest_discover_tests(norm-tests)

# ── Phase 2 unit tests ────────────────────────────────────────────────────────

# Layer and state interface tests (no model file needed).
# Covers: LayerState, KVCache conformance + truncate_to_position,
#         RecurrentState checkpoint/restore, AttentionLayer both phases,
#         FFN Phase parameter, kv_cache build/position/truncate.
add_executable(layer-tests
    unit/test_layer_state.cpp
    unit/test_kv_cache.cpp
    unit/test_recurrent_state.cpp
    unit/test_attention.cpp
    unit/test_ffn.cpp
)

target_link_libraries(layer-tests PRIVATE
    GTest::gtest_main
    qinf-layers
    qinf-state
    ggml
)

target_compile_features(layer-tests PRIVATE cxx_std_17)
gtest_discover_tests(layer-tests)

# Weight-binding tests (no model file, no ggml compute).
add_executable(weight-binding-tests
    unit/test_weight_binding.cpp
)

target_link_libraries(weight-binding-tests PRIVATE
    GTest::gtest_main
    qinf-loader
)

target_compile_features(weight-binding-tests PRIVATE cxx_std_17)
gtest_discover_tests(weight-binding-tests)

# Gemma 1 forward pass integration test (PR G1.5).
# Self-skips when the canonical Gemma file is absent.
add_executable(forward-tests-gemma1
    unit/test_gemma1_forward.cpp
)
target_link_libraries(forward-tests-gemma1 PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-loader
    core
    ggml
)
target_compile_features(forward-tests-gemma1 PRIVATE cxx_std_17)
gtest_discover_tests(forward-tests-gemma1)

# Chat template unit tests (PR G1.7).
# QwenChatTemplate / GemmaChatTemplate live in qinf-loader; the registry
# dispatch tests additionally need qinf-models (lookup_chat_template,
# register_builtin_models). core + ggml are transitive deps of
# qinf-models headers (forward_pass_base.h pulls in ggml.h).
add_executable(chat-template-tests
    unit/test_chat_template.cpp
)
target_link_libraries(chat-template-tests PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-loader
    core
    ggml
)
target_compile_features(chat-template-tests PRIVATE cxx_std_17)
gtest_discover_tests(chat-template-tests)

# Gemma tokenizer unit tests (PR G1.3) — no model file needed.
add_executable(gemma-tokenizer-tests
    unit/test_tokenizer_gemma.cpp
)
target_link_libraries(gemma-tokenizer-tests PRIVATE
    GTest::gtest_main
    qinf-loader
)
target_compile_features(gemma-tokenizer-tests PRIVATE cxx_std_17)
gtest_discover_tests(gemma-tokenizer-tests)

# GGUFKVBag unit tests — no model file needed.
# Covers: round-trip set/get for each scalar type; missing-key / wrong-type
# fail-loud; optional accessor (nullopt / value / wrong-type throw); contains().
add_executable(gguf-kv-bag-tests
    unit/test_gguf_kv_bag.cpp
)

target_link_libraries(gguf-kv-bag-tests PRIVATE
    GTest::gtest_main
    core
)

target_compile_features(gguf-kv-bag-tests PRIVATE cxx_std_17)
gtest_discover_tests(gguf-kv-bag-tests)

# Loader unit tests — no model file needed for construction / tokenizer / ASCII.
# GGUFKVBagLoaderTest suite additionally exercises raw_kv population from a
# real GGUF; those tests self-skip when QWEN35_MODEL_PATH is unset.
add_executable(loader-tests
    unit/test_loader.cpp
)

target_link_libraries(loader-tests PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-loader
    core
    ggml
)

target_compile_features(loader-tests PRIVATE cxx_std_17)
gtest_discover_tests(loader-tests)

# Model registry tests (PR G1.6) — no model file needed.
add_executable(model-registry-tests
    unit/test_model_registry.cpp
)

target_link_libraries(model-registry-tests PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-loader
    core
    ggml
)

target_compile_features(model-registry-tests PRIVATE cxx_std_17)
gtest_discover_tests(model-registry-tests)

# Model registry tests (PR G1.6) — no model file needed.
add_executable(qwen3-model-registry-tests
    unit/test_model_registry.cpp
)

target_link_libraries(qwen3-model-registry-tests PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-loader
    core
    ggml
)

target_compile_features(loader-tests PRIVATE cxx_std_17)
gtest_discover_tests(loader-tests)

# Model registry tests (PR G1.6) — no model file needed.
add_executable(model-registry-tests
    unit/test_model_registry.cpp
)

target_link_libraries(model-registry-tests PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-loader
    core
    ggml
)

target_compile_features(model-registry-tests PRIVATE cxx_std_17)
gtest_discover_tests(model-registry-tests)

# TransformerBlock unit tests — no model file needed.
# Covers: build succeeds, output finite/shape, scratch budget, Qwen2 bias path.
add_executable(transformer-block-tests
    unit/test_transformer_block.cpp
)

target_link_libraries(transformer-block-tests PRIVATE
    GTest::gtest_main
    qinf-layers
    qinf-state
    ggml
)

target_compile_features(transformer-block-tests PRIVATE cxx_std_17)
gtest_discover_tests(transformer-block-tests)

# ── Gemma 2 unit tests (Phase G2) ────────────────────────────────────────────

# Softcap op: build_softcap(x, cap) = cap * tanh(x/cap) — no model file needed.
add_executable(gemma2-softcap-tests
    unit/test_gemma2_softcap.cpp
)
target_link_libraries(gemma2-softcap-tests PRIVATE
    GTest::gtest_main
    qinf-layers
    ggml
)
target_compile_features(gemma2-softcap-tests PRIVATE cxx_std_17)
gtest_discover_tests(gemma2-softcap-tests)

# Sandwich norm: post_attn_norm / post_ffn_norm hooks in build_transformer_layer.
# No model file needed.
add_executable(gemma2-sandwich-norm-tests
    unit/test_gemma2_sandwich_norm.cpp
)
target_link_libraries(gemma2-sandwich-norm-tests PRIVATE
    GTest::gtest_main
    qinf-layers
    qinf-state
    ggml
)
target_compile_features(gemma2-sandwich-norm-tests PRIVATE cxx_std_17)
gtest_discover_tests(gemma2-sandwich-norm-tests)

# Sliding-window: Gemma2Config::from_metadata layer_window pattern +
# mask logic — no model file needed.
add_executable(gemma2-sliding-window-tests
    unit/test_gemma2_sliding_window.cpp
)
target_link_libraries(gemma2-sliding-window-tests PRIVATE
    GTest::gtest_main
    qinf-models
    core
)
target_compile_features(gemma2-sliding-window-tests PRIVATE cxx_std_17)
gtest_discover_tests(gemma2-sliding-window-tests)

# ── TQ prefill canary tests ───────────────────────────────────────────────────
# Regression gate for the TurboQuant prefill path (_build_layer_batch_graph).
# Prerequisite for the build_transformer_layer extraction PR.
# Self-skips when QWEN3_MODEL_PATH is unset or the model file is absent.
# See: docs/kv-cache-unification.md open question #3.
add_executable(tq-prefill-tests
    unit/test_qwen3_tq_prefill.cpp
)

# Bake the source-tree fixture path into the binary so it resolves correctly
# regardless of the working directory from which tests are invoked.
target_compile_definitions(tq-prefill-tests PRIVATE
    QWEN3_TQ_BASELINE_DEFAULT="${CMAKE_SOURCE_DIR}/tests/fixtures/qwen3_tq_prefill_baseline.json"
)

target_link_libraries(tq-prefill-tests PRIVATE
    GTest::gtest_main
    qinf-models
    core
    nlohmann_json::nlohmann_json
    ggml
)

target_compile_features(tq-prefill-tests PRIVATE cxx_std_17)
gtest_discover_tests(tq-prefill-tests)
