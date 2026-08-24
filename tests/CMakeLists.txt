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
    unit/test_graph_input.cpp
    unit/test_tokens_input.cpp
    unit/test_positions_input.cpp
    unit/test_gather_indices_input.cpp
    unit/test_sparse_head_input.cpp
)

# Link libraries for unit tests
target_link_libraries(unit-tests PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-graph-inputs
    core
    sampling
    ggml
)

# Discover and add unit tests to CTest
include(GoogleTest)
gtest_discover_tests(unit-tests)

# AttnMaskInput — dedicated gtest executable using gtest_main, so the
# causal / sliding-window / image-span-bidi (Phase 4) mask gate actually runs.
add_executable(attn-mask-input-tests
    unit/test_attn_mask_input.cpp
)
target_link_libraries(attn-mask-input-tests PRIVATE
    GTest::gtest_main
    qinf-graph-inputs
    ggml
)
target_compile_features(attn-mask-input-tests PRIVATE cxx_std_17)
gtest_discover_tests(attn-mask-input-tests)

# GrammarVocab — standalone non-GTest harness. test_grammar_vocab.cpp defines
# its own int main() (g_pass/g_fail counters, no RUN_ALL_TESTS). It lived in the
# unit-tests aggregate, where its main() hijacked the linker and silently
# suppressed every gtest TEST() in the other unit-tests sources (and broke
# gtest_discover_tests). Kept as its own executable so both run independently.
add_executable(grammar-vocab-tests
    unit/test_grammar_vocab.cpp
)
target_link_libraries(grammar-vocab-tests PRIVATE
    sampling
)
target_compile_features(grammar-vocab-tests PRIVATE cxx_std_17)
add_test(NAME grammar-vocab-tests COMMAND grammar-vocab-tests)

# Speculative decoding — standalone non-GTest harness (its own int main()).
# prompt_lookup.h / speculative.h are header-only under src/sampling, reachable
# via the sampling target's PUBLIC include dir.
add_executable(speculative-tests
    unit/test_speculative.cpp
)
target_link_libraries(speculative-tests PRIVATE
    sampling
)
target_compile_features(speculative-tests PRIVATE cxx_std_17)
add_test(NAME speculative-tests COMMAND speculative-tests)

# Draft-source seam (Phase 1 of the MTP plan). Header-only under src/sampling,
# same standalone-harness style as speculative-tests.
add_executable(draft-source-tests
    unit/test_draft_source.cpp
)
target_link_libraries(draft-source-tests PRIVATE
    sampling
)
target_compile_features(draft-source-tests PRIVATE cxx_std_17)
add_test(NAME draft-source-tests COMMAND draft-source-tests)

# Sampler semantics, co-located with src/sampling/sampling.cpp. Pins "greedy
# means argmax" -- see the comment at the top of the test file.
add_executable(sampling-tests
    unit/test_sampling.cpp
)
target_link_libraries(sampling-tests PRIVATE
    GTest::gtest_main
    sampling
)
target_compile_features(sampling-tests PRIVATE cxx_std_17)
gtest_discover_tests(sampling-tests)

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

# Byte-identical differential harness (compile-identical on main + branch).
add_executable(logits-dump
    diff/logits_dump.cpp
    ${CMAKE_SOURCE_DIR}/src/core/decode_step.cpp
    ${CMAKE_SOURCE_DIR}/src/core/decode_plan.cpp
)
target_link_libraries(logits-dump PRIVATE
    qinf-models
    core
    sampling
    ggml
)

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

# TokenTrie unit tests (GTest, no model needed)
add_executable(token-trie-tests
    unit/test_token_trie.cpp
)

target_link_libraries(token-trie-tests PRIVATE
    GTest::gtest_main
    sampling
)

gtest_discover_tests(token-trie-tests)

# DecodePlan truth-table tests (GTest, no model needed). Enumerates the
# resolver's input combinations and asserts the resolved plan — the
# "first-class" half of the consolidation. Drops the self-tautology
# rejections from the resolver and re-asserts them here on resolved values.
add_executable(decode-plan-tests
    unit/test_decode_plan.cpp
    ${CMAKE_SOURCE_DIR}/src/core/decode_plan.cpp
)

target_link_libraries(decode-plan-tests PRIVATE
    GTest::gtest_main
    qinf-models
)

target_compile_features(decode-plan-tests PRIVATE cxx_std_17)
gtest_discover_tests(decode-plan-tests)

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

# Chat-endpoint image-input parsing gate (server image_url data URIs). No model
# file / encoder — pure request parsing + fail-loud guards. Compiles the server
# parse module directly (header-only consumers + nlohmann json).
add_executable(image-data-uri-tests
    unit/test_image_data_uri.cpp
    ${CMAKE_SOURCE_DIR}/src/server/image_data_uri.cpp
)
target_link_libraries(image-data-uri-tests PRIVATE
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)
target_compile_features(image-data-uri-tests PRIVATE cxx_std_17)
gtest_discover_tests(image-data-uri-tests)

# Session snapshot seams (Phase 0). Model-free — links only qinf-session.
add_executable(session-tests
    unit/test_session_manifest.cpp
)
target_link_libraries(session-tests PRIVATE
    GTest::gtest_main
    qinf-session
)
target_compile_features(session-tests PRIVATE cxx_std_17)
gtest_discover_tests(session-tests)

# L1 control-state sections (Phase 1a). Model-free — links sampling + qinf-state
# (both pull qinf-session). Proves RNG/grammar/token sections round-trip bitwise.
add_executable(l1-section-tests
    unit/test_l1_sections.cpp
)
target_link_libraries(l1-section-tests PRIVATE
    GTest::gtest_main
    sampling
    qinf-state
)
target_compile_features(l1-section-tests PRIVATE cxx_std_17)
gtest_discover_tests(l1-section-tests)

# Image-embedding V1 sections + persistent store (vision). Model-free — links
# core (pulls qinf-graph-inputs + qinf-session). Byte round-trip + store
# hit/miss/version-gating semantics.
add_executable(image-embedding-section-tests
    unit/test_image_embedding_section.cpp
)
target_link_libraries(image-embedding-section-tests PRIVATE
    GTest::gtest_main
    core
)
target_compile_features(image-embedding-section-tests PRIVATE cxx_std_17)
gtest_discover_tests(image-embedding-section-tests)

# Inference server warm chat-prefix routing (--chat-prefix-cache). Model-free —
# inference_server.h is header-only (std + forward decls), so this links only
# gtest. Pins warm_prefix_pick: strict-prefix-append, longest-match, free-slot
# eligibility. The model-on-Metal transparency gate is server_chat_cache_smoke.sh.
add_executable(inference-server-tests
    unit/test_inference_server.cpp
)
target_link_libraries(inference-server-tests PRIVATE
    GTest::gtest_main
)
target_compile_features(inference-server-tests PRIVATE cxx_std_17)
gtest_discover_tests(inference-server-tests)

# Prefix library (Phase 4 L2 service surface). Model-free — links core (pulls
# qinf-session for CompatHeader + snapshot_io). Key stability, hit/miss, the
# fail-loud header refusal (the Phase 4 non-negotiable), eviction, atomic store.
add_executable(prefix-library-tests
    unit/test_prefix_library.cpp
)
target_link_libraries(prefix-library-tests PRIVATE
    GTest::gtest_main
    core
)
target_compile_features(prefix-library-tests PRIVATE cxx_std_17)
gtest_discover_tests(prefix-library-tests)

# L1 session round-trip (Phase 1b). Model-based standalone harness (takes a GGUF
# path as argv[1]); NOT auto-discovered — run manually against a Qwen and a Gemma
# recipe. Captures mid-decode, restores via feed_tokens, asserts token-stable
# continuation (gate A) + bitwise restore-vs-restore on CPU (gate B).
add_executable(test-l1-session-roundtrip
    integration/test_l1_session_roundtrip.cpp
)
target_link_libraries(test-l1-session-roundtrip PRIVATE
    qinf-models
    qinf-state
    core
    sampling
    ggml
)
target_compile_features(test-l1-session-roundtrip PRIVATE cxx_std_17)

# L2 session round-trip (Phase 3). Model-based standalone harness (GGUF argv[1]).
# Gate 1 fidelity=byte (vs live), Gate 2 substitutability=token-stable (vs L1
# feed_tokens), Gate 3 build_path_tag fail-loud. Run on Qwen35 (KV+recurrent) and
# Gemma3 (KV only).
add_executable(test-l2-session-roundtrip
    integration/test_l2_session_roundtrip.cpp
)
target_link_libraries(test-l2-session-roundtrip PRIVATE
    qinf-models
    qinf-state
    core
    sampling
    ggml
)
target_compile_features(test-l2-session-roundtrip PRIVATE cxx_std_17)

# Prefix-library round-trip (Phase 4). Model-based standalone harness (GGUF
# argv[1]). Gate 1 fidelity=byte (WARM==LIVE), Gate 2 substitutability=token-
# stable (WARM==REF prefill-from-scratch), Gate 3 header mismatch fail-loud. Run
# on Qwen35 (KV+recurrent) and Gemma3 (KV only).
add_executable(test-prefix-library-roundtrip
    integration/test_prefix_library_roundtrip.cpp
)
target_link_libraries(test-prefix-library-roundtrip PRIVATE
    qinf-models
    qinf-state
    core
    sampling
    ggml
)
target_compile_features(test-prefix-library-roundtrip PRIVATE cxx_std_17)

# Image-prefix round-trip (Vision V2). Model-based standalone harness; takes a
# GGUF model AND an mmproj (argv[1], argv[2]). Captures the post-image-prefill
# KV blob under the image-extended key, then proves WARM (load blob, skip
# encode+image-prefill) == LIVE (chunked-from-scratch) BYTE. Links vision +
# the consumer-compiled multimodal_prefill (same arrangement as the CLI). Run on
# a Gemma3 (SigLIP) AND a Gemma4 (gemma4uv) image path.
add_executable(test-image-prefix-roundtrip
    integration/test_image_prefix_roundtrip.cpp
    ${CMAKE_SOURCE_DIR}/src/core/multimodal_prefill.cpp
    ${CMAKE_SOURCE_DIR}/src/core/image_embedding_cache.cpp
)
target_link_libraries(test-image-prefix-roundtrip PRIVATE
    qinf-models
    qinf-vision
    qinf-state
    core
    sampling
    ggml
)
target_compile_features(test-image-prefix-roundtrip PRIVATE cxx_std_17)

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

# PR 4.2.A — equivalence test for ggml_deltanet_post_state (qinf-patched ggml op).
# Exercises the fused op against the chained 4-op reference (rms_norm + mul +
# silu + mul). Parameterized over CPU and Metal backends; the Metal case
# self-skips when GGML_USE_METAL is OFF. No model file needed.
add_executable(deltanet-post-state-tests
    unit/test_deltanet_post_state.cpp
)

target_link_libraries(deltanet-post-state-tests PRIVATE
    GTest::gtest_main
    ggml
)

if(GGML_USE_METAL)
    target_link_libraries(deltanet-post-state-tests PRIVATE ggml-metal)
endif()

target_compile_features(deltanet-post-state-tests PRIVATE cxx_std_17)
gtest_discover_tests(deltanet-post-state-tests)

# PR 4.2.B — equivalence test for ggml_deltanet_pre_state (qinf-patched ggml op).
# Exercises the fused op against the chained 4-op reference (l2_norm × 2 +
# repeat_4d × 2). Parameterized over CPU and Metal backends; the Metal case
# self-skips when GGML_USE_METAL is OFF. No model file needed.
add_executable(deltanet-pre-state-tests
    unit/test_deltanet_pre_state.cpp
)

target_link_libraries(deltanet-pre-state-tests PRIVATE
    GTest::gtest_main
    ggml
)

if(GGML_USE_METAL)
    target_link_libraries(deltanet-pre-state-tests PRIVATE ggml-metal)
endif()

target_compile_features(deltanet-pre-state-tests PRIVATE cxx_std_17)
gtest_discover_tests(deltanet-pre-state-tests)

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

# Qemmi-Lens attention tap (docs/plan-qemmi-lens.md P1/A1) — the ForwardPassBase
# tap seam gated off-path byte-identical, on-path live-softmax, fail-loud.
# Hosted on qwen36 (the pinned lens model); self-skips when QWEN36_MODEL_PATH unset.
add_executable(forward-pass-base-tests
    unit/test_forward_pass_base.cpp
)

target_link_libraries(forward-pass-base-tests PRIVATE
    GTest::gtest_main
    qinf-models
    core
    ggml
)

target_compile_features(forward-pass-base-tests PRIVATE cxx_std_17)
gtest_discover_tests(forward-pass-base-tests)

# Qemmi-Lens pure computation (docs/plan-qemmi-lens.md P2) — compute_lens_report
# / lens_report_to_json exercised with synthetic tap rows (no model). Compiles
# server_lens.cpp directly; the driver half links the model/grammar libs.
add_executable(server-lens-tests
    unit/test_server_lens.cpp
    ${CMAKE_SOURCE_DIR}/src/server/server_lens.cpp
)

target_link_libraries(server-lens-tests PRIVATE
    GTest::gtest_main
    nlohmann_json::nlohmann_json   # server_lens.cpp: the shape contract's real parse
    qinf-models
    qinf-loader
    sampling
    core
    ggml
)

target_compile_features(server-lens-tests PRIVATE cxx_std_17)
gtest_discover_tests(server-lens-tests)

# feed_tokens Phase 1 — qwen36 head-less state-advance + mid-decode
# differential (docs/plan-feed-tokens.md). Self-skips when model absent.
add_executable(feed-tokens-tests-qwen36
    unit/test_qwen36_feed_tokens.cpp
)

target_link_libraries(feed-tokens-tests-qwen36 PRIVATE
    GTest::gtest_main
    qinf-models
    core
    ggml
)

target_compile_features(feed-tokens-tests-qwen36 PRIVATE cxx_std_17)
gtest_discover_tests(feed-tokens-tests-qwen36)

# feed_tokens Phase 2 — qwen35 head-less state-advance + mid-decode
# differential (docs/plan-feed-tokens.md). Self-skips when model absent.
add_executable(feed-tokens-tests-qwen35
    unit/test_qwen35_feed_tokens.cpp
)

target_link_libraries(feed-tokens-tests-qwen35 PRIVATE
    GTest::gtest_main
    qinf-models
    core
    ggml
)

target_compile_features(feed-tokens-tests-qwen35 PRIVATE cxx_std_17)
gtest_discover_tests(feed-tokens-tests-qwen35)

# feed_tokens Phase 3 — gemma1–4 head-less state-advance + per-recipe
# KV-append mid-decode differential (docs/plan-feed-tokens.md). One
# parameterized harness; each recipe self-skips when its model is absent.
add_executable(feed-tokens-tests-gemma
    unit/test_gemma_feed_tokens.cpp
)

target_link_libraries(feed-tokens-tests-gemma PRIVATE
    GTest::gtest_main
    qinf-models
    core
    ggml
)

target_compile_features(feed-tokens-tests-gemma PRIVATE cxx_std_17)
gtest_discover_tests(feed-tokens-tests-gemma)

# Gemma batched decode — per-recipe two-tier differential (Tier-1 single-slot
# bitwise vs single-token run_prefill; Tier-2 multi-slot token-stable) for the
# new build_decoding_graph. One parameterized harness; Phase 1 wires Gemma 1.
# docs/plan-gemma-batched-decode.md. Self-skips when the model is absent.
add_executable(batched-decode-tests-gemma
    unit/test_gemma_batched_decode.cpp
)

target_link_libraries(batched-decode-tests-gemma PRIVATE
    GTest::gtest_main
    qinf-models
    core
    ggml
)

target_compile_features(batched-decode-tests-gemma PRIVATE cxx_std_17)
gtest_discover_tests(batched-decode-tests-gemma)

# Decode KV write: Cpy vs SetRows differential — P1 gate of
# docs/plan-persistent-decode-graph.md. Pure-unit KvWriteIndicesInput tests
# run unconditionally; the Qwen3.5 + Gemma3 bitwise model differentials
# self-skip when their model is absent.
add_executable(kv-write-setrows-tests
    unit/test_kv_write_setrows.cpp
)

target_link_libraries(kv-write-setrows-tests PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-graph-inputs
    core
    ggml
)

target_compile_features(kv-write-setrows-tests PRIVATE cxx_std_17)
gtest_discover_tests(kv-write-setrows-tests)

# Decode n_kv bucketing: exact vs bucketed (8 / 256) — P2 gate of
# docs/plan-persistent-decode-graph.md. Bitwise per-step logits, Qwen3.5 +
# Gemma3; self-skips when the model is absent.
add_executable(decode-kv-bucket-tests
    unit/test_decode_kv_bucket.cpp
)

target_link_libraries(decode-kv-bucket-tests PRIVATE
    GTest::gtest_main
    qinf-models
    core
    ggml
)

target_compile_features(decode-kv-bucket-tests PRIVATE cxx_std_17)
gtest_discover_tests(decode-kv-bucket-tests)

# Persistent decode graph reuse — P3 gate of docs/plan-persistent-decode-graph.md.
# Reused graph == same-shape rebuilt graph, bitwise (isolates reuse from the
# bucketing fork); + reuse/rebuild counters cross a bucket boundary. Self-skips.
add_executable(decode-graph-cache-tests
    unit/test_decode_graph_cache.cpp
)

target_link_libraries(decode-graph-cache-tests PRIVATE
    GTest::gtest_main
    qinf-models
    core
    ggml
)

target_compile_features(decode-graph-cache-tests PRIVATE cxx_std_17)
gtest_discover_tests(decode-graph-cache-tests)

# Prefill output-ID slice (last-token LM-head elision). No-model fail-loud
# unit tests run unconditionally; the qwen35 + gemma1 bit-identical /
# composition differentials self-skip when their model is absent.
add_executable(out-ids-slice-tests
    unit/test_out_ids_slice.cpp
)

target_link_libraries(out-ids-slice-tests PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-graph-inputs
    core
    ggml
)

target_compile_features(out-ids-slice-tests PRIVATE cxx_std_17)
gtest_discover_tests(out-ids-slice-tests)

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

# MTP/NextN inventory validation (Phase 2 of the MTP plan) — synthetic metadata.
add_executable(mtp-inventory-tests
    unit/test_mtp_inventory.cpp
)
target_link_libraries(mtp-inventory-tests PRIVATE
    GTest::gtest_main
    qinf-models
    core
)
target_compile_features(mtp-inventory-tests PRIVATE cxx_std_17)
gtest_discover_tests(mtp-inventory-tests)

# qwen35 NextN / MTP head recognition (Qwen3.8 ships one, Qwen3.5 does not)
# — synthetic metadata, no GGUF needed.
add_executable(qwen35-nextn-tests
    unit/test_qwen35_nextn.cpp
)
target_link_libraries(qwen35-nextn-tests PRIVATE
    GTest::gtest_main
    qinf-models
    core
)
target_compile_features(qwen35-nextn-tests PRIVATE cxx_std_17)
gtest_discover_tests(qwen35-nextn-tests)

# MTP/NextN head graph (Phase 3) — needs the real MTP GGUF in models/;
# skips when absent. Run from the repo root (relative model path).
add_executable(mtp-head-tests
    unit/test_mtp_head.cpp
)
target_link_libraries(mtp-head-tests PRIVATE
    GTest::gtest_main
    qinf-models
    core
    ggml
)
target_compile_features(mtp-head-tests PRIVATE cxx_std_17)
gtest_discover_tests(mtp-head-tests)

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
    qinf-session
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

# ── G4.7: Gemma 4 tokenizer + chat template tests ────────────────────────────
# Covers gemma4_tokenizer_config() shape, encode/decode round-trip with
# BOS=2 / EOS=106, and registry dispatch ("gemma4" → GemmaChatTemplate +
# gemma4_tokenizer_config()).  No model file required.
add_executable(gemma4-tokenizer-tests
    unit/test_tokenizer_gemma4.cpp
)
target_link_libraries(gemma4-tokenizer-tests PRIVATE
    GTest::gtest_main
    qinf-loader
    qinf-models
)
target_compile_features(gemma4-tokenizer-tests PRIVATE cxx_std_17)
gtest_discover_tests(gemma4-tokenizer-tests)

add_executable(gemma4-chat-template-tests
    unit/test_chat_template_gemma4.cpp
)
target_link_libraries(gemma4-chat-template-tests PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-loader
    core
    ggml
)
target_compile_features(gemma4-chat-template-tests PRIVATE cxx_std_17)
gtest_discover_tests(gemma4-chat-template-tests)

# Gemma4Config + inventory validator unit tests ──────────────────────
# Pure-unit (no model file). Synthesizes A4B-shaped metadata + tensor
# inventory in memory and exercises Gemma4Config::from_metadata + the
# widened validate_gemma4_inventory.
add_executable(gemma4-recipe-tests
    unit/test_gemma4_recipe.cpp
)
target_link_libraries(gemma4-recipe-tests PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-loader
    core
    ggml
)
target_compile_features(gemma4-recipe-tests PRIVATE cxx_std_17)
gtest_discover_tests(gemma4-recipe-tests)

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

# Model registry tests — no model file needed.
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

# Multimodal-only refusal — no model file needed.
add_executable(multimodal-check-tests
    unit/test_multimodal_check.cpp
)

target_link_libraries(multimodal-check-tests PRIVATE
    GTest::gtest_main
    qinf-loader
    core
    ggml
)

target_compile_features(multimodal-check-tests PRIVATE cxx_std_17)
gtest_discover_tests(multimodal-check-tests)

# Gemma 4 channel-stream filter — pure string logic, no model file needed.
add_executable(channel-filter-tests
    unit/test_channel_filter.cpp
)

target_link_libraries(channel-filter-tests PRIVATE
    GTest::gtest_main
    qinf-loader
)

target_compile_features(channel-filter-tests PRIVATE cxx_std_17)
gtest_discover_tests(channel-filter-tests)

# Vision loader (Phase 2.1) — requires ./mmproj-BF16.gguf to be present;
# self-skips otherwise. See docs/plan-gemma-vision-impl.md.
add_executable(vision-loader-tests
    unit/test_vision_loader.cpp
)

target_link_libraries(vision-loader-tests PRIVATE
    GTest::gtest_main
    qinf-vision
    qinf-loader
    qinf-models
    core
    ggml
)

target_compile_features(vision-loader-tests PRIVATE cxx_std_17)
gtest_discover_tests(vision-loader-tests)

# SigLIP encoder (Gemma 3 vision, Phase 2.3/2.4) — SigLIP ViT forward-graph +
# differential gate. Requires ./mmproj-BF16.gguf; self-skips otherwise. Renamed
# from vision-encoder-tests when the encoder class became SiglipEncoder (Seam A).
add_executable(siglip-encoder-tests
    unit/test_siglip_encoder.cpp
)

target_link_libraries(siglip-encoder-tests PRIVATE
    GTest::gtest_main
    qinf-vision
    qinf-loader
    qinf-models
    core
    ggml
)

target_compile_features(siglip-encoder-tests PRIVATE cxx_std_17)
if(GGML_USE_METAL)
    target_link_libraries(siglip-encoder-tests PRIVATE ggml-metal)
    target_compile_definitions(siglip-encoder-tests PRIVATE QINF_TEST_HAS_METAL)
endif()
gtest_discover_tests(siglip-encoder-tests)

# Gemma 4 unified-vision encoder (gemma4uv) — blockless projector graph smoke +
# loader-acceptance gate. Requires ./mmproj-gemma-4-12B-it-Q8_0.gguf; self-skips
# otherwise. The bit-level differential vs llama.cpp is DISABLED pending a
# captured reference (plan §8 gate 1).
add_executable(gemma4uv-encoder-tests
    unit/test_gemma4uv_encoder.cpp
)

target_link_libraries(gemma4uv-encoder-tests PRIVATE
    GTest::gtest_main
    qinf-vision
    qinf-loader
    qinf-models
    core
    ggml
)

target_compile_features(gemma4uv-encoder-tests PRIVATE cxx_std_17)
if(GGML_USE_METAL)
    target_link_libraries(gemma4uv-encoder-tests PRIVATE ggml-metal)
    target_compile_definitions(gemma4uv-encoder-tests PRIVATE QINF_TEST_HAS_METAL)
endif()
gtest_discover_tests(gemma4uv-encoder-tests)

# Multimodal prefill orchestrator (Phase 5 vision). Compiles the orchestrator
# source directly (it lives in src/core/ but is not in the `core` lib — same
# arrangement as decode_step.cpp — because it bridges qinf-vision + qinf-models
# which both sit above `core`). Self-skips without a Gemma 3 text model + mmproj.
add_executable(multimodal-prefill-tests
    unit/test_multimodal_prefill.cpp
    ${CMAKE_SOURCE_DIR}/src/core/multimodal_prefill.cpp
    ${CMAKE_SOURCE_DIR}/src/core/image_embedding_cache.cpp
    # cli host-side image pipeline, compiled directly (cli is an exe, not a lib)
    # so the e2e gate can decode a real image + expand the marker stream.
    ${CMAKE_SOURCE_DIR}/src/cli/image_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/image_prompt.cpp
)
target_link_libraries(multimodal-prefill-tests PRIVATE
    GTest::gtest_main
    qinf-vision
    qinf-models
    qinf-loader
    core
    ggml
)
target_compile_features(multimodal-prefill-tests PRIVATE cxx_std_17)
if(GGML_USE_METAL)
    target_link_libraries(multimodal-prefill-tests PRIVATE ggml-metal)
    target_compile_definitions(multimodal-prefill-tests PRIVATE QINF_TEST_HAS_METAL)
endif()
gtest_discover_tests(multimodal-prefill-tests)

# Phase 6 host-side image pipeline — no model file needed. Both compile their
# src/cli/ sources directly (the cli builds an executable, not a lib).
add_executable(image-loader-tests
    unit/test_image_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/image_loader.cpp
)
target_link_libraries(image-loader-tests PRIVATE GTest::gtest_main)
target_compile_features(image-loader-tests PRIVATE cxx_std_17)
gtest_discover_tests(image-loader-tests)

add_executable(image-prompt-tests
    unit/test_image_prompt.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/image_prompt.cpp
)
target_link_libraries(image-prompt-tests PRIVATE GTest::gtest_main)
target_compile_features(image-prompt-tests PRIVATE cxx_std_17)
gtest_discover_tests(image-prompt-tests)

# Phase 7 image-embedding reuse cache — no model file needed (counting lambda).
add_executable(image-embedding-cache-tests
    unit/test_image_embedding_cache.cpp
    ${CMAKE_SOURCE_DIR}/src/core/image_embedding_cache.cpp
)
target_link_libraries(image-embedding-cache-tests PRIVATE GTest::gtest_main)
target_compile_features(image-embedding-cache-tests PRIVATE cxx_std_17)
gtest_discover_tests(image-embedding-cache-tests)

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

# LayerSpec and per-layer attention shape tests ──────────────────────
# Covers: LayerSpec vector construction from synthetic per-layer arrays;
#   degenerate case bit-identical to zero-default path; asymmetric head_dim
#   build-and-compute succeeds (finite output).  No model file required.
add_executable(layer-spec-tests
    unit/test_layer_spec.cpp
    unit/test_transformer_block_ple.cpp   # G4.2 second-residual slot
)

target_link_libraries(layer-spec-tests PRIVATE
    GTest::gtest_main
    qinf-layers
    qinf-state
    ggml
)

target_compile_features(layer-spec-tests PRIVATE cxx_std_17)
gtest_discover_tests(layer-spec-tests)

# Pruned RoPE op tests ───────────────────────────────────────────────
# Covers: zero-position identity; tail dims pass through; full rotation matches
# direct ggml_rope_ext; non-trivial rotation under pos>0; NEOX-pair oracle.
# No model file required.
add_executable(rope-pruned-tests
    unit/test_rope_pruned.cpp
)
target_link_libraries(rope-pruned-tests PRIVATE
    GTest::gtest_main
    qinf-layers
    ggml
)
target_compile_features(rope-pruned-tests PRIVATE cxx_std_17)
gtest_discover_tests(rope-pruned-tests)

# Final logit soft-cap tests ─────────────────────────────────────────
# Covers: build_softcap on an LM-head-shaped tensor with
# cap=30.  Hand-rolled oracle: 30 * tanh(x/30) over a curated value range.
# No model file required.
add_executable(gemma4-final-softcap-tests
    unit/test_gemma4_final_softcap.cpp
)
target_link_libraries(gemma4-final-softcap-tests PRIVATE
    GTest::gtest_main
    qinf-layers
    ggml
)
target_compile_features(gemma4-final-softcap-tests PRIVATE cxx_std_17)
gtest_discover_tests(gemma4-final-softcap-tests)

# PLE (per-layer embedding) op tests ─────────────────────────────────
# Covers: build_ple lookup + projection against a synthetic oracle.  PLE is
# inactive in the 26B-A4B checkpoint (embedding_length_per_layer_input=0);
# this is the synthetic-config gate the dense 31B follow-up will promote.
# No model file required.
add_executable(ple-tests
    unit/test_ple.cpp
)
target_link_libraries(ple-tests PRIVATE
    GTest::gtest_main
    qinf-layers
    ggml
)
target_compile_features(ple-tests PRIVATE cxx_std_17)
gtest_discover_tests(ple-tests)

# KV cache reference-mode tests ──────────────────────────────────────
# Covers: synthetic state-isolation — the last N layers are configured as
# references to a producing layer; verifies (a) reference layers' K/V tensors
# alias the source layer, (b) writes to the source are visible at the reference
# without an explicit copy.  Inactive in 26B-A4B (shared_kv_layers=0).  No
# model file required.
add_executable(kv-reference-mode-tests
    unit/test_kv_reference_mode.cpp
)
target_link_libraries(kv-reference-mode-tests PRIVATE
    GTest::gtest_main
    qinf-state
    ggml
)
target_compile_features(kv-reference-mode-tests PRIVATE cxx_std_17)
gtest_discover_tests(kv-reference-mode-tests)

# ── Gemma 2 unit tests ────────────────────────────────────────────

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

# ── Gemma 3 unit tests ────────────────────────────────────────────

# Gemma3Config: 5:1 window pattern, per-layer RoPE base, QK-norm semantics.
# No model file required.
add_executable(gemma3-config-tests
    unit/test_gemma3_config.cpp
)
target_link_libraries(gemma3-config-tests PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-layers
    core
    ggml
)
target_compile_features(gemma3-config-tests PRIVATE cxx_std_17)
gtest_discover_tests(gemma3-config-tests)

# Gemma 3 forward pass integration test.
# Self-skips when GEMMA3_MODEL_PATH is unset and default file is absent.
add_executable(forward-tests-gemma3
    unit/test_gemma3_forward.cpp
)
target_link_libraries(forward-tests-gemma3 PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-loader
    core
    ggml
)
target_compile_features(forward-tests-gemma3 PRIVATE cxx_std_17)
gtest_discover_tests(forward-tests-gemma3)

# Gemma 3 image-embedding substitution (Phase 3 vision). Self-consistency gate;
# self-skips without a Gemma 3 text model. See docs/plan-gemma-vision-impl.md.
add_executable(gemma3-image-substitution-tests
    unit/test_gemma3_image_substitution.cpp
)
target_link_libraries(gemma3-image-substitution-tests PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-loader
    core
    ggml
)
target_compile_features(gemma3-image-substitution-tests PRIVATE cxx_std_17)
gtest_discover_tests(gemma3-image-substitution-tests)

# Gemma 4 recipe image-substitution hook (plan-gemma4-vision-impl §6/§8). Same
# self-consistency gate as Gemma 3, minus the bidi-mask wiring (Gemma 4 image
# attention is causal). Needs the dense gemma-4-12B-it text export; self-skips.
add_executable(gemma4-image-substitution-tests
    unit/test_gemma4_image_substitution.cpp
)
target_link_libraries(gemma4-image-substitution-tests PRIVATE
    GTest::gtest_main
    qinf-models
    qinf-loader
    core
    ggml
)
target_compile_features(gemma4-image-substitution-tests PRIVATE cxx_std_17)
gtest_discover_tests(gemma4-image-substitution-tests)

# ── Phase 4 tools ─────────────────────────────────────────────────────────────

# PR 4.0 — Performance baseline.
# Not a GTest binary — run manually, pipe output to the fixture file.
# Requires QWEN36_MODEL_PATH and/or GEMMA4_MODEL_PATH env vars.
add_executable(perf-baseline
    perf/perf_baseline.cpp
)

target_compile_definitions(perf-baseline PRIVATE
    CANARY_PROMPTS_DEFAULT="${CMAKE_SOURCE_DIR}/tests/fixtures/canary_prompts_gemma.txt"
)

target_link_libraries(perf-baseline PRIVATE
    qinf-models
    qinf-loader
    core
    sampling
    nlohmann_json::nlohmann_json
    ggml
)

target_compile_features(perf-baseline PRIVATE cxx_std_17)

# Batch-scaling probe — decode wall-clock at B = 1..10 (the "empty seats" test).
add_executable(batch-scaling
    perf/batch_scaling.cpp
)
target_link_libraries(batch-scaling PRIVATE
    qinf-models
    qinf-loader
    core
    sampling
    ggml
)
target_compile_features(batch-scaling PRIVATE cxx_std_17)

# Decode-step phase decomposition — locate the ~20ms fixed intercept.
add_executable(decode-breakdown
    perf/decode_breakdown.cpp
)
target_link_libraries(decode-breakdown PRIVATE
    qinf-models
    qinf-loader
    core
    sampling
    ggml
)
target_compile_features(decode-breakdown PRIVATE cxx_std_17)

# Alloc-reuse confirmation — is the galloc replan defeated by KV shape growth?
add_executable(alloc-reuse
    perf/alloc_reuse.cpp
)
target_link_libraries(alloc-reuse PRIVATE
    qinf-models
    qinf-loader
    core
    sampling
    ggml
)
target_compile_features(alloc-reuse PRIVATE cxx_std_17)

# Attention-provenance probe (N3) — does the decode attention row point at the
# value's source span? Reads back the post-softmax kq_soft.<il> tensors.
# Compiles server_lens.cpp so the QDOCS_S1 Stage 1 gate measures the SHIPPED
# lens math (run_lens_extract / compute_lens_report / the A5.4 predicates),
# not this file's older inline copy — the gate would be worthless against a
# lookalike. Same source-in-target pattern as server-lens-tests, same libs.
add_executable(attn-provenance
    perf/attn_provenance.cpp
    ${CMAKE_SOURCE_DIR}/src/server/server_lens.cpp
)
target_link_libraries(attn-provenance PRIVATE
    nlohmann_json::nlohmann_json   # server_lens.cpp: the shape contract's real parse
    qinf-models
    qinf-loader
    core
    sampling
    ggml
)
target_compile_features(attn-provenance PRIVATE cxx_std_17)

# PR 4.1 investigation — per-stage MoE timing.
# Disposable: once the breakdown identifies the right fusion shape, this binary
# can be deleted (or kept as a regression check).
add_executable(moe-breakdown
    perf/moe_breakdown.cpp
)

target_link_libraries(moe-breakdown PRIVATE
    qinf-models
    qinf-loader
    core
    sampling
    nlohmann_json::nlohmann_json
    ggml
)

target_compile_features(moe-breakdown PRIVATE cxx_std_17)

# Phase 4 investigation — per-stage gated-attention timing.
# Disposable: shares the breakdown_harness.h pattern with moe-breakdown.
add_executable(attention-breakdown
    perf/attention_breakdown.cpp
)

target_link_libraries(attention-breakdown PRIVATE
    qinf-models
    qinf-loader
    qinf-state
    core
    sampling
    nlohmann_json::nlohmann_json
    ggml
)

target_compile_features(attention-breakdown PRIVATE cxx_std_17)

# Phase 4 investigation — per-layer DeltaNet decode cost (coarse).
add_executable(deltanet-breakdown
    perf/deltanet_breakdown.cpp
)

target_link_libraries(deltanet-breakdown PRIVATE
    qinf-models
    qinf-loader
    qinf-layers
    qinf-state
    core
    sampling
    nlohmann_json::nlohmann_json
    ggml
)

target_compile_features(deltanet-breakdown PRIVATE cxx_std_17)

# Phase 4 PR-1 — DeltaNet sub-stage decode breakdown (resolves Q1 in
# docs/phase4-investigation.md: launch-bound vs. matmul-bound).
add_executable(deltanet-substage-breakdown
    perf/deltanet_substage_breakdown.cpp
)

target_link_libraries(deltanet-substage-breakdown PRIVATE
    qinf-models
    qinf-loader
    qinf-layers
    qinf-state
    core
    sampling
    nlohmann_json::nlohmann_json
    ggml
)

target_compile_features(deltanet-substage-breakdown PRIVATE cxx_std_17)

# Phase 4 investigation — RMSNorm marginal cost at decode shape.
add_executable(norms-breakdown
    perf/norms_breakdown.cpp
)

target_link_libraries(norms-breakdown PRIVATE
    qinf-models
    qinf-loader
    core
    sampling
    nlohmann_json::nlohmann_json
    ggml
)

target_compile_features(norms-breakdown PRIVATE cxx_std_17)
