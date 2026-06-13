// test_gemma_batched_decode.cpp — Phase 1 of docs/plan-gemma-batched-decode.md.
//
// Each Gemma recipe is gaining a real build_decoding_graph so it decodes
// through the unified batched path instead of the legacy single-token
// run_prefill bridge. This is the per-recipe correctness harness.
//
// THE ORACLE (free, exact): the single-token run_prefill path IS the correct
// output — it is the same path Gemma decode uses today via DecodeRoute::Bridge
// (decode_step.cpp). So "decode graph output ≡ single-token run_prefill output"
// is a true equivalence, not an approximation. This mirrors the feed_tokens
// differential precedent (docs/plan-feed-tokens.md).
//
// Two tiers (mirrors the project's standing Metal batch-shape fork):
//   Tier 1 — single-slot, BITWISE. The decode graph runs at batch shape N=1,
//     identical to a single-token run_prefill (also N=1). Same batch shape ⇒
//     same Metal kernel selection ⇒ bitwise equality is achievable and is the
//     gate. RUNS strict.
//       * Tier1...Bitwise      — RUNS, strict (mismatches == 0). The Phase-1 gate.
//       * Tier1...TokenStable  — RUNS, always-on guard (argmax identical).
//   Tier 2 — multi-slot, TOKEN-STABLE. Several slots decoded in ONE batched
//     graph (N>1) vs N independent single-token prefills (each N=1). Batch
//     shape differs ⇒ per the standing Metal fork, bitwise is NOT promised;
//     the gate is token-stable within a coarse ceiling. Proves the batching.
//       * Tier2...TokenStable           — RUNS, token-stable + coarse ceiling.
//       * DISABLED_Tier2...Bitwise      — DISABLED, noise-floor measurement.
//
// One parameterized harness (template on the ForwardPass type), NOT four
// copies. Phase 1 instantiates Gemma 1 only; Gemma 2/3/4 are added by their
// phases once their build_decoding_graph stops throwing. Each recipe self-skips
// when its GGUF is absent.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "../../src/core/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/gemma1.h"
#include "../../src/models/gemma2.h"
#include "../../src/models/gemma3.h"

// Coarse universal gross-regression ceiling, identical in spirit and value to
// the feed_tokens harness (kFeedTokensMaxAbsDiff). It is NOT a precision claim:
// wrong positions / corrupt KV gather → O(1) divergence AND/OR a token flip.
// One bound, all recipes — no per-recipe ε taxonomy.
static constexpr float kDecodeMaxAbsDiff = 1.0f;

static std::string env_or(const char* var, const char* fallback) {
    if (const char* e = std::getenv(var))
        if (e[0]) return std::string(e);
    return std::string(fallback);
}

static bool file_exists(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

// Bitwise float-vector diff: {count of raw-bit-differing indices, max |a-b|
// over the mismatched indices (0 if none)}. Delegated to the local Qwenium
// server and verified against this spec before use (see
// docs/qwenium-shortcomings-log.md, task-class "trivial function from exact
// signature").
static std::pair<size_t, float> bitwise_diff(const std::vector<float>& a,
                                             const std::vector<float>& b) {
    size_t diff_count = 0;
    float  max_diff   = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            ++diff_count;
            float diff = std::fabs(a[i] - b[i]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    return {diff_count, max_diff};
}

static size_t argmax(const std::vector<float>& v) {
    return static_cast<size_t>(
        std::distance(v.begin(), std::max_element(v.begin(), v.end())));
}

struct DiffResult {
    size_t mismatches   = 0;     // raw-bit mismatched logits (summed over slots)
    float  max_abs_diff = 0.0f;  // worst |a-b| over all compared logits
    bool   token_stable = true;  // argmax identical for every compared slot
    size_t n_logits     = 0;     // total logits compared
    size_t top1_ref     = 0;     // last-slot diagnostics for the failure message
    size_t top1_dec     = 0;
};

// Deterministic in-vocab token ids — no tokenizer variance. Gemma vocab is
// large (>256k); ids mod 1000 are always valid.
static std::vector<int32_t> mk_tokens(int base, int n) {
    std::vector<int32_t> v;
    for (int i = 0; i < n; ++i)
        v.push_back(static_cast<int32_t>((base + i * 7 + 3) % 1000));
    return v;
}

// Run ONE single-token decode through the unified decode graph (the
// DecodeRoute::Unified path of decode_step), returning its logits.
template <typename FP>
static std::vector<float> decode_one(FP& fp, ggml_backend_sched_t sched,
                                     int32_t token, uint32_t slot, int pos) {
    const std::vector<int32_t>  tokens    = {token};
    const std::vector<uint32_t> slots     = {slot};
    const std::vector<int32_t>  positions = {pos};
    ggml_backend_sched_reset(sched);
    ggml_cgraph* gf = fp.build_decoding_graph(tokens, slots, positions);
    ggml_backend_sched_alloc_graph(sched, gf);
    fp.set_decode_inputs(gf, tokens, slots, positions);
    ggml_backend_sched_graph_compute(sched, gf);
    fp.advance_cache(1, slot);
    return fp.get_output_logits(gf);
}

// ── Tier 1: single-slot, bitwise ─────────────────────────────────────────────
//
// Reference = single-token run_prefill onto a non-empty mid-decode cache (the
// exact DecodeRoute::Bridge oracle). Decode = build_decoding_graph for the same
// token at the same position. Same batch shape (N=1) on both sides.
template <typename FP>
static DiffResult run_tier1_single_slot(Model& model) {
    const auto& meta = model.get_metadata();
    const uint32_t vocab = meta.vocab_size;
    ggml_backend_sched_t sched = model.get_scheduler();

    const std::vector<int32_t> init = mk_tokens(11, 12);  // non-empty cache
    const int32_t X = 123;                                // the decoded token
    const int P = static_cast<int>(init.size());

    // Reference path: prefill init, then single-token run_prefill of X at P.
    FP fpRef(model, &meta, 1024, 1);
    fpRef.run_prefill(init, 0, 0, sched);
    std::vector<float> logits_ref = fpRef.run_prefill({X}, P, 0, sched);

    // Decode path: prefill init, then the unified decode graph for X at P.
    FP fpDec(model, &meta, 1024, 1);
    fpDec.run_prefill(init, 0, 0, sched);
    std::vector<float> logits_dec = decode_one(fpDec, sched, X, 0, P);

    EXPECT_EQ(logits_ref.size(), logits_dec.size());
    EXPECT_EQ(logits_ref.size(), static_cast<size_t>(vocab));

    auto [mm, mad] = bitwise_diff(logits_ref, logits_dec);
    DiffResult r;
    r.n_logits     = logits_ref.size();
    r.mismatches   = mm;
    r.max_abs_diff = mad;
    r.top1_ref     = argmax(logits_ref);
    r.top1_dec     = argmax(logits_dec);
    r.token_stable = (r.top1_ref == r.top1_dec);
    return r;
}

// ── Tier 2: multi-slot, token-stable ─────────────────────────────────────────
//
// Two slots decoded in ONE batched graph (N=2) vs two independent single-token
// run_prefills (each N=1). Batch shape differs from the reference, so the gate
// is token-stable + coarse ceiling, not bitwise.
template <typename FP>
static DiffResult run_tier2_multi_slot(Model& model) {
    const auto& meta = model.get_metadata();
    const uint32_t vocab = meta.vocab_size;
    ggml_backend_sched_t sched = model.get_scheduler();

    const std::vector<int32_t> init0 = mk_tokens(11, 12);  // slot 0 context
    const std::vector<int32_t> init1 = mk_tokens(40,  9);  // slot 1 context
    const int32_t X0 = 123, X1 = 321;
    const int P0 = static_cast<int>(init0.size());
    const int P1 = static_cast<int>(init1.size());

    // Batched path: one fp with batch capacity 2; prefill both slots, then ONE
    // decode graph over both tokens/slots.
    FP fpBatch(model, &meta, 1024, 2);
    fpBatch.run_prefill(init0, 0, /*slot=*/0, sched);
    fpBatch.run_prefill(init1, 0, /*slot=*/1, sched);

    const std::vector<int32_t>  tokens    = {X0, X1};
    const std::vector<uint32_t> slots     = {0u, 1u};
    const std::vector<int32_t>  positions = {P0, P1};
    ggml_backend_sched_reset(sched);
    ggml_cgraph* gf = fpBatch.build_decoding_graph(tokens, slots, positions);
    ggml_backend_sched_alloc_graph(sched, gf);
    fpBatch.set_decode_inputs(gf, tokens, slots, positions);
    ggml_backend_sched_graph_compute(sched, gf);
    std::vector<float> dec0 = fpBatch.get_output_logits_for_slot(gf, 0);
    std::vector<float> dec1 = fpBatch.get_output_logits_for_slot(gf, 1);

    // Reference path: two independent single-slot run_prefills.
    FP fpRef0(model, &meta, 1024, 1);
    fpRef0.run_prefill(init0, 0, 0, sched);
    std::vector<float> ref0 = fpRef0.run_prefill({X0}, P0, 0, sched);

    FP fpRef1(model, &meta, 1024, 1);
    fpRef1.run_prefill(init1, 0, 0, sched);
    std::vector<float> ref1 = fpRef1.run_prefill({X1}, P1, 0, sched);

    EXPECT_EQ(dec0.size(), static_cast<size_t>(vocab));
    EXPECT_EQ(dec1.size(), static_cast<size_t>(vocab));
    EXPECT_EQ(ref0.size(), static_cast<size_t>(vocab));
    EXPECT_EQ(ref1.size(), static_cast<size_t>(vocab));

    // Fold both slots into one result: mismatches summed, max_abs_diff maxed,
    // token_stable AND-ed.
    DiffResult r;
    auto fold = [&](const std::vector<float>& ref, const std::vector<float>& dec) {
        auto [mm, mad] = bitwise_diff(ref, dec);
        r.mismatches   += mm;
        r.max_abs_diff  = std::max(r.max_abs_diff, mad);
        r.n_logits     += ref.size();
        r.top1_ref      = argmax(ref);
        r.top1_dec      = argmax(dec);
        r.token_stable  = r.token_stable && (r.top1_ref == r.top1_dec);
    };
    fold(ref0, dec0);
    fold(ref1, dec1);
    return r;
}

// ── Gates ────────────────────────────────────────────────────────────────────

static void expect_tier1_bitwise(const DiffResult& r, const char* recipe) {
    EXPECT_EQ(r.mismatches, 0u)
        << recipe << " Tier-1 (single-slot, N=1): the unified decode graph is "
        << "NOT bitwise-identical to single-token run_prefill at the same batch "
        << "shape. mismatched=" << r.mismatches << "/" << r.n_logits
        << " max_abs_diff=" << r.max_abs_diff
        << " token_stable=" << (r.token_stable ? "true" : "false")
        << " (top1_ref=" << r.top1_ref << " top1_dec=" << r.top1_dec << "). "
        << "Same batch shape should select the same Metal kernels; a nonzero "
        << "mismatch means scatter/gather decode diverges from direct-write "
        << "prefill even at N=1. Per docs/plan-gemma-batched-decode.md this is "
        << "the Phase-1 gate — investigate before demoting to DISABLED.";
}

static void expect_token_stable(const DiffResult& r, const char* recipe,
                                const char* tier) {
    EXPECT_TRUE(r.token_stable)
        << recipe << " " << tier << ": the unified decode graph flipped the "
        << "greedily-sampled token vs single-token run_prefill (top1_ref="
        << r.top1_ref << " top1_dec=" << r.top1_dec << "). The decode graph is "
        << "not equivalent to the run_prefill oracle — a real decode-path "
        << "defect (wrong positions, KV gather, embed scale, or final norm).";
    EXPECT_LT(r.max_abs_diff, kDecodeMaxAbsDiff)
        << recipe << " " << tier << ": diverged by max_abs_diff="
        << r.max_abs_diff << " — past the coarse gross-regression ceiling ("
        << kDecodeMaxAbsDiff << "). Not Metal batch-shape low-bit noise; a "
        << "decode-path regression.";
}

static void expect_tier2_bitwise(const DiffResult& r, const char* recipe) {
    EXPECT_EQ(r.mismatches, 0u)
        << "NOISE-FLOOR MEASUREMENT (docs/plan-gemma-batched-decode.md, "
        << "standing Metal batch-shape fork): " << recipe << " Tier-2 batched "
        << "decode (N=2) is NOT bitwise-reproducible vs N independent N=1 "
        << "run_prefills. mismatched=" << r.mismatches << "/" << r.n_logits
        << " max_abs_diff=" << r.max_abs_diff << ". Different batch shape ⇒ "
        << "different Metal kernels; the shipping contract is token-stable, "
        << "this strict variant is a diagnostic, not a gate.";
}

// ── Per-recipe fixtures: one model load per suite, self-skipping ─────────────

#define GEMMA_BATCHED_DECODE_SUITE(SUITE, FP, ENVVAR, DEFAULT_PATH)          \
    class SUITE : public ::testing::Test {                                  \
    protected:                                                              \
        static void SetUpTestSuite() {                                      \
            const std::string p = env_or(ENVVAR, DEFAULT_PATH);             \
            if (!file_exists(p)) return;                                    \
            register_builtin_models();                                      \
            model_ = std::make_unique<Model>();                            \
            model_->load_metadata(p);                                       \
            model_->load_tensors();                                         \
        }                                                                   \
        static void TearDownTestSuite() { model_.reset(); }                 \
        static std::unique_ptr<Model> model_;                               \
    };                                                                      \
    std::unique_ptr<Model> SUITE::model_ = nullptr;                         \
                                                                            \
    /* Tier 1 — single-slot, bitwise (the Phase-1 gate). */                 \
    TEST_F(SUITE, Tier1SingleSlotBitwise) {                                 \
        if (!model_) GTEST_SKIP() << ENVVAR " model not found — skipping";  \
        expect_tier1_bitwise(run_tier1_single_slot<FP>(*model_), #SUITE);   \
    }                                                                       \
    /* Tier 1 — single-slot, token-stable (always-on guard). */            \
    TEST_F(SUITE, Tier1SingleSlotTokenStable) {                             \
        if (!model_) GTEST_SKIP() << ENVVAR " model not found — skipping";  \
        expect_token_stable(run_tier1_single_slot<FP>(*model_), #SUITE,     \
                            "Tier-1");                                       \
    }                                                                       \
    /* Tier 2 — multi-slot, token-stable (proves the batching). */         \
    TEST_F(SUITE, Tier2MultiSlotTokenStable) {                              \
        if (!model_) GTEST_SKIP() << ENVVAR " model not found — skipping";  \
        expect_token_stable(run_tier2_multi_slot<FP>(*model_), #SUITE,      \
                            "Tier-2");                                       \
    }                                                                       \
    /* Tier 2 — multi-slot, strict bitwise (DISABLED noise-floor). */      \
    TEST_F(SUITE, DISABLED_Tier2MultiSlotBitwise) {                         \
        if (!model_) GTEST_SKIP() << ENVVAR " model not found — skipping";  \
        expect_tier2_bitwise(run_tier2_multi_slot<FP>(*model_), #SUITE);    \
    }

// Phase 1: Gemma 1. Gemma 3/4 are added by Phases 3–4 once their
// build_decoding_graph is implemented (today they still throw).
GEMMA_BATCHED_DECODE_SUITE(Gemma1BatchedDecodeTest, Gemma1ForwardPass,
                           "GEMMA1_MODEL_PATH", "./Gemma_2b_it_v1p1.gguf")

// Phase 2: Gemma 2 — interleaved local/global sliding-window mask + attention
// and final logit soft-capping + sandwich norm, all in the batched decode graph.
GEMMA_BATCHED_DECODE_SUITE(Gemma2BatchedDecodeTest, Gemma2ForwardPass,
                           "GEMMA2_MODEL_PATH", "./models/Gemma_2b_it_v2.gguf")

// Phase 3: Gemma 3 — per-layer RoPE base/scale (5:1 local:global), QK-norm,
// no softcap, all in the batched decode graph.
GEMMA_BATCHED_DECODE_SUITE(Gemma3BatchedDecodeTest, Gemma3ForwardPass,
                           "GEMMA3_MODEL_PATH", "./models/gemma-3-1b-it-BF16.gguf")
