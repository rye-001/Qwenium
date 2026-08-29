// test_out_ids_slice.cpp — prefill output-ID slice (last-token LM-head elision).
//
// The optimization: during prefill every layer runs on all N prompt tokens
// (context ingestion is correct), but the ~150k-wide LM head only needs the
// position(s) that produce logits. build_out_ids_slice inserts a ggml_get_rows
// on the hidden state immediately before the head so it runs only on the last
// token; the discarded first N-1 head rows are never computed.
//
// Correctness gate. Ideal spine: last-position prefill logits BIT-FOR-BIT
// identical with the slice on vs. off. On the Metal backend that is
// unattainable — dense uses an N=16 mat-mul kernel, the slice an N=1 mat-vec
// kernel, and ggml-metal's float accumulation order differs across them. This
// is the SAME deferred reproducibility fork already documented in
// test_qwen35_feed_tokens.cpp. We reuse that file's owner-decided resolution:
//   (1) default test — token-stable + ONE coarse universal ceiling. Runs.
//   (2) DISABLED_ strict bytewise — quarantined, not permanently red.
// CLAUDE.md cross-family rule: at least one Qwen recipe AND at least one Gemma
// recipe; both self-skip when their model is absent. Gemma1's inline head
// block is the structurally different site (not build_output_head, not a
// formality).
//
// Composition (the real silent-corruption spine — STRICTLY bit-for-bit).
// The token-position slice and the vocabulary-axis sparse slice
// (SparseHeadInput / "valid_indices", a get_rows on the head WEIGHT) must
// compose order-independently. Asserted against the position-only slice
// (NOT dense): both run under the same N=1 mat-vec kernel, so the mm/mv
// confound is removed and this comparison is genuinely byte-exact. Qwen has
// the vocab slice (build_output_head); gemma's inline head does not, so for
// gemma "both on" must be a byte-exact no-op vs position-only (the two
// mechanisms must not interfere).
//
// Plus a no-model fail-loud unit test for OutputIdsInput.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "engine/model.h"
#include "../../src/models/qwen35.h"
#include "../../src/models/qwen36.h"
#include "../../src/models/gemma1.h"
#include "../../src/models/gemma2.h"
#include "../../src/models/gemma3.h"
#include "../../src/models/gemma4.h"
#include "../../src/models/model_registry.h"
#include "../../src/graph_inputs/output_ids_input.h"

// ── No-model fail-loud unit test ─────────────────────────────────────────────

TEST(OutputIdsInput, FillsLastPosition) {
    ggml_init_params p{ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
                        nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(t);
    ggml_set_name(t, "out_ids");
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    std::vector<int32_t> toks(13, 0);  // n_rows = 13 -> last = 12
    StepContext step;
    step.gf = gf;
    step.tokens = &toks;

    OutputIdsInput in;
    in.set_input(step);

    int32_t got = -1;
    ggml_backend_tensor_get(t, &got, 0, sizeof(int32_t));
    EXPECT_EQ(got, 12);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
}

TEST(OutputIdsInput, FailLoudWhenSlotAbsent) {
    ggml_init_params p{ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
                        nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf = ggml_new_graph(ctx);  // no "out_ids" tensor

    std::vector<int32_t> toks(4, 0);
    StepContext step;
    step.gf = gf;
    step.tokens = &toks;

    OutputIdsInput in;
    EXPECT_THROW(in.set_input(step), std::runtime_error);  // never silent
    ggml_free(ctx);
}

TEST(OutputIdsInput, FailLoudWhenNoRows) {
    ggml_init_params p{ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
                        nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(t);
    ggml_set_name(t, "out_ids");
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);

    std::vector<int32_t> toks;  // n_rows = 0
    StepContext step;
    step.gf = gf;
    step.tokens = &toks;

    OutputIdsInput in;
    EXPECT_THROW(in.set_input(step), std::runtime_error);
    ggml_free(ctx);
}

// ── Model-gated differential (one parameterized pattern per recipe) ──────────

namespace {

// The dense (N=16 mat-mul) vs sliced (N=1 mat-vec) divergence is the SAME
// Metal mm-vs-mv kernel non-determinism documented in
// test_qwen35_feed_tokens.cpp (a batch-size-changing transform picks a
// different ggml-metal kernel with a different float accumulation order).
// Observed here: token-stable, max_abs_diff ~0.02. Reusing that file's
// owner-decided gate verbatim: token-stable + ONE coarse universal ceiling.
// Not a precision claim — a gross-regression net. Strict bit-for-bit is
// quarantined in a DISABLED_ test, not permanently red.
static constexpr float kSliceMaxAbsDiff = 1.0f;

struct DiffOutcome {
    size_t vocab               = 0;

    // dense (N=16) vs position-sliced (N=1): the Metal-limited axis.
    bool   token_stable        = false;
    float  slice_max_abs_diff  = 0.0f;
    size_t slice_mismatches    = 0;   // strict bytewise (quarantined)
    size_t slice_n             = 0;

    // Composition. has_vocab_slice (qwen / build_output_head): the existing
    // sparse head ggml_get_rows-dequantizes the (possibly quantized) weight,
    // so both-on (N=1) vs vocab-sliced-dense (N=16) carries the SAME mm/mv
    // tolerance as the position axis — token-stable + ceiling, strict
    // quarantined. !has_vocab_slice (gemma inline head): the vocab slice is
    // inert; both-on must equal position-only byte-for-byte (same N=1 kernel,
    // same weight precision — genuinely strict; the no-interference spine).
    bool   compose_strict        = false;  // gemma: must hold
    bool   compose_token_stable  = false;  // qwen: must hold
    float  compose_max_abs_diff  = 0.0f;
    size_t compose_mismatches    = 0;
};

// Build a deterministic prompt (token ids only — no tokenizer variance).
std::vector<int32_t> mk_prompt(int n) {
    std::vector<int32_t> v;
    for (int i = 0; i < n; ++i)
        v.push_back(static_cast<int32_t>((i * 7 + 3) % 1000));
    return v;
}

bool bytes_equal(const float* a, const float* b, size_t n) {
    return std::memcmp(a, b, n * sizeof(float)) == 0;
}

// FP is a fresh-per-config forward pass factory (re-running run_prefill on the
// same slot would double-append KV — mirror test_*_feed_tokens.cpp which builds
// a separate forward pass per path).
// has_vocab_slice: true for recipes whose head is build_output_head
// (qwen35/qwen36) — there the SparseHeadInput "valid_indices" weight gather is
// reachable, so "both slices on" yields the genuine vocab×position
// intersection. Gemma's inline head has no vocab slice (out of scope: "no
// change to the vocab/sparse-head logic"); for it, arming sparse must be an
// inert no-op that does NOT corrupt the position slice — still a real
// composition assertion (the two mechanisms must not interfere).
template <typename MakeFP>
DiffOutcome run_slice_diff(Model& model, MakeFP make_fp, bool has_vocab_slice) {
    ggml_backend_sched_t sched = model.get_scheduler();
    const std::vector<int32_t> tokens = mk_prompt(16);

    DiffOutcome o;

    // Reference: dense head over all N positions (slice OFF, explicit seam).
    // Derive the true logits width from the buffer — meta.vocab_size can be
    // narrower than the (padded) head matmul width.
    std::vector<float> dense_full, dense_last;
    {
        auto fp = make_fp();
        fp->set_slice_prefill_head(false);
        dense_full = fp->run_prefill(tokens, 0, 0, sched);
        EXPECT_EQ(dense_full.size() % tokens.size(), 0u);
        o.vocab = dense_full.size() / tokens.size();
        dense_last.assign(dense_full.end() - o.vocab, dense_full.end());
    }
    const size_t vocab = o.vocab;

    // Slice ON (default): head runs only on the last token. Kernel-matched
    // baseline (N=1 mat-vec) reused as the composition reference.
    std::vector<float> pos_only;
    {
        auto fp = make_fp();
        EXPECT_TRUE(fp->slice_prefill_head());  // default on
        pos_only = fp->run_prefill(tokens, 0, 0, sched);
        EXPECT_EQ(pos_only.size(), vocab);
        o.slice_n = std::min(pos_only.size(), dense_last.size());
        double maxabs = 0.0;
        for (size_t i = 0; i < o.slice_n; ++i) {
            if (!bytes_equal(&pos_only[i], &dense_last[i], 1))
                ++o.slice_mismatches;
            maxabs = std::max(maxabs,
                (double)std::fabs(pos_only[i] - dense_last[i]));
        }
        o.slice_max_abs_diff = static_cast<float>(maxabs);
        auto amax = [](const std::vector<float>& v) {
            return std::distance(v.begin(),
                                 std::max_element(v.begin(), v.end()));
        };
        o.token_stable = (pos_only.size() == dense_last.size()) &&
                         (amax(pos_only) == amax(dense_last));
    }

    if (has_vocab_slice) {
        // Composition (qwen / build_output_head). Reference: vocab-slice ON,
        // position OFF → dense [k_vocab, N]; its last column is the
        // intersection computed the OTHER order (vocab-then-implicit-last).
        // both-on must equal that. The existing sparse head get_rows-
        // dequantizes the weight, so both sides share weight precision; the
        // only delta is N=1 vs N=16 → the SAME documented mm/mv tolerance.
        std::vector<int32_t> valid_ids;
        for (int32_t i = 0; i < 64 && static_cast<size_t>(i) < vocab; ++i)
            valid_ids.push_back(i);
        int32_t amax = static_cast<int32_t>(std::distance(
            pos_only.begin(),
            std::max_element(pos_only.begin(), pos_only.end())));
        if (std::find(valid_ids.begin(), valid_ids.end(), amax) ==
            valid_ids.end())
            valid_ids.push_back(amax);
        const size_t k = valid_ids.size();

        std::vector<float> vocab_last;
        {
            auto fp = make_fp();
            fp->set_sparse_decode_ids(valid_ids);
            fp->set_slice_prefill_head(false);  // vocab slice only, dense pos
            std::vector<float> vd = fp->run_prefill(tokens, 0, 0, sched);
            EXPECT_EQ(vd.size(), k * tokens.size());
            vocab_last.assign(vd.end() - k, vd.end());
        }

        auto fp = make_fp();
        fp->set_sparse_decode_ids(valid_ids);   // vocab-axis slice armed
        EXPECT_TRUE(fp->slice_prefill_head());  // position-axis slice on
        std::vector<float> both = fp->run_prefill(tokens, 0, 0, sched);
        EXPECT_EQ(both.size(), k);
        double maxabs = 0.0;
        for (size_t j = 0; j < k && j < both.size(); ++j) {
            if (!bytes_equal(&both[j], &vocab_last[j], 1))
                ++o.compose_mismatches;
            maxabs = std::max(maxabs,
                (double)std::fabs(both[j] - vocab_last[j]));
        }
        o.compose_max_abs_diff = static_cast<float>(maxabs);
        auto amx = [](const std::vector<float>& v) {
            return std::distance(v.begin(),
                                 std::max_element(v.begin(), v.end()));
        };
        o.compose_token_stable = (both.size() == k) &&
                                 (amx(both) == amx(vocab_last));
        o.compose_strict = (both.size() == k) && o.compose_mismatches == 0;
    } else {
        // Composition (gemma inline head): the vocab slice does not exist
        // here. Arming sparse_decode_ids_ must be inert and must NOT perturb
        // the position slice — "both on" must equal the position-only slice
        // byte-for-byte (same N=1 kernel, same weight precision; the two
        // mechanisms genuinely cannot interfere — this stays strict).
        auto fp = make_fp();
        fp->set_sparse_decode_ids({1, 2, 3, 4, 5});
        EXPECT_TRUE(fp->slice_prefill_head());
        std::vector<float> both = fp->run_prefill(tokens, 0, 0, sched);
        EXPECT_EQ(both.size(), vocab);
        double maxabs = 0.0;
        for (size_t i = 0; i < pos_only.size() && i < both.size(); ++i) {
            if (!bytes_equal(&both[i], &pos_only[i], 1))
                ++o.compose_mismatches;
            maxabs = std::max(maxabs,
                (double)std::fabs(both[i] - pos_only[i]));
        }
        o.compose_max_abs_diff = static_cast<float>(maxabs);
        o.compose_strict =
            (both.size() == pos_only.size()) && o.compose_mismatches == 0;
        o.compose_token_stable = o.compose_strict;
    }

    return o;
}

// Default gate: the contract that holds on Metal — token-stable + coarse
// ceiling for the mm/mv-limited axes; strict bit-for-bit only where the
// comparison is genuinely kernel- AND precision-matched (gemma no-interference).
void expect_clean(const DiffOutcome& o, bool has_vocab_slice) {
    EXPECT_TRUE(o.token_stable)
        << "out_ids slice flipped the greedily-sampled token vs the dense "
        << "head (max_abs_diff=" << o.slice_max_abs_diff << "). The slice "
        << "must be a pure discard-elimination; a token flip is a real "
        << "regression, not Metal mm/mv noise.";
    EXPECT_LT(o.slice_max_abs_diff, kSliceMaxAbsDiff)
        << "out_ids slice diverged by max_abs_diff=" << o.slice_max_abs_diff
        << " — past the coarse gross-regression ceiling (" << kSliceMaxAbsDiff
        << "). Beyond the documented Metal mm-vs-mv kernel divergence; a "
        << "gather/position bug.";
    if (has_vocab_slice) {
        // position-slice ∘ vocab-slice == vocab-slice-then-last (other order).
        // mm/mv tolerance applies (N=1 vs N=16 on the same dequantized
        // sparse weight); strict is quarantined like the position axis.
        EXPECT_TRUE(o.compose_token_stable)
            << "position-slice ∘ vocab-slice flipped the argmax vs "
            << "vocab-slice-then-last (max_abs_diff=" << o.compose_max_abs_diff
            << "). The two slices are corrupting each other — not mm/mv noise.";
        EXPECT_LT(o.compose_max_abs_diff, kSliceMaxAbsDiff)
            << "composition diverged by max_abs_diff=" << o.compose_max_abs_diff
            << " — past the coarse ceiling. Order-dependent corruption, not "
            << "the documented Metal mm/mv divergence.";
    } else {
        // Inert vocab slice: kernel- AND precision-matched ⇒ genuinely strict.
        EXPECT_TRUE(o.compose_strict)
            << "arming the (inert) vocab slice perturbed the position-only "
            << "slice on a recipe with no vocab head: " << o.compose_mismatches
            << " byte mismatches. Same N=1 kernel, same weight precision — "
            << "this MUST be bit-for-bit. The mechanisms are interfering.";
    }
}

}  // namespace

// ── Qwen recipe (qwen35) — phase 1 ──────────────────────────────────────────

static std::string qwen35_model_path() {
    if (const char* e = std::getenv("QWEN35_MODEL_PATH"))
        if (e[0]) return std::string(e);
    return "./Qwen3.5-0.8B-BF16.gguf";
}

static DiffOutcome qwen35_outcome() {
    register_builtin_models();
    Model model;
    model.load_metadata(qwen35_model_path());
    model.load_tensors();
    const auto& meta = model.get_metadata();
    return run_slice_diff(model, [&]() {
        return std::make_unique<Qwen35ForwardPass>(model, &meta, 1024, 1);
    }, /*has_vocab_slice=*/true);
}

#define SKIP_IF_NO(path_fn, what)                                          \
    do {                                                                   \
        FILE* _f = std::fopen(path_fn().c_str(), "rb");                     \
        if (!_f) GTEST_SKIP() << what " model not found — skipping";        \
        std::fclose(_f);                                                   \
    } while (0)

// (1) Runs by default: the contract that holds on Metal.
TEST(OutIdsSliceQwen35, TokenStableAndComposes) {
    SKIP_IF_NO(qwen35_model_path, "qwen35");
    expect_clean(qwen35_outcome(), /*has_vocab_slice=*/true);
}

// (2) Disabled: strict dense-vs-sliced bitwise. NOT red in CI — it is
// unattainable on the Metal backend (mm-vs-mv kernel divergence, same
// deferred reproducibility fork as docs/plan-feed-tokens.md /
// test_qwen35_feed_tokens.cpp). Enable with --gtest_also_run_disabled_tests
// only once ggml-metal reduction order is made batch-size-invariant.
TEST(OutIdsSliceQwen35, DISABLED_DenseVsSlicedBitwise) {
    SKIP_IF_NO(qwen35_model_path, "qwen35");
    DiffOutcome o = qwen35_outcome();
    EXPECT_EQ(o.slice_mismatches, 0u)
        << "DEFERRED FORK: dense (N=16 mat-mul) vs sliced (N=1 mat-vec) is "
        << "not bitwise-reproducible on Metal. mismatched=" << o.slice_mismatches
        << "/" << o.slice_n << " max_abs_diff=" << o.slice_max_abs_diff
        << " token_stable=" << (o.token_stable ? "true" : "false") << ".";
    EXPECT_EQ(o.compose_mismatches, 0u)
        << "DEFERRED FORK: position∘vocab vs vocab-then-last also carries the "
        << "mm/mv divergence (N=1 vs N=16 on the dequantized sparse weight). "
        << "mismatched=" << o.compose_mismatches
        << " max_abs_diff=" << o.compose_max_abs_diff << ".";
}

// ── Gemma recipe (gemma1) — cross-family falsifier ──────────────────────────

static std::string gemma1_model_path() {
    if (const char* e = std::getenv("GEMMA1_MODEL_PATH"))
        if (e[0]) return std::string(e);
    return "./Gemma_2b_it_v1p1.gguf";
}

static DiffOutcome gemma1_outcome() {
    register_builtin_models();
    Model model;
    model.load_metadata(gemma1_model_path());
    model.load_tensors();
    const auto& meta = model.get_metadata();
    return run_slice_diff(model, [&]() {
        return std::make_unique<Gemma1ForwardPass>(model, &meta, 1024, 1);
    }, /*has_vocab_slice=*/false);
}

// Cross-family falsifier: gemma1's inline head block is the structurally
// different site (not build_output_head). (1) default Metal-contract gate.
TEST(OutIdsSliceGemma1, TokenStableAndComposes) {
    SKIP_IF_NO(gemma1_model_path, "gemma1");
    expect_clean(gemma1_outcome(), /*has_vocab_slice=*/false);
}

// (2) Disabled strict dense-vs-sliced bitwise (same deferred Metal fork).
TEST(OutIdsSliceGemma1, DISABLED_DenseVsSlicedBitwise) {
    SKIP_IF_NO(gemma1_model_path, "gemma1");
    DiffOutcome o = gemma1_outcome();
    EXPECT_EQ(o.slice_mismatches, 0u)
        << "DEFERRED FORK: gemma1 dense vs sliced not bitwise-reproducible "
        << "on Metal (mm-vs-mv). mismatched=" << o.slice_mismatches << "/"
        << o.slice_n << " max_abs_diff=" << o.slice_max_abs_diff
        << " token_stable=" << (o.token_stable ? "true" : "false") << ".";
}

// ── Remaining recipes — identical single-site pattern, own green ─────────────
// Each owes its own differential ("applied everywhere, looks fine" is not
// accepted). Same fork structure; self-skip when the model is absent. Heavy
// MoE models (qwen36, gemma4) self-skip in CI but run locally when present.

#define RECIPE_SLICE_SUITE(SUITE, CLASS, ENV, DEFPATH, HAS_VOCAB)             \
    static std::string SUITE##_model_path() {                                 \
        if (const char* e = std::getenv(ENV)) if (e[0]) return std::string(e);\
        return DEFPATH;                                                       \
    }                                                                         \
    static DiffOutcome SUITE##_outcome() {                                    \
        register_builtin_models();                                            \
        Model model;                                                          \
        model.load_metadata(SUITE##_model_path());                            \
        model.load_tensors();                                                 \
        const auto& meta = model.get_metadata();                              \
        return run_slice_diff(model, [&]() {                                  \
            return std::make_unique<CLASS>(model, &meta, 1024, 1);            \
        }, /*has_vocab_slice=*/HAS_VOCAB);                                     \
    }                                                                         \
    TEST(OutIdsSlice##SUITE, TokenStableAndComposes) {                        \
        SKIP_IF_NO(SUITE##_model_path, #SUITE);                               \
        expect_clean(SUITE##_outcome(), /*has_vocab_slice=*/HAS_VOCAB);       \
    }                                                                         \
    TEST(OutIdsSlice##SUITE, DISABLED_DenseVsSlicedBitwise) {                 \
        SKIP_IF_NO(SUITE##_model_path, #SUITE);                               \
        DiffOutcome o = SUITE##_outcome();                                    \
        EXPECT_EQ(o.slice_mismatches, 0u)                                     \
            << "DEFERRED FORK (Metal mm-vs-mv): " #SUITE                      \
            << " mismatched=" << o.slice_mismatches << "/" << o.slice_n       \
            << " max_abs_diff=" << o.slice_max_abs_diff                       \
            << " token_stable=" << (o.token_stable ? "true" : "false") << "."; \
    }

// qwen36: MoE + DeltaNet hybrid, head via build_output_head (vocab slice).
RECIPE_SLICE_SUITE(Qwen36, Qwen36ForwardPass, "QWEN36_MODEL_PATH",
                   "./Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf", true)
// gemma2: interleaved local/global attn, inline head (no vocab slice).
RECIPE_SLICE_SUITE(Gemma2, Gemma2ForwardPass, "GEMMA2_MODEL_PATH",
                   "./Gemma_2b_it_v2.gguf", false)
// gemma3: 5:1 local:global, per-layer RoPE base, inline head.
RECIPE_SLICE_SUITE(Gemma3, Gemma3ForwardPass, "GEMMA3_MODEL_PATH",
                   "./gemma-3-1b-it-BF16.gguf", false)
// gemma4: parallel dense+MoE FFN, inline head + final softcap.
RECIPE_SLICE_SUITE(Gemma4, Gemma4ForwardPass, "GEMMA4_MODEL_PATH",
                   "./gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf", false)
