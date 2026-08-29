// test_decode_kv_bucket.cpp — P2 gate of docs/plan-persistent-decode-graph.md.
//
// THE DIFFERENTIAL: decode with exact n_kv sizing (set_decode_kv_bucket(0),
// the pre-P2 shape) vs bucketed sizing (8 and the production 256). Bucketing
// pads the KV read width; padded columns are −inf-masked (verified: the mask
// is exactly −inf on the pad) and read zero-initialized rows, so they add
// nothing to softmax.
//
// EXPECTED RESULT — token-stable, NOT bitwise (the standing shape-change fork).
// The padding is provably inert (bucket-8 and bucket-256 diverge from exact by
// the *identical* amount — extra pad rows contribute zero), yet the REAL rows
// round differently: widening n_kv past the exact count re-blocks the softmax /
// scores·V reduction (CPU SIMD lane grouping; Metal kernel tiling). n_kv=14 is
// not a SIMD/tile multiple, 16 and 256 both are, so 16≡256≠14. This is the
// same class as the Metal matmul batch-shape fork (docs/architecture.md §11,
// project_metal_mm_mv_fork): a transform that changes a reduction axis' extent
// cannot promise bit-identity. So the gate is TOKEN-STABLE + a loose ceiling,
// with the strict bitwise variant kept DISABLED as a noise-floor measurement.
//
// Cross-family: Qwen3.5 (gated, hybrid) + Gemma 3 (plain, sliding-window).
// Both write modes covered so the fork is proven bucketing-intrinsic, not a
// set_rows artifact. Model legs self-skip when the GGUF is absent.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/qwen35.h"
#include "../../src/models/gemma3.h"

// Loose gross-regression ceiling, same spirit/value as the batched-decode and
// feed_tokens harnesses. NOT a precision claim: a real leak (wrong rows, unmasked
// pad, corrupt gather) blows past this AND flips tokens.
static constexpr float kBucketMaxAbsDiff = 0.5f;

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
static std::vector<int32_t> mk_tokens(int base, int n) {
    std::vector<int32_t> v;
    for (int i = 0; i < n; ++i)
        v.push_back(static_cast<int32_t>((base + i * 7 + 3) % 1000));
    return v;
}
static size_t argmax(const std::vector<float>& v) {
    return static_cast<size_t>(
        std::distance(v.begin(), std::max_element(v.begin(), v.end())));
}

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

// A bucketed-vs-exact argmax flip is legitimate ONLY at a genuine tie — when
// the exact top1/top2 logits are within this gap, either token is an equally
// valid greedy pick and reduction reordering may choose the other. Flips at a
// gap WIDER than this are real divergence and fail the gate. Sized well above
// the observed per-step reduction noise (CPU worst ~0.28) yet far below a
// meaningful logit separation.
static constexpr float kTieGap = 0.35f;

static float top1_top2_gap(const std::vector<float>& v) {
    float m1 = -INFINITY, m2 = -INFINITY;
    for (float x : v) {
        if (x > m1) { m2 = m1; m1 = x; }
        else if (x > m2) { m2 = x; }
    }
    return m1 - m2;
}

struct BucketDiff {
    bool   token_stable = true;   // argmax match every step EXCEPT genuine ties
    size_t tie_flips = 0;         // flips that occurred within kTieGap (allowed)
    float  worst_flip_gap = 0.0f; // widest exact gap at which a flip occurred
    float  max_abs_diff = 0.0f;   // worst |exact-bucketed| logit over all steps
    size_t bit_mismatches = 0;    // raw-bit differing logits (for the DISABLED note)
    size_t n_logits = 0;
    bool   ran = false;
};

// Decode `feeds` on three forward passes (exact / bucket 8 / bucket 256) fed
// identically; collect token-stability + max diff of each bucketed leg vs exact.
template <typename FP>
static BucketDiff measure(const std::string& path,
                          ForwardPassBase::KvWriteMode mode) {
    BucketDiff r;
    if (!file_exists(path)) return r;

    register_builtin_models();
    Model model;
    model.load_metadata(path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    ggml_backend_sched_t sched = model.get_scheduler();

    const std::vector<int32_t> init  = mk_tokens(11, 12);
    const std::vector<int32_t> feeds = mk_tokens(500, 40);  // long run — bound check
    const int P = static_cast<int>(init.size());

    FP fpExact(model, &meta, 1024, 1);
    fpExact.set_kv_write_mode(mode);  fpExact.set_decode_kv_bucket(0);
    fpExact.run_prefill(init, 0, 0, sched);
    FP fpB8(model, &meta, 1024, 1);
    fpB8.set_kv_write_mode(mode);     fpB8.set_decode_kv_bucket(8);
    fpB8.run_prefill(init, 0, 0, sched);
    FP fpB256(model, &meta, 1024, 1);
    fpB256.set_kv_write_mode(mode);   fpB256.set_decode_kv_bucket(256);
    fpB256.run_prefill(init, 0, 0, sched);

    r.ran = true;
    for (size_t i = 0; i < feeds.size(); ++i) {
        const int pos = P + static_cast<int>(i);
        std::vector<float> ref = decode_one(fpExact, sched, feeds[i], 0, pos);
        std::vector<float> b8  = decode_one(fpB8,   sched, feeds[i], 0, pos);
        std::vector<float> b256= decode_one(fpB256, sched, feeds[i], 0, pos);
        const size_t am  = argmax(ref);
        const float  gap = top1_top2_gap(ref);
        for (const auto* got : {&b8, &b256}) {
            r.n_logits += ref.size();
            if (argmax(*got) != am) {
                if (gap < kTieGap) {                 // legitimate tie flip
                    ++r.tie_flips;
                    if (gap > r.worst_flip_gap) r.worst_flip_gap = gap;
                } else {                             // real divergence
                    r.token_stable = false;
                    if (gap > r.worst_flip_gap) r.worst_flip_gap = gap;
                }
            }
            for (size_t j = 0; j < ref.size(); ++j) {
                if (std::memcmp(&ref[j], &(*got)[j], sizeof(float)) != 0) {
                    ++r.bit_mismatches;
                    float d = std::fabs(ref[j] - (*got)[j]);
                    if (d > r.max_abs_diff) r.max_abs_diff = d;
                }
            }
        }
    }
    return r;
}

static void check(const BucketDiff& r, const char* recipe) {
    if (!r.ran) { GTEST_SKIP() << "model absent for " << recipe; }
    EXPECT_TRUE(r.token_stable)
        << recipe << ": bucketed decode flipped the sampled token vs exact-width "
        << "decode at a gap WIDER than a tie (" << r.worst_flip_gap << " > "
        << kTieGap << ") — that is real divergence, not the reduction fork. "
        << "max_abs_diff=" << r.max_abs_diff;
    EXPECT_LT(r.max_abs_diff, kBucketMaxAbsDiff)
        << recipe << ": bucketed-vs-exact logit gap " << r.max_abs_diff
        << " exceeds the gross-regression ceiling " << kBucketMaxAbsDiff
        << " — too large for reduction reordering; suspect a real leak.";
    // Two signals it IS a fork, not identity: some bits differ every step, and
    // any flips are confined to ties. If bit_mismatch were ever 0, bucketing
    // became free — promote to the DISABLED bitwise gate.
    std::fprintf(stderr,
        "[bucket-fork %s] token_stable=%d tie_flips=%zu worst_flip_gap=%.4g "
        "max_abs_diff=%.6g bit_mismatch=%zu/%zu\n",
        recipe, r.token_stable ? 1 : 0, r.tie_flips, r.worst_flip_gap,
        r.max_abs_diff, r.bit_mismatches, r.n_logits);
}

TEST(DecodeKvBucket, Qwen35_TokenStable_SetRows) {
    check(measure<Qwen35ForwardPass>(
        env_or("QWEN35_MODEL_PATH", "./models/Qwen3.5-0.8B-BF16.gguf"),
        ForwardPassBase::KvWriteMode::SetRows), "Qwen35/SetRows");
}
TEST(DecodeKvBucket, Gemma3_TokenStable_SetRows) {
    check(measure<Gemma3ForwardPass>(
        env_or("GEMMA3_MODEL_PATH", "./models/gemma-3-1b-it-BF16.gguf"),
        ForwardPassBase::KvWriteMode::SetRows), "Gemma3/SetRows");
}
// Proves the fork is bucketing-intrinsic (reproduces under the legacy cpy write),
// not a set_rows artifact.
TEST(DecodeKvBucket, Qwen35_TokenStable_CpyWrite) {
    check(measure<Qwen35ForwardPass>(
        env_or("QWEN35_MODEL_PATH", "./models/Qwen3.5-0.8B-BF16.gguf"),
        ForwardPassBase::KvWriteMode::Cpy), "Qwen35/Cpy");
}
TEST(DecodeKvBucket, Gemma3_TokenStable_CpyWrite) {
    check(measure<Gemma3ForwardPass>(
        env_or("GEMMA3_MODEL_PATH", "./models/gemma-3-1b-it-BF16.gguf"),
        ForwardPassBase::KvWriteMode::Cpy), "Gemma3/Cpy");
}

// Strict bitwise across bucket widths — DISABLED. Documents that widening
// n_kv re-blocks the reduction on both backends (CPU SIMD lanes, Metal tiles),
// so bit-identity is not achievable. Kept as the executable record of the
// decision (docs/architecture.md §11 standing fork); flip to RUNS only if a
// future kernel makes bucketed reduction order width-invariant.
TEST(DecodeKvBucket, DISABLED_Gemma3_BitwiseAcrossBuckets) {
    BucketDiff r = measure<Gemma3ForwardPass>(
        env_or("GEMMA3_MODEL_PATH", "./models/gemma-3-1b-it-BF16.gguf"),
        ForwardPassBase::KvWriteMode::SetRows);
    if (!r.ran) GTEST_SKIP();
    EXPECT_EQ(r.bit_mismatches, 0u);
}
