// test_decode_graph_cache.cpp — P3 gate of docs/plan-persistent-decode-graph.md.
//
// THE PROPERTY: reusing a built+allocated decode graph across steps is
// BITWISE-neutral — a persistent graph reused for a step must produce the exact
// same logits as the SAME-shape graph rebuilt from scratch that step. Isolates
// the reuse mechanism from the bucketing numeric fork (that fork — bucketed vs
// exact — is gated separately by test_decode_kv_bucket). Both sides here run at
// the same bucket, so same shape ⇒ same kernels ⇒ bit-identity is the gate.
//
// A small bucket (8) makes the ~30-step run cross several bucket boundaries, so
// the cache is exercised on BOTH the reuse fast-path AND the boundary rebuild
// (reuses>0, rebuilds>=2). Cross-family: Qwen3.5 + Gemma3. Self-skips absent.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "../../src/core/model.h"
#include "../../src/core/decode_graph_cache.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/qwen35.h"
#include "../../src/models/gemma3.h"

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

// One rebuild-every-step decode at the fp's current bucket, on the main sched.
template <typename FP>
static std::vector<float> decode_rebuild(FP& fp, ggml_backend_sched_t sched,
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

template <typename FP>
static void run_reuse_neutral(const std::string& path) {
    if (!file_exists(path)) GTEST_SKIP() << "model absent: " << path;

    register_builtin_models();
    Model model;
    model.load_metadata(path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    ggml_backend_sched_t main_sched = model.get_scheduler();

    const std::vector<int32_t> init  = mk_tokens(11, 12);
    const std::vector<int32_t> feeds = mk_tokens(500, 32);
    const int P = static_cast<int>(init.size());
    const uint32_t kTestBucket = 8;  // small ⇒ cross boundaries fast

    // Reference: rebuild-every-step, bucket 8, SetRows write.
    FP fpRebuild(model, &meta, 1024, 1);
    fpRebuild.set_kv_write_mode(ForwardPassBase::KvWriteMode::SetRows);
    fpRebuild.set_decode_kv_bucket(kTestBucket);
    fpRebuild.run_prefill(init, 0, 0, main_sched);

    // Candidate: same config, driven through the persistent cache.
    FP fpCache(model, &meta, 1024, 1);
    fpCache.set_kv_write_mode(ForwardPassBase::KvWriteMode::SetRows);
    fpCache.set_decode_kv_bucket(kTestBucket);
    fpCache.run_prefill(init, 0, 0, main_sched);
    DecodeGraphCache cache(model, &fpCache);

    for (size_t i = 0; i < feeds.size(); ++i) {
        const int pos = P + static_cast<int>(i);
        std::vector<float> ref = decode_rebuild(fpRebuild, main_sched, feeds[i], 0, pos);

        ggml_cgraph* gf = cache.step({feeds[i]}, {0u}, {pos});
        std::vector<float> got = fpCache.get_output_logits(gf);
        fpCache.advance_cache(1, 0);

        ASSERT_EQ(ref.size(), got.size()) << "step " << i;
        size_t mism = 0; float maxd = 0.0f;
        for (size_t j = 0; j < ref.size(); ++j)
            if (std::memcmp(&ref[j], &got[j], sizeof(float)) != 0) {
                ++mism; float d = std::fabs(ref[j]-got[j]); if (d>maxd) maxd=d;
            }
        EXPECT_EQ(mism, 0u)
            << "step " << i << ": reused persistent graph is NOT bitwise-"
            << "identical to the same-shape rebuilt graph. mismatch=" << mism
            << "/" << ref.size() << " max_abs_diff=" << maxd
            << " — graph reuse perturbed the computation (a real defect; the "
            << "bucketing fork is a SEPARATE, same-on-both gate).";
        if (mism) return;
    }

    // The cache must actually have reused (fast path) AND rebuilt at the bucket
    // crossings we forced — otherwise the bitwise result above is vacuous.
    std::fprintf(stderr, "[graph-cache] rebuilds=%llu reuses=%llu\n",
                 (unsigned long long)cache.rebuilds(),
                 (unsigned long long)cache.reuses());
    EXPECT_GT(cache.reuses(), 0u)  << "cache never hit the reuse fast path";
    EXPECT_GE(cache.rebuilds(), 2u) << "cache never crossed a bucket boundary "
                                       "(reuse would be untested against rebuild)";
    EXPECT_EQ(cache.rebuilds() + cache.reuses(), feeds.size());
}

TEST(DecodeGraphCache, Qwen35_ReuseIsBitwiseNeutral) {
    run_reuse_neutral<Qwen35ForwardPass>(
        env_or("QWEN35_MODEL_PATH", "./models/Qwen3.5-0.8B-BF16.gguf"));
}
TEST(DecodeGraphCache, Gemma3_ReuseIsBitwiseNeutral) {
    run_reuse_neutral<Gemma3ForwardPass>(
        env_or("GEMMA3_MODEL_PATH", "./models/gemma-3-1b-it-BF16.gguf"));
}

// enable_persistent_decode refuses a non-persistent-capable recipe fail-loud.
// (No model needed — pure contract check via a recipe that returns false. All
// three converted recipes return true, so we assert the positive path sets the
// {SetRows, 256} config on one when a model is present.)
TEST(DecodeGraphCache, EnablePersistentDecodeConfiguresFp) {
    const std::string path =
        env_or("GEMMA3_MODEL_PATH", "./models/gemma-3-1b-it-BF16.gguf");
    if (!file_exists(path)) GTEST_SKIP() << "model absent: " << path;
    register_builtin_models();
    Model model; model.load_metadata(path); model.load_tensors();
    const auto& meta = model.get_metadata();
    Gemma3ForwardPass fp(model, &meta, 1024, 1);
    ASSERT_TRUE(fp.supports_persistent_decode());
    enable_persistent_decode(&fp);
    EXPECT_EQ(fp.decode_kv_bucket(), DecodeGraphCache::kBucket);
    EXPECT_EQ(fp.kv_write_mode(), ForwardPassBase::KvWriteMode::SetRows);
}
