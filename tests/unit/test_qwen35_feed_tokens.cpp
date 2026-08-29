// test_qwen35_feed_tokens.cpp — Phase 2 of docs/plan-feed-tokens.md.
//
// feed_tokens(span, slot) advances a slot's model state (attention KV-append
// AND DeltaNet/SSM conv + recurrent overwrite) over a span of already-known
// tokens WITHOUT building the LM head. qwen35 is the second-phase recipe
// (attention + SSM/recurrent, lower risk than the qwen36 MoE hybrid); it
// owes its own mid-stream differential — "lower risk" is not a skip.
//
// Mirrors test_qwen36_feed_tokens.cpp exactly (one parameterized pattern,
// not a divergent copy): one head-guard test plus the split spine —
//   1. MidDecodeDifferentialTokenStable — RUNS: the weaker contract that
//      holds today (token-stable AND within ε). A real regression guard.
//   2. DISABLED_MidDecodeDifferentialBitwise — strict (a), gtest-disabled
//      so the deferred reproducibility fork is a quarantined documented
//      signal, not a permanently-red suite.
//
// Requires a qwen35 GGUF; self-skips when absent.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "engine/model.h"
#include "../../src/models/qwen35.h"
#include "../../src/models/model_registry.h"

static std::string get_model_path() {
    if (const char* e = std::getenv("QWEN35_MODEL_PATH"))
        if (e[0]) return std::string(e);
    return "./Qwen3.5-0.8B-BF16.gguf";
}

#define SKIP_IF_NO_MODEL()                                                 \
    do {                                                                   \
        FILE* _f = std::fopen(get_model_path().c_str(), "rb");             \
        if (!_f) GTEST_SKIP() << "qwen35 model not found — skipping";      \
        std::fclose(_f);                                                   \
    } while (0)

class Qwen35FeedTokensTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        FILE* f = std::fopen(get_model_path().c_str(), "rb");
        if (!f) return;
        std::fclose(f);
        register_builtin_models();
        model_ = std::make_unique<Model>();
        model_->load_metadata(get_model_path());
        model_->load_tensors();
    }
    static void TearDownTestSuite() { model_.reset(); }
    static std::unique_ptr<Model> model_;
};
std::unique_ptr<Model> Qwen35FeedTokensTest::model_ = nullptr;

// ── Test 1: head guard prunes the logits node ────────────────────────────────

TEST_F(Qwen35FeedTokensTest, HeadlessGraphHasNoLogits) {
    SKIP_IF_NO_MODEL();
    const auto& meta = model_->get_metadata();
    Qwen35ForwardPass fp(*model_, &meta, 512, 1);

    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8};

    ggml_cgraph* gf_full = fp.build_prefill_graph(tokens, 0, 0, /*want_logits=*/true);
    EXPECT_NE(ggml_graph_get_tensor(gf_full, "logits"), nullptr)
        << "want_logits=true must keep the LM head";

    ggml_cgraph* gf_headless = fp.build_prefill_graph(tokens, 0, 0, /*want_logits=*/false);
    EXPECT_EQ(ggml_graph_get_tensor(gf_headless, "logits"), nullptr)
        << "want_logits=false must prune the LM head (the single head-guard site)";

    EXPECT_GT(ggml_graph_n_nodes(gf_headless), 0)
        << "head-less graph must still carry the state-write roots";

    EXPECT_TRUE(fp.feed_tokens_supported())
        << "qwen35 is Phase 2 — must report feed_tokens supported";
}

// ── The correctness contract (spine) ─────────────────────────────────────────
//
// Same deferred-fork quarantine structure as qwen36 (docs/plan-feed-tokens.md
// → open decision fork). qwen35 owes its own measurement; "attention + SSM,
// lower risk" is not a waiver.

// Gate design (owner decision, docs/plan-feed-tokens.md): token-stable +
// a COARSE universal ceiling — same constant across qwen36/qwen35/gemma.
// gemma4 (parallel dense+MoE) falsified any tight ε by diverging ~0.126
// (expert-selection flip) while staying token-stable. The contract that
// holds across all 6 recipes is token-stable; this ceiling is only a
// gross-regression sanity net, not a precision claim.
static constexpr float kFeedTokensMaxAbsDiff = 1.0f;

struct DiffResult {
    size_t mismatches    = 0;
    float  max_abs_diff  = 0.0f;
    size_t top1_A        = 0;
    size_t top1_B        = 0;
    bool   token_stable  = false;
    size_t n_logits      = 0;
};

static DiffResult run_mid_decode_diff(Model& model) {
    const auto& meta = model.get_metadata();
    const uint32_t vocab = meta.vocab_size;
    ggml_backend_sched_t sched = model.get_scheduler();

    auto mk = [&](int base, int n) {
        std::vector<int32_t> v;
        for (int i = 0; i < n; ++i)
            v.push_back(static_cast<int32_t>((base + i * 7 + 3) % 1000));
        return v;
    };
    const std::vector<int32_t> init  = mk(11, 12);  // establishes non-empty cache
    const std::vector<int32_t> span  = mk(97,  6);  // the fed run
    const std::vector<int32_t> probe = {123};       // next-token prediction

    const int span_pos  = static_cast<int>(init.size());
    const int probe_pos = static_cast<int>(init.size() + span.size());

    // Path A: prefill init → feed_tokens(span) [head-less] → predict probe.
    Qwen35ForwardPass fpA(model, &meta, 1024, 1);
    fpA.run_prefill(init, 0, 0, sched);
    fpA.feed_tokens(span, 0, sched);
    std::vector<float> logits_A = fpA.run_prefill(probe, probe_pos, 0, sched);

    // Path B: prefill init → N sequential single-token steps → predict probe.
    Qwen35ForwardPass fpB(model, &meta, 1024, 1);
    fpB.run_prefill(init, 0, 0, sched);
    for (size_t i = 0; i < span.size(); ++i)
        fpB.run_prefill({span[i]}, span_pos + static_cast<int>(i), 0, sched);
    std::vector<float> logits_B = fpB.run_prefill(probe, probe_pos, 0, sched);

    EXPECT_EQ(logits_A.size(), logits_B.size());
    EXPECT_EQ(logits_A.size(), static_cast<size_t>(vocab) * probe.size());

    DiffResult r;
    r.n_logits = logits_A.size();
    for (size_t i = 0; i < logits_A.size(); ++i) {
        if (std::memcmp(&logits_A[i], &logits_B[i], sizeof(float)) != 0) {
            ++r.mismatches;
            r.max_abs_diff = std::max(r.max_abs_diff,
                                      std::fabs(logits_A[i] - logits_B[i]));
        }
    }
    auto argmax = [](const std::vector<float>& v) {
        return static_cast<size_t>(
            std::distance(v.begin(), std::max_element(v.begin(), v.end())));
    };
    r.top1_A = argmax(logits_A);
    r.top1_B = argmax(logits_B);
    r.token_stable = (r.top1_A == r.top1_B);
    return r;
}

// (1) Runs by default. The weaker contract that holds today.
TEST_F(Qwen35FeedTokensTest, MidDecodeDifferentialTokenStable) {
    SKIP_IF_NO_MODEL();
    DiffResult r = run_mid_decode_diff(*model_);

    EXPECT_TRUE(r.token_stable)
        << "feed_tokens(span) flipped the greedily-sampled token vs "
        << "N×single-step decode (top1_A=" << r.top1_A
        << " top1_B=" << r.top1_B << "). Option (b) token-stable no longer "
        << "holds — a regression beyond the deferred low-bit fork.";

    EXPECT_LT(r.max_abs_diff, kFeedTokensMaxAbsDiff)
        << "feed_tokens(span) diverged by max_abs_diff=" << r.max_abs_diff
        << " — past the coarse gross-regression ceiling ("
        << kFeedTokensMaxAbsDiff << "). Not recurrent FP noise; a "
        << "state-advance regression (e.g. wrong positions / corrupt state).";
}

// (2) Disabled by default: strict bitwise (a). NOT red in CI. Enable with
//     --gtest_also_run_disabled_tests once the fork is resolved toward (a).
TEST_F(Qwen35FeedTokensTest, DISABLED_MidDecodeDifferentialBitwise) {
    SKIP_IF_NO_MODEL();
    DiffResult r = run_mid_decode_diff(*model_);

    EXPECT_EQ(r.mismatches, 0u)
        << "DEFERRED FORK (docs/plan-feed-tokens.md): feed_tokens(span) is "
        << "NOT bitwise-reproducible vs N×single-step decode on qwen35. "
        << "mismatched logits=" << r.mismatches << "/" << r.n_logits
        << " max_abs_diff=" << r.max_abs_diff
        << " token_stable=" << (r.token_stable ? "true" : "false")
        << " (top1_A=" << r.top1_A << " top1_B=" << r.top1_B << "). "
        << "Option (a) is the unmet strict contract; the fork (a/b/c, "
        << "global vs per-consumer) remains consciously deferred — do not "
        << "default by re-enabling this test without an owner decision.";
}
