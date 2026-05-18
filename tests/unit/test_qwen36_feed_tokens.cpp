// test_qwen36_feed_tokens.cpp — Phase 1 of docs/plan-feed-tokens.md.
//
// feed_tokens(span, slot) advances a slot's model state (attention KV-append
// AND DeltaNet conv + recurrent overwrite) over a span of already-known
// tokens WITHOUT building the LM head. qwen36 is the highest-risk recipe
// (hybrid DeltaNet + attention + MoE) and is gated FIRST.
//
// Tests:
//   1. HeadlessGraphHasNoLogits — the one-site head guard actually prunes the
//      "logits" node when want_logits=false, and keeps it when true.
//   2. MidDecodeDifferential (the spine / correctness contract) — feeding a
//      span onto a NON-EMPTY mid-decode cache must leave the slot in a state
//      that produces the SAME next-token logits as feeding those same tokens
//      as N sequential single-token decode steps from the same starting
//      state. A fresh-prefill-from-0 comparison would prove nothing; this
//      starts from a populated cache. This bitwise probe also resolves-or-
//      triggers the reproducibility decision fork (owner decision: defer
//      until measured — this test IS the measurement for qwen36).
//
// Requires a qwen35moe GGUF; self-skips when absent.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "../../src/core/model.h"
#include "../../src/models/qwen36.h"
#include "../../src/models/model_registry.h"

static std::string get_model_path() {
    if (const char* e = std::getenv("QWEN36_MODEL_PATH"))
        if (e[0]) return std::string(e);
    return "./Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf";
}

#define SKIP_IF_NO_MODEL()                                                 \
    do {                                                                   \
        FILE* _f = std::fopen(get_model_path().c_str(), "rb");             \
        if (!_f) GTEST_SKIP() << "qwen35moe model not found — skipping";   \
        std::fclose(_f);                                                   \
    } while (0)

class Qwen36FeedTokensTest : public ::testing::Test {
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
std::unique_ptr<Model> Qwen36FeedTokensTest::model_ = nullptr;

// ── Test 1: head guard prunes the logits node ────────────────────────────────

TEST_F(Qwen36FeedTokensTest, HeadlessGraphHasNoLogits) {
    SKIP_IF_NO_MODEL();
    const auto& meta = model_->get_metadata();
    Qwen36ForwardPass fp(*model_, &meta, 512, 1);

    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8};

    ggml_cgraph* gf_full = fp.build_prefill_graph(tokens, 0, 0, /*want_logits=*/true);
    EXPECT_NE(ggml_graph_get_tensor(gf_full, "logits"), nullptr)
        << "want_logits=true must keep the LM head";

    ggml_cgraph* gf_headless = fp.build_prefill_graph(tokens, 0, 0, /*want_logits=*/false);
    EXPECT_EQ(ggml_graph_get_tensor(gf_headless, "logits"), nullptr)
        << "want_logits=false must prune the LM head (the single head-guard site)";

    // State-write roots survive head pruning: the head-less graph must still
    // have nodes (KV cpy + DeltaNet conv/recurrent cpy are independent roots).
    EXPECT_GT(ggml_graph_n_nodes(gf_headless), 0)
        << "head-less graph must still carry the state-write roots";

    EXPECT_TRUE(fp.feed_tokens_supported())
        << "qwen36 is Phase 1 — must report feed_tokens supported";
}

// ── The correctness contract (spine) ─────────────────────────────────────────
//
// The reproducibility decision fork (docs/plan-feed-tokens.md → open decision
// fork) is, by owner decision, KEPT DEFERRED. Measured on qwen36:
//
//   feed_tokens(6-token span) vs 6 sequential single-token steps onto a
//   non-empty mid-decode cache →
//     option (a) bitwise   : FAILS  (max_abs_diff ≈ 3.34e-6, ~79% of logits)
//     option (b) token-stable: HOLDS (greedy top-1 identical)
//
// To keep the deferred fork a *quarantined documented signal* rather than a
// permanently-red suite that masks future regressions, the spine is split:
//
//   1. MidDecodeDifferentialTokenStable — RUNS, asserts the weaker contract
//      that actually holds today (token-stable AND within ε). This is a real
//      regression guard: if span-advance drifts beyond ε or flips a token,
//      this goes red.
//   2. DISABLED_MidDecodeDifferentialBitwise — the strict (a) contract,
//      gtest-disabled so it is not red by default. Carries the 3.34e-6
//      measurement; run with --gtest_also_run_disabled_tests when the fork
//      is consciously resolved toward (a).

// Gate design (owner decision, docs/plan-feed-tokens.md): token-stable +
// a COARSE universal ceiling — same constant across qwen36/qwen35/gemma.
// qwen36 measures ≈3.34e-6 here, but gemma4 (parallel dense+MoE) diverges
// ~0.126 (expert-selection flip) while staying token-stable, falsifying
// any tight ε. The contract that holds across all 6 recipes is
// token-stable; this ceiling is only a gross-regression sanity net.
static constexpr float kFeedTokensMaxAbsDiff = 1.0f;

// Shared driver: runs Path A (feed_tokens) and Path B (N single steps) from
// the same non-empty mid-decode starting state and reports the metrics both
// split tests judge. A fresh-prefill-from-0 comparison would prove nothing;
// this deliberately starts from a populated cache (the entire risk surface).
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

    // Deterministic token ids (no tokenizer variance), all in-vocab.
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
    Qwen36ForwardPass fpA(model, &meta, 1024, 1);
    fpA.run_prefill(init, 0, 0, sched);
    fpA.feed_tokens(span, 0, sched);
    std::vector<float> logits_A = fpA.run_prefill(probe, probe_pos, 0, sched);

    // Path B: prefill init → N sequential single-token steps → predict probe.
    Qwen36ForwardPass fpB(model, &meta, 1024, 1);
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

// (1) Runs by default. The weaker contract that holds today — a genuine
//     regression guard, NOT a restatement of the deferred fork.
TEST_F(Qwen36FeedTokensTest, MidDecodeDifferentialTokenStable) {
    SKIP_IF_NO_MODEL();
    DiffResult r = run_mid_decode_diff(*model_);

    EXPECT_TRUE(r.token_stable)
        << "feed_tokens(span) flipped the greedily-sampled token vs "
        << "N×single-step decode (top1_A=" << r.top1_A
        << " top1_B=" << r.top1_B << "). Option (b) token-stable no longer "
        << "holds — this is a regression beyond the deferred fork, not the "
        << "documented 3.34e-6 low-bit divergence.";

    EXPECT_LT(r.max_abs_diff, kFeedTokensMaxAbsDiff)
        << "feed_tokens(span) diverged from N×single-step decode by "
        << "max_abs_diff=" << r.max_abs_diff << " — past the coarse "
        << "gross-regression ceiling (" << kFeedTokensMaxAbsDiff << "). Not "
        << "chunked-recurrence FP noise (≈3.34e-6); a state-advance "
        << "regression (e.g. wrong positions / corrupt state).";
}

// (2) Disabled by default: the strict bitwise (a) contract. NOT red in CI.
//     Carries the recorded measurement; enable with
//     --gtest_also_run_disabled_tests once the fork is resolved toward (a).
//     Quarantined documented signal of docs/plan-feed-tokens.md's open fork.
TEST_F(Qwen36FeedTokensTest, DISABLED_MidDecodeDifferentialBitwise) {
    SKIP_IF_NO_MODEL();
    DiffResult r = run_mid_decode_diff(*model_);

    EXPECT_EQ(r.mismatches, 0u)
        << "DEFERRED FORK (docs/plan-feed-tokens.md): feed_tokens(span) is "
        << "NOT bitwise-reproducible vs N×single-step decode on qwen36. "
        << "mismatched logits=" << r.mismatches << "/" << r.n_logits
        << " max_abs_diff=" << r.max_abs_diff
        << " token_stable=" << (r.token_stable ? "true" : "false")
        << " (top1_A=" << r.top1_A << " top1_B=" << r.top1_B << "). "
        << "Recorded measurement ≈3.34e-6 (chunked DeltaNet recurrence FP "
        << "reduction order, not an adapter defect). Option (a) is the "
        << "unmet strict contract; the fork (a/b/c, global vs per-consumer) "
        << "remains consciously deferred — do not default by re-enabling "
        << "this test without an owner decision.";
}
