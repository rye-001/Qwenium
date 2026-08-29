// test_gemma_feed_tokens.cpp — Phase 3 of docs/plan-feed-tokens.md.
//
// gemma1–4 are attention-only (no recurrent state) — the recurrent
// reproducibility questions are moot, but EACH still owes its own
// KV-append mid-stream differential. "Attention-only so it's fine" is
// explicitly NOT a skip (that reasoning has burned this project before).
//
// One parameterized harness (template on the ForwardPass type), NOT four
// divergent copies. Same split as qwen35/qwen36 per recipe:
//   1. HeadlessGraphHasNoLogits — the one head-guard site prunes "logits".
//   2. MidDecodeDifferentialTokenStable — RUNS: the contract that holds
//      today (token-stable AND within ε). Real KV-append regression guard.
//   3. DISABLED_MidDecodeDifferentialBitwise — strict (a), quarantined.
//
// Each recipe self-skips when its model is absent.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/gemma1.h"
#include "../../src/models/gemma2.h"
#include "../../src/models/gemma3.h"
#include "../../src/models/gemma4.h"

// Gate design (owner decision, docs/plan-feed-tokens.md): token-stable +
// a COARSE universal ceiling. gemma4's parallel dense+MoE FFN diverges
// chunk-vs-sequential by ~0.126 (a top-k expert-selection flip) while
// staying token-stable — falsifying any tight ε sized for recurrent
// low-bit noise (qwen35/36 ≈1e-6). The contract that holds across all 6
// recipes is token-stable; this ceiling is only a gross-regression sanity
// net (wrong positions / garbage state → O(1) divergence AND/OR a token
// flip), not a precision claim. One bound, all recipes — no taxonomy.
static constexpr float kFeedTokensMaxAbsDiff = 1.0f;

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

struct DiffResult {
    size_t mismatches   = 0;
    float  max_abs_diff = 0.0f;
    size_t top1_A       = 0;
    size_t top1_B       = 0;
    bool   token_stable = false;
    size_t n_logits     = 0;
};

// One harness, parameterized on the recipe's ForwardPass type. Feeds a
// 6-token span onto a NON-EMPTY mid-decode cache vs. 6 sequential
// single-token steps from the same starting state (a fresh-from-0 compare
// would prove nothing — mid-stream append is the entire risk surface).
template <typename FP>
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
    const std::vector<int32_t> init  = mk(11, 12);
    const std::vector<int32_t> span  = mk(97,  6);
    const std::vector<int32_t> probe = {123};

    const int span_pos  = static_cast<int>(init.size());
    const int probe_pos = static_cast<int>(init.size() + span.size());

    FP fpA(model, &meta, 1024, 1);
    fpA.run_prefill(init, 0, 0, sched);
    fpA.feed_tokens(span, 0, sched);
    std::vector<float> logits_A = fpA.run_prefill(probe, probe_pos, 0, sched);

    FP fpB(model, &meta, 1024, 1);
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

template <typename FP>
static void headless_graph_has_no_logits(Model& model) {
    const auto& meta = model.get_metadata();
    FP fp(model, &meta, 512, 1);
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8};

    ggml_cgraph* gf_full = fp.build_prefill_graph(tokens, 0, 0, /*want_logits=*/true);
    EXPECT_NE(ggml_graph_get_tensor(gf_full, "logits"), nullptr)
        << "want_logits=true must keep the LM head";

    ggml_cgraph* gf_headless = fp.build_prefill_graph(tokens, 0, 0, /*want_logits=*/false);
    EXPECT_EQ(ggml_graph_get_tensor(gf_headless, "logits"), nullptr)
        << "want_logits=false must prune the LM head (single head-guard site)";
    EXPECT_GT(ggml_graph_n_nodes(gf_headless), 0)
        << "head-less graph must still carry KV-append state-write roots";
    EXPECT_TRUE(fp.feed_tokens_supported())
        << "Phase 3 — recipe must report feed_tokens supported";
}

static void expect_token_stable(const DiffResult& r, const char* recipe) {
    EXPECT_TRUE(r.token_stable)
        << recipe << ": feed_tokens(span) flipped the greedily-sampled token "
        << "vs N×single-step decode (top1_A=" << r.top1_A
        << " top1_B=" << r.top1_B << "). The token-stable contract (option b,"
        << " the one the consumer table needs) is broken — a KV-append "
        << "state-advance regression, not the deferred low-bit fork.";
    EXPECT_LT(r.max_abs_diff, kFeedTokensMaxAbsDiff)
        << recipe << ": feed_tokens(span) diverged by max_abs_diff="
        << r.max_abs_diff << " — past the coarse gross-regression ceiling ("
        << kFeedTokensMaxAbsDiff << "). This is not MoE expert-flip / "
        << "recurrent FP noise; it's a state-advance regression (e.g. wrong "
        << "positions or corrupted state).";
}

static void expect_bitwise(const DiffResult& r, const char* recipe) {
    EXPECT_EQ(r.mismatches, 0u)
        << "DEFERRED FORK (docs/plan-feed-tokens.md): " << recipe
        << " feed_tokens(span) NOT bitwise-reproducible vs N×single-step. "
        << "mismatched=" << r.mismatches << "/" << r.n_logits
        << " max_abs_diff=" << r.max_abs_diff
        << " token_stable=" << (r.token_stable ? "true" : "false")
        << " (top1_A=" << r.top1_A << " top1_B=" << r.top1_B << "). "
        << "Strict option (a) unmet; fork consciously deferred — do not "
        << "default by re-enabling without an owner decision.";
}

// ── Per-recipe fixtures: one model load per suite, self-skipping ─────────────

#define GEMMA_FEED_TOKENS_SUITE(SUITE, FP, ENVVAR, DEFAULT_PATH)             \
    class SUITE : public ::testing::Test {                                  \
    protected:                                                              \
        static void SetUpTestSuite() {                                      \
            const std::string p = env_or(ENVVAR, DEFAULT_PATH);             \
            if (!file_exists(p)) return;                                    \
            register_builtin_models();                                      \
            model_ = std::make_unique<Model>();                             \
            model_->load_metadata(p);                                       \
            model_->load_tensors();                                         \
        }                                                                   \
        static void TearDownTestSuite() { model_.reset(); }                 \
        static std::unique_ptr<Model> model_;                               \
    };                                                                      \
    std::unique_ptr<Model> SUITE::model_ = nullptr;                         \
                                                                            \
    TEST_F(SUITE, HeadlessGraphHasNoLogits) {                               \
        if (!model_) GTEST_SKIP() << ENVVAR " model not found — skipping";  \
        headless_graph_has_no_logits<FP>(*model_);                          \
    }                                                                       \
    TEST_F(SUITE, MidDecodeDifferentialTokenStable) {                       \
        if (!model_) GTEST_SKIP() << ENVVAR " model not found — skipping";  \
        expect_token_stable(run_mid_decode_diff<FP>(*model_), #SUITE);      \
    }                                                                       \
    TEST_F(SUITE, DISABLED_MidDecodeDifferentialBitwise) {                  \
        if (!model_) GTEST_SKIP() << ENVVAR " model not found — skipping";  \
        expect_bitwise(run_mid_decode_diff<FP>(*model_), #SUITE);           \
    }

GEMMA_FEED_TOKENS_SUITE(Gemma1FeedTokensTest, Gemma1ForwardPass,
                        "GEMMA1_MODEL_PATH", "./Gemma_2b_it_v1p1.gguf")
GEMMA_FEED_TOKENS_SUITE(Gemma2FeedTokensTest, Gemma2ForwardPass,
                        "GEMMA2_MODEL_PATH", "./Gemma_2b_it_v2.gguf")
GEMMA_FEED_TOKENS_SUITE(Gemma3FeedTokensTest, Gemma3ForwardPass,
                        "GEMMA3_MODEL_PATH", "./gemma-3-1b-it-BF16.gguf")
GEMMA_FEED_TOKENS_SUITE(Gemma4FeedTokensTest, Gemma4ForwardPass,
                        "GEMMA4_MODEL_PATH", "./gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf")
