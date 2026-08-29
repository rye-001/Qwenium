// test_mtp_head.cpp — Phase 3 gate for the qwen36 NextN/MTP head
// (docs/plan-mtp-decode.md §7 Phase 3).
//
// Requires the MTP-converted GGUF (models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf,
// overridable via QWEN36_MTP_MODEL_PATH); skips when absent. Covers:
//   1. capability flag on an MTP GGUF,
//   2. draft shape + in-range tokens + call-to-call determinism,
//   3. acceptance sanity: on a strongly-predictable (repetition/induction)
//      prompt, the head's greedy drafts must agree with the main model's own
//      greedy continuation — a mis-wired head agrees ~never at 248k vocab.

#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "engine/model.h"
#include "../../src/models/qwen36.h"
#include "../../src/models/model_registry.h"
#include "../../src/loader/tokenizer.h"

static std::string get_model_path() {
    if (const char* env = std::getenv("QWEN36_MTP_MODEL_PATH"))
        return env;
    const char* p = "models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf";
    if (FILE* f = std::fopen(p, "rb")) { std::fclose(f); return p; }
    return "";
}

#define SKIP_IF_NO_MODEL()                                                   \
    do {                                                                     \
        if (get_model_path().empty())                                        \
            GTEST_SKIP() << "MTP GGUF not found — skipping";                 \
    } while (0)

class MtpHeadTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (get_model_path().empty()) return;
        register_builtin_models();
        model_ = std::make_unique<Model>();
        model_->load_metadata(get_model_path());
        model_->load_tensors();

        // Dedicated scheduler for the head graphs: a new graph shape must not
        // share galloc state with the main graphs
        // (docs/server-image-multirequest-bug.md precedent).
        if (model_->has_metal_backend()) {
            ggml_backend_t backends[] = {model_->get_backend_metal(),
                                         model_->get_backend_cpu()};
            mtp_sched_ = ggml_backend_sched_new(backends, nullptr, 2,
                                                FP_GRAPH_SIZE, true, false);
        } else {
            ggml_backend_t backends[] = {model_->get_backend_cpu()};
            mtp_sched_ = ggml_backend_sched_new(backends, nullptr, 1,
                                                FP_GRAPH_SIZE, false, false);
        }
        ASSERT_NE(mtp_sched_, nullptr);
    }

    static void TearDownTestSuite() {
        if (mtp_sched_) ggml_backend_sched_free(mtp_sched_);
        mtp_sched_ = nullptr;
        model_.reset();
    }

    static int argmax(const std::vector<float>& v, size_t off, size_t n) {
        size_t best = 0;
        for (size_t i = 1; i < n; ++i)
            if (v[off + i] > v[off + best]) best = i;
        return static_cast<int>(best);
    }

    static std::unique_ptr<Model>  model_;
    static ggml_backend_sched_t    mtp_sched_;
};

std::unique_ptr<Model> MtpHeadTest::model_    = nullptr;
ggml_backend_sched_t   MtpHeadTest::mtp_sched_ = nullptr;

// ── 1. Capability ────────────────────────────────────────────────────────────

TEST_F(MtpHeadTest, MtpSupportedOnMtpGguf) {
    SKIP_IF_NO_MODEL();
    const auto& meta = model_->get_metadata();
    Qwen36ForwardPass fp(*model_, &meta, 512, 1);
    EXPECT_TRUE(fp.mtp_supported());
}

// ── 2. Shape, range, determinism ────────────────────────────────────────────

TEST_F(MtpHeadTest, DraftShapeAndDeterminism) {
    SKIP_IF_NO_MODEL();
    const auto& meta = model_->get_metadata();
    Qwen36ForwardPass fp(*model_, &meta, 512, 1);
    fp.set_output_hidden(true);

    std::vector<int32_t> prompt = {100, 200, 300, 400, 500, 600, 700, 800};
    std::vector<float> logits =
        fp.run_prefill(prompt, 0, 0, model_->get_scheduler());

    // Prefill slices the head to the last position, but hidden_out carries all
    // positions — take the last one.
    // (run_prefill consumed the graph; get_output_hidden must be called on the
    // same graph, so re-run through the manual path.)
    ggml_backend_sched_reset(model_->get_scheduler());
    fp.set_cache_pos(0, 0);
    ggml_cgraph* gf = fp.build_prefill_graph(prompt, 0, 0);
    ASSERT_TRUE(ggml_backend_sched_alloc_graph(model_->get_scheduler(), gf));
    fp.set_prefill_inputs(gf, prompt, 0);
    ggml_backend_sched_graph_compute(model_->get_scheduler(), gf);
    std::vector<float> hidden_all = fp.get_output_hidden(gf);
    fp.advance_cache(prompt.size(), 0);
    ASSERT_EQ(hidden_all.size(), meta.embedding_length * prompt.size());

    std::vector<float> h(
        hidden_all.end() - meta.embedding_length, hidden_all.end());
    const int32_t x = argmax(logits,
        (prompt.size() - 1) * meta.vocab_size, meta.vocab_size);

    auto d1 = fp.mtp_draft(0, h, x, (int)prompt.size(), 3, mtp_sched_);
    auto d2 = fp.mtp_draft(0, h, x, (int)prompt.size(), 3, mtp_sched_);

    ASSERT_EQ(d1.size(), 3u);
    for (int32_t t : d1) {
        EXPECT_GE(t, 0);
        EXPECT_LT(t, (int32_t)meta.vocab_size);
    }
    EXPECT_EQ(d1, d2) << "mtp_draft must be deterministic call-to-call";
}

// ── 3. Acceptance sanity (the mis-wired-head tripwire) ──────────────────────

TEST_F(MtpHeadTest, AcceptanceSanityOnInductionPrompt) {
    SKIP_IF_NO_MODEL();
    const auto& meta = model_->get_metadata();
    Qwen36ForwardPass fp(*model_, &meta, 512, 1);
    fp.set_output_hidden(true);
    ggml_backend_sched_t main_sched = model_->get_scheduler();

    // Strong induction pattern in REAL text (raw token IDs degenerate to EOS —
    // the model treats them as garbage and stops): a repeated sentence whose
    // greedy continuation is maximally predictable.
    Tokenizer* tokenizer = model_->get_tokenizer();
    ASSERT_NE(tokenizer, nullptr);
    std::vector<int32_t> prompt = tokenizer->encode(
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of France is Paris. The capital of Germany is");
    ASSERT_GT(prompt.size(), 8u);

    // Prefill (manual path so hidden_out is readable from the same graph).
    ggml_backend_sched_reset(main_sched);
    ggml_cgraph* gf = fp.build_prefill_graph(prompt, 0, 0);
    ASSERT_TRUE(ggml_backend_sched_alloc_graph(main_sched, gf));
    fp.set_prefill_inputs(gf, prompt, 0);
    ggml_backend_sched_graph_compute(main_sched, gf);
    std::vector<float> logits     = fp.get_output_logits(gf);
    std::vector<float> hidden_all = fp.get_output_hidden(gf);
    fp.advance_cache(prompt.size(), 0);

    std::vector<float> h(
        hidden_all.end() - meta.embedding_length, hidden_all.end());
    // Prefill head is sliced to the last position: logits = [vocab] for it.
    const int32_t x = argmax(logits, logits.size() - meta.vocab_size,
                             meta.vocab_size);

    // Head drafts continuations of x.
    const uint32_t K = 3;
    auto draft = fp.mtp_draft(0, h, x, (int)prompt.size(), K, mtp_sched_);
    ASSERT_EQ(draft.size(), K);

    // Ground truth: the main model's own greedy continuation after x.
    int pos = (int)prompt.size();
    std::vector<int32_t> truth;
    int32_t tok = x;
    for (uint32_t i = 0; i < K; ++i) {
        std::vector<float> lg = fp.run_prefill({tok}, pos, 0, main_sched);
        tok = argmax(lg, lg.size() - meta.vocab_size, meta.vocab_size);
        truth.push_back(tok);
        pos += 1;
    }

    // Sequential acceptance, exactly as SpeculativeDecoder counts it.
    uint32_t accepted = 0;
    for (uint32_t i = 0; i < K; ++i) {
        if (draft[i] != truth[i]) break;
        ++accepted;
    }
    std::cout << "[mtp acceptance-sanity] draft={" << draft[0] << ","
              << draft[1] << "," << draft[2] << "}='"
              << tokenizer->decode(std::vector<int32_t>(draft.begin(), draft.end()))
              << "'  truth={" << truth[0] << "," << truth[1] << "," << truth[2]
              << "}='" << tokenizer->decode(truth) << "'  accepted="
              << accepted << "/" << K << std::endl;

    // A correctly wired head on an induction prompt must get at least the
    // first draft right; a mis-wired head matches ~never at 248k vocab.
    EXPECT_GE(accepted, 1u)
        << "near-zero acceptance — head likely mis-wired (concat order / norm "
           "variant / hidden tap point)";
}
