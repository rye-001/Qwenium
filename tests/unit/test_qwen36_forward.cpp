// test_qwen36_forward.cpp — PR 3.I.2 smoke test for Qwen36ForwardPass.
//
// Runs build_prefill_graph() on a short prompt, allocates, executes, and checks
// that logits are the right shape and free of NaN/Inf. Output correctness is
// deferred to 3.I.3 (logit-comparison test).
//
// Requires: QWEN36_MODEL_PATH env var pointing to a qwen35moe GGUF.

#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#include "../../src/core/model.h"
#include "../../src/models/qwen36.h"

static std::string get_model_path() {
    const char* p = "./Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf";// std::getenv("QWEN36_MODEL_PATH");
    return p ? std::string(p) : "";
}

#define SKIP_IF_NO_MODEL()                                              \
    do {                                                                \
        if (get_model_path().empty())                                   \
            GTEST_SKIP() << "QWEN36_MODEL_PATH not set — skipping";    \
    } while (0)

class Qwen36ForwardTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (get_model_path().empty()) return;
        model_ = std::make_unique<Model>();
        model_->load_metadata(get_model_path());
        model_->load_tensors();
    }

    static void TearDownTestSuite() { model_.reset(); }

    static std::unique_ptr<Model> model_;
};

std::unique_ptr<Model> Qwen36ForwardTest::model_ = nullptr;

// ── Test 1: graph builds and allocates without crashing ───────────────────────

TEST_F(Qwen36ForwardTest, BuildPrefillGraphSucceeds) {
    SKIP_IF_NO_MODEL();

    const auto& meta = model_->get_metadata();
    Qwen36ForwardPass fp(*model_, &meta, 512, 1);

    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8};

    ggml_cgraph* gf = nullptr;
    ASSERT_NO_THROW(gf = fp.build_prefill_graph(tokens, 0, 0));
    ASSERT_NE(gf, nullptr);

    ggml_backend_sched_t sched = model_->get_scheduler();
    ggml_backend_sched_reset(sched);
    bool alloc_ok = ggml_backend_sched_alloc_graph(sched, gf);
    EXPECT_TRUE(alloc_ok) << "Graph allocation failed";
}

// ── Test 2: logits length == vocab_size, no NaN / Inf ────────────────────────

TEST_F(Qwen36ForwardTest, PrefillProducesFiniteLogits) {
    SKIP_IF_NO_MODEL();

    const auto& meta = model_->get_metadata();
    Qwen36ForwardPass fp(*model_, &meta, 512, 1);
    ggml_backend_sched_t sched = model_->get_scheduler();

    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8};

    ggml_backend_sched_reset(sched);
    ggml_cgraph* gf = fp.build_prefill_graph(tokens, 0, 0);
    ASSERT_NE(gf, nullptr);

    ASSERT_TRUE(ggml_backend_sched_alloc_graph(sched, gf)) << "Allocation failed";
    fp.set_inputs(gf, tokens, 0);
    ggml_backend_sched_graph_compute(sched, gf);

    std::vector<float> logits = fp.get_output_logits(gf);

    // Length check: vocab_size × n_tokens (we inspect the last token's slice)
    ASSERT_EQ(logits.size(), meta.vocab_size * tokens.size())
        << "Expected vocab_size=" << meta.vocab_size
        << " × n_tokens=" << tokens.size() << " logits";

    const size_t offset = (tokens.size() - 1) * meta.vocab_size;
    uint32_t nan_count = 0, inf_count = 0;
    for (uint32_t i = 0; i < meta.vocab_size; ++i) {
        float v = logits[offset + i];
        if (std::isnan(v))       ++nan_count;
        else if (!std::isfinite(v)) ++inf_count;
    }

    EXPECT_EQ(nan_count, 0u) << "Found NaN values in logits";
    EXPECT_EQ(inf_count, 0u) << "Found Inf values in logits";
}

// ── Test 3: top-1 token index is in [0, vocab_size) ──────────────────────────

TEST_F(Qwen36ForwardTest, TopTokenIndexInRange) {
    SKIP_IF_NO_MODEL();

    const auto& meta = model_->get_metadata();
    Qwen36ForwardPass fp(*model_, &meta, 512, 1);
    ggml_backend_sched_t sched = model_->get_scheduler();

    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8};

    ggml_backend_sched_reset(sched);
    ggml_cgraph* gf = fp.build_prefill_graph(tokens, 0, 0);
    ASSERT_NE(gf, nullptr);

    ASSERT_TRUE(ggml_backend_sched_alloc_graph(sched, gf));
    fp.set_inputs(gf, tokens, 0);
    ggml_backend_sched_graph_compute(sched, gf);

    std::vector<float> logits = fp.get_output_logits(gf);

    // Inspect last token's logit slice
    const size_t offset = (tokens.size() - 1) * meta.vocab_size;
    auto begin = logits.begin() + static_cast<ptrdiff_t>(offset);
    auto end   = begin + static_cast<ptrdiff_t>(meta.vocab_size);
    auto best  = std::max_element(begin, end);

    const uint32_t top1 = static_cast<uint32_t>(std::distance(begin, best));
    EXPECT_LT(top1, meta.vocab_size)
        << "top-1 token index " << top1 << " out of range [0, " << meta.vocab_size << ")";
}
