/**
 * test_qwen35_forward_attn.cpp — Step 6 TDD for Qwen 3.5 support
 * 
 * Tests that the forward pass can prefill a short prompt through
 * the hybrid layer loop (full attention + stubbed SSM) and produce
 * finite logits. Output won't be correct yet (SSM layers are stubs),
 * but the graph builds, allocates, computes, and doesn't crash.
 *
 * Requires: QWEN35_MODEL_PATH env var pointing to Qwen3.5-0.8B-BF16.gguf
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include "../../src/core/model.h"
#include "../../src/models/qwen35.h"
#include "../../src/loader/tokenizer.h"

static std::string get_qwen35_model_path() {
    const char* path = std::getenv("QWEN35_MODEL_PATH");
    return path ? std::string(path) : "";
}

#define SKIP_IF_NO_MODEL()                                          \
    do {                                                            \
        if (get_qwen35_model_path().empty()) {                     \
            GTEST_SKIP() << "QWEN35_MODEL_PATH not set, skipping"; \
        }                                                          \
    } while (0)

class Qwen35ForwardAttnTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (get_qwen35_model_path().empty()) return;

        model_ = std::make_unique<Model>();
        model_->load_metadata(get_qwen35_model_path());
        model_->load_tensors();
    }

    static void TearDownTestSuite() {
        model_.reset();
    }

    static std::unique_ptr<Model> model_;
};

std::unique_ptr<Model> Qwen35ForwardAttnTest::model_ = nullptr;

// ============================================================
// Test: Build prefill graph without crash
// ============================================================
TEST_F(Qwen35ForwardAttnTest, BuildPrefillGraphSucceeds) {
    SKIP_IF_NO_MODEL();

    Qwen35ForwardPass fp(*model_, &model_->get_metadata(), 2048, 1);

    std::vector<int32_t> tokens = {1, 2, 3, 4, 5};  // Dummy tokens
    
    EXPECT_NO_THROW({
        ggml_cgraph* gf = fp.build_prefill_graph(tokens, 0, 0);
        EXPECT_NE(gf, nullptr);
    });
}

// ============================================================
// Test: Full prefill → logits pipeline with real tokens
// ============================================================
TEST_F(Qwen35ForwardAttnTest, PrefillProducesFiniteLogits) {
    SKIP_IF_NO_MODEL();

    const auto& meta = model_->get_metadata();
    Qwen35ForwardPass fp(*model_, &meta, 2048, 1);
    ggml_backend_sched_t sched = model_->get_scheduler();

    // Encode a simple prompt
    Tokenizer* tok = model_->get_tokenizer();
    ASSERT_NE(tok, nullptr);

    std::vector<int32_t> tokens = tok->encode("Hello");
    ASSERT_GT(tokens.size(), 0u);

    // Build graph
    ggml_backend_sched_reset(sched);
    ggml_cgraph* gf = fp.build_prefill_graph(tokens, 0, 0);
    ASSERT_NE(gf, nullptr);

    // Allocate
    bool alloc_ok = ggml_backend_sched_alloc_graph(sched, gf);
    ASSERT_TRUE(alloc_ok) << "Graph allocation failed";

    // Set inputs
    fp.set_inputs(gf, tokens, 0);

    // Compute
    ggml_backend_sched_graph_compute(sched, gf);

    // Get logits
    std::vector<float> logits = fp.get_output_logits(gf);
    ASSERT_EQ(logits.size(), meta.vocab_size * tokens.size())
        << "Expected vocab_size * n_tokens logits";

    // Verify logits are finite (not NaN or Inf)
    // Check last token's logits (the ones we'd sample from)
    size_t offset = (tokens.size() - 1) * meta.vocab_size;
    uint32_t finite_count = 0;
    uint32_t nan_count = 0;
    uint32_t inf_count = 0;

    for (uint32_t i = 0; i < meta.vocab_size; ++i) {
        float v = logits[offset + i];
        if (std::isfinite(v)) finite_count++;
        else if (std::isnan(v)) nan_count++;
        else inf_count++;
    }

    EXPECT_EQ(nan_count, 0u) << "Found NaN in logits";
    EXPECT_EQ(inf_count, 0u) << "Found Inf in logits";
    EXPECT_EQ(finite_count, meta.vocab_size) << "Not all logits are finite";
}

// ============================================================
// Test: Logits are not all zeros (model is doing something)
// ============================================================
TEST_F(Qwen35ForwardAttnTest, LogitsNotAllZero) {
    SKIP_IF_NO_MODEL();

    const auto& meta = model_->get_metadata();
    Qwen35ForwardPass fp(*model_, &meta, 2048, 1);
    ggml_backend_sched_t sched = model_->get_scheduler();

    Tokenizer* tok = model_->get_tokenizer();
    std::vector<int32_t> tokens = tok->encode("Hello");

    ggml_backend_sched_reset(sched);
    ggml_cgraph* gf = fp.build_prefill_graph(tokens, 0, 0);
    ggml_backend_sched_alloc_graph(sched, gf);
    fp.set_inputs(gf, tokens, 0);
    ggml_backend_sched_graph_compute(sched, gf);

    std::vector<float> logits = fp.get_output_logits(gf);

    // Check last token's logits have some variance
    size_t offset = (tokens.size() - 1) * meta.vocab_size;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < meta.vocab_size; ++i) {
        float v = logits[offset + i];
        sum += v;
        sum_sq += v * v;
    }

    float mean = sum / meta.vocab_size;
    float variance = (sum_sq / meta.vocab_size) - (mean * mean);

    // Logits should have non-trivial variance
    EXPECT_GT(variance, 1e-6f) << "Logits have near-zero variance — model output is flat";
}

// ============================================================
// Test: Cache advance works after prefill
// ============================================================
TEST_F(Qwen35ForwardAttnTest, CacheAdvanceAfterPrefill) {
    SKIP_IF_NO_MODEL();

    const auto& meta = model_->get_metadata();
    Qwen35ForwardPass fp(*model_, &meta, 2048, 1);
    ggml_backend_sched_t sched = model_->get_scheduler();

    Tokenizer* tok = model_->get_tokenizer();
    std::vector<int32_t> tokens = tok->encode("Hello world");

    ggml_backend_sched_reset(sched);
    ggml_cgraph* gf = fp.build_prefill_graph(tokens, 0, 0);
    ggml_backend_sched_alloc_graph(sched, gf);
    fp.set_inputs(gf, tokens, 0);
    ggml_backend_sched_graph_compute(sched, gf);

    // Advance cache — should not crash
    EXPECT_NO_THROW(fp.advance_cache(tokens.size(), 0));
    EXPECT_EQ(fp.get_cache_pos(0), tokens.size());
}
