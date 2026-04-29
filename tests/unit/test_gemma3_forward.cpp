// test_gemma3_forward.cpp — PR G3.4
//
// Integration smoke test for the Gemma 3 model recipe: load a real Gemma 3
// GGUF, verify the architecture string is "gemma3", and confirm that prefill
// produces finite logits with the expected shape.
//
// The HF reference-logit comparison lives in the PR G3.0 fixture harness.
// This test asserts the structural plumbing — load → config → prefill — works
// end-to-end before we tighten to numeric agreement.
//
// Self-skips when GEMMA3_MODEL_PATH is unset and the default path is absent.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "../../src/loader/tokenizer.h"
#include "../../src/models/forward_pass_base.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/gemma3.h"
#include "../../src/core/model.h"

namespace {

std::string find_gemma3_path() {
    if (const char* p = std::getenv("GEMMA3_MODEL_PATH")) return p;
    return "gemma-3-1b-it-BF16.gguf";
}

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

} // namespace

class Gemma3ModelFile : public ::testing::Test {
protected:
    static std::string path_;

    static void SetUpTestSuite() {
        path_ = find_gemma3_path();
    }

    void SetUp() override {
        if (!file_exists(path_)) {
            GTEST_SKIP() << "GEMMA3_MODEL_PATH not set and " << path_
                         << " not found — skipping";
        }
        register_builtin_models();
    }
};
std::string Gemma3ModelFile::path_;

// ── Metadata loading ──────────────────────────────────────────────────────────

TEST_F(Gemma3ModelFile, ArchitectureIsGemma3) {
    Model model;
    model.load_metadata(path_);
    EXPECT_EQ(model.get_metadata().architecture, "gemma3");
}

TEST_F(Gemma3ModelFile, MetadataMatchesKnown1BShape) {
    Model model;
    model.load_metadata(path_);
    const auto& m = model.get_metadata();

    EXPECT_EQ(m.block_count,             26u);
    EXPECT_EQ(m.embedding_length,       1152u);
    EXPECT_EQ(m.attention_head_count,      4u);
    EXPECT_EQ(m.attention_head_count_kv,   1u);
    EXPECT_EQ(m.attention_key_length,    256u);
    EXPECT_EQ(m.context_length,        32768u);
    EXPECT_NEAR(m.rope_freq_base, 1000000.0f, 1.0f);
}

// ── Gemma3Config extraction ───────────────────────────────────────────────────

TEST_F(Gemma3ModelFile, ConfigFromRealMetadata) {
    Model model;
    model.load_metadata(path_);
    auto cfg = Gemma3Config::from_metadata(model.get_metadata());

    EXPECT_EQ(cfg.n_layers,  26u);
    EXPECT_EQ(cfg.n_head,     4u);
    EXPECT_EQ(cfg.n_head_kv,  1u);
    EXPECT_EQ(cfg.sliding_window, 512u);
    EXPECT_FLOAT_EQ(cfg.local_rope_base,  10000.0f);
    EXPECT_NEAR(cfg.global_rope_base, 1000000.0f, 1.0f);
}

// 5:1 alternation verified against the real GGUF: layers 5,11,17,23 global.
TEST_F(Gemma3ModelFile, FiveOnePatternFromRealMetadata) {
    Model model;
    model.load_metadata(path_);
    auto cfg = Gemma3Config::from_metadata(model.get_metadata());

    for (uint32_t i = 0; i < cfg.n_layers; ++i) {
        const bool expected_global = (i % 6 == 5);
        if (expected_global) {
            EXPECT_EQ(cfg.layer_window[i], 0u)
                << "layer " << i << " should be global";
            EXPECT_FLOAT_EQ(cfg.layer_rope_base[i], cfg.global_rope_base)
                << "layer " << i << " should use global RoPE base";
        } else {
            EXPECT_EQ(cfg.layer_window[i], cfg.sliding_window)
                << "layer " << i << " should be local (window=" << cfg.sliding_window << ")";
            EXPECT_FLOAT_EQ(cfg.layer_rope_base[i], cfg.local_rope_base)
                << "layer " << i << " should use local RoPE base";
        }
    }
}

// ── Inventory validation ──────────────────────────────────────────────────────

TEST_F(Gemma3ModelFile, InventoryValidationPasses) {
    Model model;
    model.load_metadata(path_);
    EXPECT_NO_THROW(validate_gemma3_inventory(model.get_metadata()));
}

// ── Forward pass ─────────────────────────────────────────────────────────────

// Use hardcoded token IDs to bypass the tokenizer entirely.
// Gemma 3 has a 262K-entry vocabulary; std::regex on macOS (libc++) hits its
// DFA complexity limit when the tokenizer builds a pre-tokenization pattern
// from that many entries. The forward pass itself is independent of the
// tokenizer — we exercise it here with synthetic token IDs that are valid
// indices into the embedding table (BOS=2 plus a few arbitrary indices).
TEST_F(Gemma3ModelFile, PrefillProducesFiniteLogits) {
    Model model;
    model.load_metadata(path_);
    model.load_tensors();

    // Synthetic tokens: BOS(2) + a handful of valid vocab indices.
    // No tokenizer needed — any valid index in [0, vocab_size) is fine.
    const std::vector<int32_t> tokens = {2, 100, 200, 1000, 5000};

    auto fp = create_forward_pass(model, &model.get_metadata(),
                                  /*context_len=*/128, /*max_batch=*/1, /*kvb=*/0);
    ASSERT_NE(fp, nullptr);

    std::vector<float> logits = fp->run_prefill(
        tokens, /*pos=*/0, /*slot_idx=*/0, model.get_scheduler());

    const size_t vocab = model.get_metadata().vocab_size;
    ASSERT_GE(logits.size(), vocab);

    // Last-token logit slice.
    std::vector<float> last(logits.end() - vocab, logits.end());
    bool any_nonzero = false;
    for (float v : last) {
        ASSERT_TRUE(std::isfinite(v)) << "non-finite logit";
        if (v != 0.0f) any_nonzero = true;
    }
    EXPECT_TRUE(any_nonzero) << "all logits zero — graph likely disconnected";
}

// Verify the logit distribution has meaningful structure: the top-1 logit must be
// significantly larger than the median (model is not producing a uniform distribution).
// Uses synthetic token IDs — see the note above on the tokenizer limitation.
// This is architecture-agnostic: any working transformer produces a peaked distribution.
TEST_F(Gemma3ModelFile, LogitDistributionIsPeaked) {
    Model model;
    model.load_metadata(path_);
    model.load_tensors();

    const std::vector<int32_t> tokens = {2, 100, 200, 1000, 5000};

    auto fp = create_forward_pass(model, &model.get_metadata(),
                                  /*context_len=*/128, /*max_batch=*/1, /*kvb=*/0);

    std::vector<float> logits = fp->run_prefill(
        tokens, /*pos=*/0, /*slot_idx=*/0, model.get_scheduler());

    const size_t vocab = model.get_metadata().vocab_size;
    ASSERT_GE(logits.size(), vocab);
    std::vector<float> last(logits.end() - vocab, logits.end());

    // Find max and median.
    std::vector<float> sorted = last;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    const float median = sorted[sorted.size() / 2];
    const float top1   = *std::max_element(last.begin(), last.end());

    std::cerr << "logit top-1=" << top1 << " median=" << median
              << " gap=" << (top1 - median) << "\n";

    // A working transformer produces a distribution with at least a 5-logit gap
    // between the top prediction and the median. A constant / disconnected graph
    // would produce near-zero variance.
    EXPECT_GT(top1 - median, 5.0f)
        << "logit distribution looks too uniform — forward pass may be disconnected";
}
