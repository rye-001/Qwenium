/**
 * test_turboquant_generation.cpp
 *
 * Integration correctness test: greedy token sequences produced with
 * kv_quant_bits=0 (F32 KV cache) and kv_quant_bits=4 (TurboQuant) must be
 * identical for the first N decode steps.
 *
 * Rationale: 4-bit Lloyd-Max + WHT pre-conditioning preserves enough fidelity
 * that greedy argmax should be unaffected on normal text prompts. If this test
 * fails it means either (a) the TQ path has a correctness bug or (b) the model
 * is unusually sensitive to this prompt — in which case reduce N_TOKENS or
 * switch to a less sensitive prompt before investigating further.
 *
 * Requires: QWEN35_MODEL_PATH env var pointing to tests/Qwen3.5-0.8B-BF16.gguf
 * (or any compatible checkpoint).
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../../src/models/qwen3.h"
#include "../../src/models/qwen35.h"
#include "../../src/core/model.h"
#include "../../src/loader/tokenizer.h"
#include "../../src/sampling/sampling.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string get_model_path() {
    const char* p = std::getenv("QWEN35_MODEL_PATH");
    return p ? std::string(p) : "";
}

#define SKIP_IF_NO_MODEL() \
    do { \
        if (get_model_path().empty()) { \
            GTEST_SKIP() << "QWEN35_MODEL_PATH not set — skipping TQ generation test"; \
        } \
    } while (0)

// ---------------------------------------------------------------------------
// Test fixture — model loaded once per suite
// ---------------------------------------------------------------------------

class TurboQuantGenerationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (get_model_path().empty()) return;
        model_ = std::make_unique<Model>();
        model_->load_metadata(get_model_path());
        model_->load_tensors();
    }
    static void TearDownTestSuite() { model_.reset(); }

    /**
     * Greedy decode for up to max_tokens steps.
     *
     * Uses run_prefill() throughout (both the initial prompt prefill and each
     * single-token decode step) so that the TQ per-layer path is exercised
     * identically to how complete.cpp calls it.
     *
     * Returns the generated token IDs (not including the prompt).
     */
    std::vector<int32_t> generate_tokens(const std::string& prompt,
                                         int kv_quant_bits,
                                         int max_tokens = 20) {
        const auto& meta = model_->get_metadata();
        // Fresh forward pass for each call — no shared state between bits=0 / bits=4 runs.
        auto fp   = std::make_unique<Qwen35ForwardPass>(*model_, &meta, 2048, /*max_batch=*/1, kv_quant_bits);
        auto* sched = model_->get_scheduler();
        auto* tok   = model_->get_tokenizer();
        qwenium::GreedySampler sampler;

        std::vector<int32_t> ctx_tokens = tok->encode(prompt);
        EXPECT_GT(ctx_tokens.size(), 0u) << "Tokeniser returned empty sequence";

        // ── Prefill ──────────────────────────────────────────────────────────
        std::vector<float> all_logits = fp->run_prefill(ctx_tokens, 0, 0, sched);

        // Extract last-token logits
        const size_t last_off = (ctx_tokens.size() - 1) * meta.vocab_size;
        std::vector<float> cur_logits(all_logits.begin() + last_off,
                                      all_logits.begin() + last_off + meta.vocab_size);

        // ── Decode loop ──────────────────────────────────────────────────────
        std::vector<int32_t> generated;
        generated.reserve(max_tokens);

        for (int step = 0; step < max_tokens; ++step) {
            const int32_t next = sampler.sample(cur_logits, ctx_tokens);

            // EOS / stop-string check
            if (next == tok->get_eos_token_id()) break;
            const std::string decoded = tok->decode(next);
            if (decoded == "<|im_end|>" || decoded == "<|endoftext|>") break;

            generated.push_back(next);
            ctx_tokens.push_back(next);

            const int pos = static_cast<int>(fp->get_cache_pos(0));
            std::vector<float> step_logits = fp->run_prefill({next}, pos, 0, sched);
            cur_logits.assign(step_logits.begin(),
                              step_logits.begin() + meta.vocab_size);
        }
        return generated;
    }

    // Helper: decode a token-ID vector to a human-readable string for diagnostics
    std::string decode_tokens(const std::vector<int32_t>& ids) {
        auto* tok = model_->get_tokenizer();
        std::string out;
        for (int32_t id : ids) out += tok->decode(id);
        return out;
    }

    static std::unique_ptr<Model> model_;
};

std::unique_ptr<Model> TurboQuantGenerationTest::model_ = nullptr;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/**
 * Short plain-text prompt.
 * Verifies that TQ produces the same greedy token sequence as the F32 baseline.
 */
TEST_F(TurboQuantGenerationTest, TokenParity_ShortPrompt) {
    SKIP_IF_NO_MODEL();
    constexpr int N = 20;
    const std::string prompt = "The capital of France is";

    auto seq_f32 = generate_tokens(prompt, /*kv_quant_bits=*/0, N);
    auto seq_tq4 = generate_tokens(prompt, /*kv_quant_bits=*/4, N);

    std::cout << "[bits=0] " << decode_tokens(seq_f32) << "\n";
    std::cout << "[bits=4] " << decode_tokens(seq_tq4) << "\n";

    ASSERT_FALSE(seq_f32.empty()) << "F32 path produced no tokens";
    ASSERT_FALSE(seq_tq4.empty()) << "TQ-4 path produced no tokens";

    // Both sequences must be non-trivially long (model is running, not crashing)
    EXPECT_GE(seq_f32.size(), 3u);
    EXPECT_GE(seq_tq4.size(), 3u);

    // Token-by-token parity up to the length of the shorter sequence
    const size_t compare_len = std::min(seq_f32.size(), seq_tq4.size());
    for (size_t i = 0; i < compare_len; ++i) {
        EXPECT_EQ(seq_f32[i], seq_tq4[i])
            << "Token mismatch at position " << i
            << ": bits=0 gave " << seq_f32[i]
            << " ('" << model_->get_tokenizer()->decode(seq_f32[i]) << "')"
            << ", bits=4 gave " << seq_tq4[i]
            << " ('" << model_->get_tokenizer()->decode(seq_tq4[i]) << "')";
    }
}

/**
 * Chat-template prompt (tests the KV cache across the structured prefix tokens).
 */
TEST_F(TurboQuantGenerationTest, TokenParity_ChatTemplate) {
    SKIP_IF_NO_MODEL();
    constexpr int N = 20;
    const std::string prompt =
        "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n";

    auto seq_f32 = generate_tokens(prompt, 0, N);
    auto seq_tq4 = generate_tokens(prompt, 4, N);

    std::cout << "[bits=0] " << decode_tokens(seq_f32) << "\n";
    std::cout << "[bits=4] " << decode_tokens(seq_tq4) << "\n";

    ASSERT_FALSE(seq_f32.empty()) << "F32 path produced no tokens";
    ASSERT_FALSE(seq_tq4.empty()) << "TQ-4 path produced no tokens";

    const size_t compare_len = std::min(seq_f32.size(), seq_tq4.size());
    for (size_t i = 0; i < compare_len; ++i) {
        EXPECT_EQ(seq_f32[i], seq_tq4[i])
            << "Token mismatch at position " << i
            << ": bits=0 gave " << seq_f32[i]
            << " ('" << model_->get_tokenizer()->decode(seq_f32[i]) << "')"
            << ", bits=4 gave " << seq_tq4[i]
            << " ('" << model_->get_tokenizer()->decode(seq_tq4[i]) << "')";
    }
}

/**
 * Smoke test: bits=2 and bits=3 paths don't crash and produce non-empty output.
 * We don't assert token parity here — lower bit-widths have higher quantization
 * noise — but the paths must be runnable without assertion failures.
 */
TEST_F(TurboQuantGenerationTest, NoCrash_Bits2And3) {
    SKIP_IF_NO_MODEL();
    const std::string prompt = "Hello";

    auto seq2 = generate_tokens(prompt, 2, 10);
    auto seq3 = generate_tokens(prompt, 3, 10);

    std::cout << "[bits=2] " << decode_tokens(seq2) << "\n";
    std::cout << "[bits=3] " << decode_tokens(seq3) << "\n";

    EXPECT_GT(seq2.size(), 0u) << "bits=2 path produced no tokens";
    EXPECT_GT(seq3.size(), 0u) << "bits=3 path produced no tokens";
}
