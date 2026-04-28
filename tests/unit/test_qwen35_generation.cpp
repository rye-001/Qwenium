/**
 * test_qwen35_generation.cpp — Step 8: End-to-end text generation
 * Requires: QWEN35_MODEL_PATH env var
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>

#include "../../src/core/model.h"
#include "../../src/models/qwen3.h"
#include "../../src/models/qwen35.h"
#include "../../src/loader/tokenizer.h"
#include "../../src/sampling/sampling.h"

static std::string get_qwen35_model_path() {
    const char* path = std::getenv("QWEN35_MODEL_PATH");
    return path ? std::string(path) : "";
}

#define SKIP_IF_NO_MODEL() \
    do { if (get_qwen35_model_path().empty()) { GTEST_SKIP() << "QWEN35_MODEL_PATH not set"; } } while(0)

class Qwen35GenerationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (get_qwen35_model_path().empty()) return;
        model_ = std::make_unique<Model>();
        model_->load_metadata(get_qwen35_model_path());
        model_->load_tensors();
    }
    static void TearDownTestSuite() { model_.reset(); }

    std::string generate(const std::string& prompt, int max_tokens = 20) {
        const auto& meta = model_->get_metadata();
        auto fp = std::make_unique<Qwen35ForwardPass>(*model_, &meta, 2048, 1);
        ggml_backend_sched_t sched = model_->get_scheduler();
        Tokenizer* tok = model_->get_tokenizer();
        qwenium::GreedySampler sampler;

        std::vector<int32_t> tokens = tok->encode(prompt);
        EXPECT_GT(tokens.size(), 0u);

        // Prefill
        ggml_backend_sched_reset(sched);
        ggml_cgraph* gf = fp->build_prefill_graph(tokens, 0, 0);
        EXPECT_TRUE(ggml_backend_sched_alloc_graph(sched, gf));
        fp->set_inputs(gf, tokens, 0);
        ggml_backend_sched_graph_compute(sched, gf);

        std::vector<float> all_logits = fp->get_output_logits(gf);
        size_t offset = (tokens.size() - 1) * meta.vocab_size;
        std::vector<float> logits(all_logits.begin() + offset,
                                   all_logits.begin() + offset + meta.vocab_size);
        fp->advance_cache(tokens.size(), 0);
        int pos = tokens.size();

        // Decode loop
        std::string generated;
        for (int i = 0; i < max_tokens; ++i) {
            int32_t next_token = sampler.sample(logits, tokens);
            if (next_token == tok->get_eos_token_id()) break;
            std::string decoded_tok = tok->decode(next_token);
            if (decoded_tok == "<|im_end|>" || decoded_tok == "<|endoftext|>") break;

            tokens.push_back(next_token);
            generated += decoded_tok;

            std::vector<int32_t> next_vec = {next_token};
            ggml_backend_sched_reset(sched);
            gf = fp->build_prefill_graph(next_vec, pos, 0);
            EXPECT_TRUE(ggml_backend_sched_alloc_graph(sched, gf));
            fp->set_inputs(gf, next_vec, pos);
            ggml_backend_sched_graph_compute(sched, gf);

            logits = fp->get_output_logits(gf);
            logits.resize(meta.vocab_size);
            fp->advance_cache(1, 0);
            pos++;
        }
        return generated;
    }

    static std::unique_ptr<Model> model_;
};

std::unique_ptr<Model> Qwen35GenerationTest::model_ = nullptr;

TEST_F(Qwen35GenerationTest, GeneratesNonEmpty) {
    SKIP_IF_NO_MODEL();
    std::string output = generate("Hello");
    std::cout << "[Qwen35] 'Hello' -> '" << output << "'" << std::endl;
    EXPECT_GT(output.size(), 0u);
}

TEST_F(Qwen35GenerationTest, NotDegenerate) {
    SKIP_IF_NO_MODEL();
    std::string output = generate("The capital of France is", 30);
    std::cout << "[Qwen35] 'The capital of France is' -> '" << output << "'" << std::endl;
    EXPECT_GT(output.size(), 2u);
    bool has_variety = false;
    for (size_t i = 1; i < output.size(); ++i) {
        if (output[i] != output[0]) { has_variety = true; break; }
    }
    EXPECT_TRUE(has_variety) << "Degenerate: '" << output << "'";
}

TEST_F(Qwen35GenerationTest, ChatTemplate) {
    SKIP_IF_NO_MODEL();
    std::string prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";
    std::string output = generate(prompt, 30);
    std::cout << "[Qwen35] Chat -> '" << output << "'" << std::endl;
    EXPECT_GT(output.size(), 0u);
}

TEST_F(Qwen35GenerationTest, SequentialGenerations) {
    SKIP_IF_NO_MODEL();
    std::string out1 = generate("Hello", 10);
    std::string out2 = generate("Goodbye", 10);
    std::cout << "[Qwen35] Gen1: '" << out1 << "'" << std::endl;
    std::cout << "[Qwen35] Gen2: '" << out2 << "'" << std::endl;
    EXPECT_GT(out1.size(), 0u);
    EXPECT_GT(out2.size(), 0u);
}

TEST_F(Qwen35GenerationTest, LogitsStayFinite) {
    SKIP_IF_NO_MODEL();
    const auto& meta = model_->get_metadata();
    auto fp = std::make_unique<Qwen35ForwardPass>(*model_, &meta, 2048, 1);
    ggml_backend_sched_t sched = model_->get_scheduler();
    Tokenizer* tok = model_->get_tokenizer();
    qwenium::GreedySampler sampler;

    std::vector<int32_t> tokens = tok->encode("Hello world");

    ggml_backend_sched_reset(sched);
    ggml_cgraph* gf = fp->build_prefill_graph(tokens, 0, 0);
    ggml_backend_sched_alloc_graph(sched, gf);
    fp->set_inputs(gf, tokens, 0);
    ggml_backend_sched_graph_compute(sched, gf);

    std::vector<float> all_logits = fp->get_output_logits(gf);
    size_t offset = (tokens.size() - 1) * meta.vocab_size;
    std::vector<float> logits(all_logits.begin() + offset,
                               all_logits.begin() + offset + meta.vocab_size);
    fp->advance_cache(tokens.size(), 0);
    int pos = tokens.size();

    for (int step = 0; step < 15; ++step) {
        uint32_t nan_count = 0, inf_count = 0;
        for (uint32_t i = 0; i < meta.vocab_size; ++i) {
            if (std::isnan(logits[i])) nan_count++;
            else if (std::isinf(logits[i])) inf_count++;
        }
        EXPECT_EQ(nan_count, 0u) << "NaN at step " << step;
        EXPECT_EQ(inf_count, 0u) << "Inf at step " << step;

        int32_t next = sampler.sample(logits, tokens);
        if (next == tok->get_eos_token_id()) break;
        tokens.push_back(next);

        std::vector<int32_t> next_vec = {next};
        ggml_backend_sched_reset(sched);
        gf = fp->build_prefill_graph(next_vec, pos, 0);
        ggml_backend_sched_alloc_graph(sched, gf);
        fp->set_inputs(gf, next_vec, pos);
        ggml_backend_sched_graph_compute(sched, gf);

        logits = fp->get_output_logits(gf);
        logits.resize(meta.vocab_size);
        fp->advance_cache(1, 0);
        pos++;
    }
}