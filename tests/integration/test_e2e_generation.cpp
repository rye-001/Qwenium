#include "gtest/gtest.h"
#include "qwen3-core/qwen3-model.h"
#include "qwen3-core/gguf-loader.h"
#include "qwen3-core/forward-pass.h"
#include "qwen3-core/tokenizer.h"
#include "sampling/sampling.h"
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <set>

// Struct to hold test parameters
struct GenerationTestParam {
    std::string prompt;
    std::vector<std::string> expected_keywords;
    int max_tokens_to_generate;
};

// Test fixture for end-to-end generation
class E2EGenerationTest : public ::testing::TestWithParam<GenerationTestParam> {
protected:
    static std::map<std::string, std::shared_ptr<Qwen3Model>> model_cache_;
    static std::string model_path_1_7b_;
    static std::string model_path_0_6b_;
    static std::string model_path_qwen2_14b_;

    static void SetUpTestSuite() {
        model_path_1_7b_ = "./Qwen3-1.7B-BF16.gguf";
        model_path_0_6b_ = "./Qwen3-0.6B-Q8_0.gguf";
        model_path_qwen2_14b_ = "./qwen2.5-coder-14b-instruct-q4_0.gguf";
        load_model(model_path_1_7b_);
        load_model(model_path_0_6b_);
        load_model(model_path_qwen2_14b_);
    }

    static void load_model(const std::string& model_path) {
        if (model_cache_.find(model_path) == model_cache_.end()) {
            try {
                auto model = std::make_shared<Qwen3Model>();
                model->load_metadata(model_path);
                model->load_tensors();
                model_cache_[model_path] = model;
            } catch (const std::exception& e) {
                std::cerr << "Failed to load model for testing: " << e.what()
                          << "\nMake sure '" << model_path << "' is in the project root." << std::endl;
                model_cache_[model_path] = nullptr; // Mark as failed to avoid re-trying
            }
        }
    }

    std::string generate_text(const std::string& model_path, const GenerationTestParam& param) {
        auto model_it = model_cache_.find(model_path);
        if (model_it == model_cache_.end() || !model_it->second) {
            throw std::runtime_error("Model not loaded or failed to load, skipping test.");
        }
        auto model = model_it->second;

        Tokenizer tokenizer(&model->get_metadata());
        qwen3::GreedySampler sampler;
        Qwen3ForwardPass forward_pass(*model, &model->get_metadata(), 4096);
        ggml_backend_sched_t scheduler = model->get_scheduler();

        // --- Prefill Phase ---
        std::vector<int32_t> tokens = tokenizer.encode(param.prompt);
        ggml_backend_sched_reset(scheduler);
        ggml_cgraph* gf_prefill = forward_pass.build_prefill_graph(tokens, 0);
        ggml_backend_sched_alloc_graph(scheduler, gf_prefill);
        forward_pass.set_inputs(gf_prefill, tokens, 0);
        ggml_backend_sched_graph_compute(scheduler, gf_prefill);
        forward_pass.advance_cache(tokens.size(), 0);

        std::vector<float> logits = forward_pass.get_output_logits(gf_prefill);
        size_t vocab_size = model->get_metadata().vocab_size;
        std::vector<float> last_token_logits(logits.end() - vocab_size, logits.end());
        int next_token_id = sampler.sample(last_token_logits, tokens);
        tokens.push_back(next_token_id);

        // --- Decode Phase ---
        std::string generated_text = tokenizer.decode(next_token_id);
        for (int i = 1; i < param.max_tokens_to_generate; ++i) {
            if (next_token_id == model->get_metadata().eos_token_id) {
                break;
            }

            std::vector<int32_t> current_token_vec = {next_token_id};
            int current_pos = forward_pass.get_cache_pos(0);

            ggml_backend_sched_reset(scheduler);
            ggml_cgraph* gf_decode = forward_pass.build_prefill_graph(current_token_vec, current_pos);
            ggml_backend_sched_alloc_graph(scheduler, gf_decode);
            forward_pass.set_inputs(gf_decode, current_token_vec, current_pos);
            ggml_backend_sched_graph_compute(scheduler, gf_decode);
            forward_pass.advance_cache(1, 0);

            logits = forward_pass.get_output_logits(gf_decode);
            last_token_logits.assign(logits.begin(), logits.begin() + vocab_size);
            
            next_token_id = sampler.sample(last_token_logits, tokens);
            tokens.push_back(next_token_id);
            generated_text += tokenizer.decode(next_token_id);
        }
        return generated_text;
    }

    void run_test_for_model(const std::string& model_path) {

 ggml_log_set([](ggml_log_level level, const char * text, void * user_data) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
    (void)user_data;
}, nullptr);



        auto model_it = model_cache_.find(model_path);
        if (model_it == model_cache_.end() || !model_it->second) {
            GTEST_SKIP() << "Model not loaded or failed to load, skipping test.";
            return;
        }

        const auto& param = GetParam();
        std::string output = generate_text(model_path, param);

        // Log output for debugging on failure
        SCOPED_TRACE("Prompt: '" + param.prompt + "'\nModel: " + model_path + "\nGenerated: '" + output + "'");

        // Negative Assertion: Check for minimal length
        ASSERT_GT(output.length(), 5) << "Generated text is too short.";

        // Negative Assertion: Check for degenerative repetition
        if (output.length() >= 10) {
            std::string first_char_repeated = std::string(10, output[0]);
            ASSERT_NE(output.substr(0, 10), first_char_repeated) << "Output appears to be a degenerative loop.";
        }

        // Positive Assertion: Check for expected keywords
        bool found_keyword = false;
        for (const auto& keyword : param.expected_keywords) {
            if (output.find(keyword) != std::string::npos) {
                found_keyword = true;
                break;
            }
        }
        ASSERT_TRUE(found_keyword) << "None of the expected keywords were found in the output.";
    }
};

// Initialize static members
std::map<std::string, std::shared_ptr<Qwen3Model>> E2EGenerationTest::model_cache_;
std::string E2EGenerationTest::model_path_1_7b_;
std::string E2EGenerationTest::model_path_0_6b_;
std::string E2EGenerationTest::model_path_qwen2_14b_;

// Test cases
TEST_P(E2EGenerationTest, GenerationWithQwen3_1_7B) {
    run_test_for_model(model_path_1_7b_);
}

TEST_P(E2EGenerationTest, GenerationWithQwen3_0_6B) {
    run_test_for_model(model_path_0_6b_);
}

TEST_P(E2EGenerationTest, GenerationWithQwen2_14B) {
    run_test_for_model(model_path_qwen2_14b_);
}

// Instantiate the test suite with parameters
INSTANTIATE_TEST_SUITE_P(
    DefaultPrompts,
    E2EGenerationTest,
    ::testing::Values(
        GenerationTestParam{
            "Hello",
            {"I'm", "How can", "IĠneed"},
            20
        },
        GenerationTestParam{
            "What is C++?",
            {"programming", "language", "code"},
            30
        }
    )
);
