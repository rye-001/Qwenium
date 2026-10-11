// test_embed.cpp — co-located test for src/engine/embed.cpp, and the gate for
// the gemma-embedding2 recipe.
//
// Pure tests (always run): framing, pooling and normalisation.
//
// Model gate (EMBEDDINGGEMMA2_MODEL_PATH = embeddinggemma-2-BF16.gguf):
//   tokens   — our tokenizer + frame give EXACTLY llama.cpp's ids;
//   vectors  — cosine with llama.cpp's embedding >= kMinCosine for every
//              prompt, materialized and flash. Not byte-identical by design:
//              a different ggml revision and different kernels (llama.cpp ran
//              flash on Metal).
// Reference: tests/fixtures/embed/embeddinggemma2_llamacpp.txt — llama.cpp
//   f498f864f `llama-embedding` on 10 prompts, the longest 1948 tokens so the
//   ±512 sliding window is exercised (a shorter text cannot tell a windowed
//   mask from an open one).
//
// One model resident at a time (architecture.md §5): each model-backed test
// loads its own and frees it.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/embed.h"
#include "engine/model.h"
#include "loader/tokenizer.h"
#include "models/model_registry.h"
#include "models/gemma_embedding2.h"

using qinf::engine::embed::frame;
using qinf::engine::embed::pool_and_normalize;
using Pooling = IEmbeddingOutput::Pooling;

namespace {

constexpr double kMinCosine = 0.9999;

double norm2(const std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += (double)x * x;
    return std::sqrt(s);
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += (double)a[i] * b[i];
    return d / (norm2(a) * norm2(b));
}

}  // namespace

// ── Pure: pooling and normalisation ──────────────────────────────────────────

TEST(EmbedPool, MeanAveragesEveryTokenThenNormalises) {
    // dim 2, three tokens: mean = (2, 2)/... → unit (1/√2, 1/√2).
    const std::vector<float> per_token{1, 3,  2, 2,  3, 1};
    const auto v = pool_and_normalize(per_token, 2, 3, Pooling::Mean);
    ASSERT_EQ(v.size(), 2u);
    EXPECT_NEAR(v[0], 1.0 / std::sqrt(2.0), 1e-7);
    EXPECT_NEAR(v[1], 1.0 / std::sqrt(2.0), 1e-7);
}

TEST(EmbedPool, LastTakesTheFinalTokenOnly) {
    const std::vector<float> per_token{9, 9,  9, 9,  3, 4};
    const auto v = pool_and_normalize(per_token, 2, 3, Pooling::Last);
    EXPECT_NEAR(v[0], 0.6, 1e-7);
    EXPECT_NEAR(v[1], 0.8, 1e-7);
}

TEST(EmbedPool, ResultIsUnitLength) {
    std::vector<float> per_token(768 * 5);
    for (size_t i = 0; i < per_token.size(); ++i) per_token[i] = std::sin(0.3f * i) * 7.0f;
    EXPECT_NEAR(norm2(pool_and_normalize(per_token, 768, 5, Pooling::Mean)), 1.0, 1e-6);
}

TEST(EmbedPool, RefusesWhatItCannotPool) {
    EXPECT_THROW(pool_and_normalize({1, 2, 3}, 2, 2, Pooling::Mean), std::runtime_error);  // size
    EXPECT_THROW(pool_and_normalize({}, 2, 0, Pooling::Mean), std::runtime_error);         // empty
    EXPECT_THROW(pool_and_normalize({0, 0, 0, 0}, 2, 2, Pooling::Mean), std::runtime_error);  // zero
}

// ── Pure: framing ────────────────────────────────────────────────────────────

TEST(EmbedFrame, AddsBosAndEosWhenTheArchitectureAsks) {
    ModelMetadata meta;
    meta.bos_token_id = 2;
    meta.eos_token_id = 1;
    TokenizerConfig cfg;
    cfg.add_bos_token = true;
    cfg.add_eos_token = true;
    EXPECT_EQ(frame(cfg, meta, {10, 11}), (std::vector<int32_t>{2, 10, 11, 1}));
}

// The GGUF's flags do not decide: a file that says "no BOS, no EOS" still gets
// the framing its architecture was trained with — and the reverse.
TEST(EmbedFrame, TheGgufFlagsDoNotDecide) {
    ModelMetadata meta;
    meta.bos_token_id = 2;
    meta.eos_token_id = 1;
    meta.add_bos_token = false;
    meta.raw_kv.set("tokenizer.ggml.add_eos_token", false);
    EXPECT_EQ(frame(gemma_embedding2_tokenizer_config(), meta, {10}),
              (std::vector<int32_t>{2, 10, 1}));
    meta.add_bos_token = true;
    meta.raw_kv.set("tokenizer.ggml.add_eos_token", true);
    EXPECT_EQ(frame(TokenizerConfig{}, meta, {10}), (std::vector<int32_t>{10}));
}

TEST(EmbedFrame, RefusesAFramingTokenTheFileLacks) {
    ModelMetadata meta;
    meta.bos_token_id = 2;
    meta.eos_token_id = -1;
    TokenizerConfig cfg;
    cfg.add_eos_token = true;
    EXPECT_THROW(frame(cfg, meta, {10}), std::runtime_error);
}

// ── Model gate: EmbeddingGemma-2 against llama.cpp ───────────────────────────

namespace {

struct RefPrompt {
    std::string text;
    std::vector<int32_t> tokens;
    std::vector<float> embedding;
};

std::vector<RefPrompt> load_reference() {
    const std::string path = QINF_TEST_FIXTURES_DIR "/embed/embeddinggemma2_llamacpp.txt";
    std::ifstream f(path);
    if (!f) throw std::runtime_error("test_embed: fixture '" + path + "' expected readable, got: not");
    std::vector<RefPrompt> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("text ", 0) == 0) {
            out.push_back(RefPrompt{line.substr(5), {}, {}});
        } else if (line.rfind("tokens ", 0) == 0) {
            std::istringstream ss(line.substr(7));
            size_t n; ss >> n;
            out.back().tokens.resize(n);
            for (auto& t : out.back().tokens) ss >> t;
        } else if (line.rfind("embedding ", 0) == 0) {
            std::istringstream ss(line.substr(10));
            size_t n; ss >> n;
            out.back().embedding.resize(n);
            for (auto& v : out.back().embedding) ss >> v;
        }
    }
    return out;
}

class EmbeddingGemma2Test : public ::testing::Test {
protected:
    void SetUp() override {
        const char* p = std::getenv("EMBEDDINGGEMMA2_MODEL_PATH");
        if (!p || !*p) GTEST_SKIP() << "EMBEDDINGGEMMA2_MODEL_PATH not set — skipping the model gate";
        register_builtin_models();
        model_ = std::make_unique<Model>();
        model_->load_metadata(p);
        model_->load_tensors();
        fp_ = create_forward_pass(*model_, &model_->get_metadata(), 4096, 1);
    }
    void TearDown() override { fp_.reset(); model_.reset(); }

    std::vector<int32_t> tokens_of(const std::string& text) {
        const ModelMetadata& meta = model_->get_metadata();
        return frame(lookup_tokenizer_config(meta.architecture), meta,
                     model_->get_tokenizer()->encode(text));
    }
    std::vector<float> embed(const std::vector<int32_t>& toks) {
        return qinf::engine::embed::run(*fp_, model_->get_scheduler(), toks,
                                        model_->get_metadata().architecture);
    }

    std::unique_ptr<Model> model_;
    std::unique_ptr<ForwardPassBase> fp_;
};

}  // namespace

TEST_F(EmbeddingGemma2Test, ConfigReadsTheFile) {
    auto* r = dynamic_cast<GemmaEmbedding2ForwardPass*>(fp_.get());
    ASSERT_NE(r, nullptr);
    const auto& c = r->config();
    EXPECT_EQ(c.n_layers, 24u);
    EXPECT_EQ(c.embd_out, 768u);
    EXPECT_EQ(c.pooling, Pooling::Mean);
    EXPECT_EQ(c.sliding_radius, 512u);   // GGUF 1024 = 2 × HF's inclusive radius
    // 5:1 pattern, globals at 5, 11, 17, 23 — and every layer has its own V.
    for (uint32_t il = 0; il < c.n_layers; ++il)
        EXPECT_EQ(c.is_global[il], il % 6 == 5) << "layer " << il;
    EXPECT_EQ(c.n_kv_heads[0], 2u);
    EXPECT_EQ(c.n_kv_heads[5], 1u);
}

TEST_F(EmbeddingGemma2Test, TokensMatchLlamaCpp) {
    for (const RefPrompt& rp : load_reference())
        EXPECT_EQ(tokens_of(rp.text), rp.tokens) << "prompt: " << rp.text.substr(0, 60);
}

TEST_F(EmbeddingGemma2Test, EmbeddingsMatchLlamaCpp) {
    double worst = 1.0;
    for (const RefPrompt& rp : load_reference()) {
        const std::vector<float> got = embed(rp.tokens);
        ASSERT_EQ(got.size(), rp.embedding.size());
        const double c = cosine(got, rp.embedding);
        worst = std::min(worst, c);
        std::printf("[ gate ] %4zu tokens  cosine %.7f\n", rp.tokens.size(), c);
        EXPECT_GE(c, kMinCosine) << rp.tokens.size() << " tokens, prompt: " << rp.text.substr(0, 60);
        EXPECT_NEAR(norm2(got), 1.0, 1e-5);
    }
    std::printf("[ gate ] materialized: worst cosine vs llama.cpp = %.7f\n", worst);
}

TEST_F(EmbeddingGemma2Test, FlashEmbeddingsMatchLlamaCpp) {
    fp_->set_prefill_attn_impl(ForwardPassBase::AttnImpl::Flash);
    double worst = 1.0;
    for (const RefPrompt& rp : load_reference()) {
        const double c = cosine(embed(rp.tokens), rp.embedding);
        worst = std::min(worst, c);
        EXPECT_GE(c, kMinCosine) << rp.tokens.size() << " tokens, prompt: " << rp.text.substr(0, 60);
    }
    std::printf("[ gate ] flash: worst cosine vs llama.cpp = %.7f\n", worst);
}

// The default (materialized) path is byte-reproducible: same tokens, same bits.
TEST_F(EmbeddingGemma2Test, RepeatIsBitwiseIdentical) {
    const auto ref = load_reference();
    const auto a = embed(ref[0].tokens);
    const auto b = embed(ref[9].tokens);   // a different, longer graph in between
    const auto c = embed(ref[0].tokens);
    (void)b;
    EXPECT_EQ(a, c);
}

// An encoder has no logits, cache, decode or chat — each refuses by name.
TEST_F(EmbeddingGemma2Test, RefusesTheGenerationInterface) {
    EXPECT_THROW(fp_->build_prefill_graph({2, 10, 1}, 0), std::runtime_error);
    EXPECT_THROW(fp_->build_decoding_graph({10}, {0}, {0}), std::runtime_error);
    EXPECT_THROW(fp_->clear_slot(0), std::runtime_error);
    EXPECT_THROW(fp_->get_cache_pos(0), std::runtime_error);
    EXPECT_THROW(lookup_chat_template("gemma-embedding2"), std::runtime_error);
}

TEST_F(EmbeddingGemma2Test, RefusesTooLongAndEmpty) {
    EXPECT_THROW(embed({}), std::runtime_error);
    EXPECT_THROW(embed(std::vector<int32_t>(4097, 10)), std::runtime_error);
}

// ── A generation recipe is not an embedder ──────────────────────────────────

TEST(EmbedRun, RefusesARecipeWithoutIEmbeddingOutput) {
    const char* p = std::getenv("QWEN3_MODEL_PATH");
    if (!p || !*p) GTEST_SKIP() << "QWEN3_MODEL_PATH not set";
    register_builtin_models();
    Model model;
    model.load_metadata(p);
    model.load_tensors();
    auto fp = create_forward_pass(model, &model.get_metadata(), 512, 1);
    EXPECT_THROW(qinf::engine::embed::run(*fp, model.get_scheduler(), {1, 2, 3},
                                          model.get_metadata().architecture),
                 std::runtime_error);
}
