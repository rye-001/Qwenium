// test_embed_mode.cpp — co-located test for src/cli/embed_mode.cpp (--embed).
//
// Pure (always): the flag refusals, the input-file rules, the JSONL record.
// End to end (EMBEDDINGGEMMA2_MODEL_PATH): run_embed over an --embed-in file
//   writes one record per line whose vectors match llama.cpp's reference
//   (tests/fixtures/embed/) to the same cosine bar as engine/embed's gate —
//   proof the CLI adds framing and formatting, not error.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "embed_mode.h"
#include "engine/model.h"
#include "models/model_registry.h"

namespace {

CliArgs embed_args() {
    CliArgs a;
    a.embed = true;
    a.prompt = "hello";
    return a;
}

std::vector<float> parse_vector(const std::string& line) {
    const size_t lb = line.find('['), rb = line.rfind(']');
    std::vector<float> v;
    std::istringstream ss(line.substr(lb + 1, rb - lb - 1));
    std::string tok;
    while (std::getline(ss, tok, ',')) v.push_back(std::stof(tok));
    return v;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return d / std::sqrt(na * nb);
}

}  // namespace

// ── Flag refusals ────────────────────────────────────────────────────────────

TEST(EmbedMode, AcceptsAPromptOrAFile) {
    EXPECT_EQ(embed_mode_refusal(embed_args()), "");
    CliArgs a = embed_args();
    a.prompt.clear();
    a.embed_in_path = "texts.txt";
    a.flash_attn = true;           // allowed: it applies to the one pass
    a.context_length = 8192;       // allowed: the longest text accepted
    EXPECT_EQ(embed_mode_refusal(a), "");
}

TEST(EmbedMode, RefusesNeitherOrBothInputs) {
    CliArgs a = embed_args();
    a.prompt.clear();
    EXPECT_NE(embed_mode_refusal(a).find("actual: none"), std::string::npos);
    a.prompt = "x";
    a.embed_in_path = "texts.txt";
    EXPECT_NE(embed_mode_refusal(a).find("actual: 2"), std::string::npos);
}

// ── Images (docs/plan-embed-image.md G6) ─────────────────────────────────────

TEST(EmbedMode, AcceptsAnImageWithItsTower) {
    CliArgs a = embed_args();
    a.prompt.clear();
    a.image_path = "x.png";
    a.mmproj_path = "mmproj.gguf";
    EXPECT_EQ(embed_mode_refusal(a), "");
}

TEST(EmbedMode, RefusesAnImageWithoutItsTowerAndTheReverse) {
    CliArgs a = embed_args();
    a.prompt.clear();
    a.image_path = "x.png";
    EXPECT_NE(embed_mode_refusal(a).find("expected --mmproj"), std::string::npos) << embed_mode_refusal(a);
    a.image_path.clear();
    a.mmproj_path = "mmproj.gguf";
    a.prompt = "text";
    EXPECT_NE(embed_mode_refusal(a).find("expected --image PATH"), std::string::npos) << embed_mode_refusal(a);
}

TEST(EmbedMode, RefusesAnImageAndATextTogether) {
    CliArgs a = embed_args();   // has -p
    a.image_path = "x.png";
    a.mmproj_path = "mmproj.gguf";
    EXPECT_NE(embed_mode_refusal(a).find("exactly one input"), std::string::npos);
}

TEST(EmbedMode, RefusesFlagsAnEmbeddingWouldIgnore) {
    auto refusal_names = [](CliArgs a, const char* flag) {
        const std::string r = embed_mode_refusal(a);
        return r.find(flag) != std::string::npos;
    };
    CliArgs a;
    a = embed_args(); a.chat_mode = true;                 EXPECT_TRUE(refusal_names(a, "--chat"));
    a = embed_args(); a.image_embed_cache_dir = "c";      EXPECT_TRUE(refusal_names(a, "--image-embed-cache"));
    a = embed_args(); a.grammar_file = "g.gbnf";          EXPECT_TRUE(refusal_names(a, "--grammar-file"));
    a = embed_args(); a.speculative = true;               EXPECT_TRUE(refusal_names(a, "--speculative"));
    a = embed_args(); a.save_session_path = "s.bin";      EXPECT_TRUE(refusal_names(a, "--save-session"));
    a = embed_args(); a.persistent_graph = true;          EXPECT_TRUE(refusal_names(a, "--persistent-graph"));
    a = embed_args(); a.kv_type = GGML_TYPE_F16;          EXPECT_TRUE(refusal_names(a, "--kv-type"));
    a = embed_args(); a.system_prompt = "be brief";       EXPECT_TRUE(refusal_names(a, "--system-prompt"));
}

// ── Input file ───────────────────────────────────────────────────────────────

TEST(EmbedMode, ReadsOneTextPerLineAndDropsCr) {
    std::istringstream in("first text\r\nsecond text\n");
    EXPECT_EQ(read_embed_inputs(in, "t.txt"),
              (std::vector<std::string>{"first text", "second text"}));
}

TEST(EmbedMode, RefusesAnEmptyLineNamingIt) {
    std::istringstream in("one\n\nthree\n");
    try {
        read_embed_inputs(in, "t.txt");
        FAIL() << "expected a throw";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("'t.txt' line 2"), std::string::npos) << e.what();
    }
}

TEST(EmbedMode, RefusesAnEmptyInput) {
    std::istringstream in("");
    EXPECT_THROW(read_embed_inputs(in, "t.txt"), std::runtime_error);
}

// ── JSONL record ─────────────────────────────────────────────────────────────

TEST(EmbedMode, RecordIsOneJsonObjectThatRoundTripsFloats) {
    const std::vector<float> v{0.1f, -1.0f / 3.0f, 1e-8f};
    const std::string line = format_embedding_line(7, 12, v);
    EXPECT_EQ(line.rfind("{\"index\":7,\"n_tokens\":12,\"dim\":3,\"embedding\":[", 0), 0u) << line;
    EXPECT_EQ(line.back(), '}');
    EXPECT_EQ(line.find('\n'), std::string::npos);
    EXPECT_EQ(parse_vector(line), v);   // %.9g is exact for float
}

// ── End to end ───────────────────────────────────────────────────────────────

TEST(EmbedMode, EndToEndMatchesLlamaCpp) {
    const char* model_path = std::getenv("EMBEDDINGGEMMA2_MODEL_PATH");
    if (!model_path || !*model_path) GTEST_SKIP() << "EMBEDDINGGEMMA2_MODEL_PATH not set";

    // The first four reference prompts and their vectors.
    std::ifstream ref(QINF_TEST_FIXTURES_DIR "/embed/embeddinggemma2_llamacpp.txt");
    ASSERT_TRUE(ref);
    std::vector<std::string> texts;
    std::vector<std::vector<float>> want;
    std::string line;
    while (std::getline(ref, line) && want.size() < 4) {
        if (line.rfind("text ", 0) == 0) texts.push_back(line.substr(5));
        if (line.rfind("embedding ", 0) == 0) {
            std::istringstream ss(line.substr(10));
            size_t n; ss >> n;
            std::vector<float> v(n);
            for (auto& x : v) ss >> x;
            want.push_back(v);
        }
    }
    texts.resize(want.size());

    const std::string dir = ::testing::TempDir();
    const std::string in_path = dir + "embed_mode_in.txt", out_path = dir + "embed_mode_out.jsonl";
    {
        std::ofstream in(in_path);
        for (const auto& t : texts) in << t << "\n";
    }

    register_builtin_models();
    Model model;
    model.load_metadata(model_path);
    model.load_tensors();
    CliArgs args;
    args.embed = true;
    args.embed_in_path = in_path;
    args.embed_out_path = out_path;
    ASSERT_EQ(embed_mode_refusal(args), "");
    ASSERT_EQ(run_embed(model, args), 0);

    std::ifstream out(out_path);
    std::vector<std::string> lines;
    while (std::getline(out, line)) lines.push_back(line);
    ASSERT_EQ(lines.size(), want.size());
    for (size_t i = 0; i < lines.size(); ++i) {
        EXPECT_EQ(lines[i].rfind("{\"index\":" + std::to_string(i) + ",", 0), 0u);
        const std::vector<float> got = parse_vector(lines[i]);
        ASSERT_EQ(got.size(), 768u);
        EXPECT_GE(cosine(got, want[i]), 0.9999) << "text " << i;
    }
    std::remove(in_path.c_str());
    std::remove(out_path.c_str());
}

// A generation model is refused by name, and nothing is written.
TEST(EmbedMode, RefusesAGenerationModel) {
    const char* model_path = std::getenv("QWEN3_MODEL_PATH");
    if (!model_path || !*model_path) GTEST_SKIP() << "QWEN3_MODEL_PATH not set";
    register_builtin_models();
    Model model;
    model.load_metadata(model_path);
    model.load_tensors();
    const std::string out_path = ::testing::TempDir() + "embed_mode_refused.jsonl";
    std::remove(out_path.c_str());
    CliArgs args = embed_args();
    args.embed_out_path = out_path;
    EXPECT_EQ(run_embed(model, args), 1);
    EXPECT_FALSE(std::ifstream(out_path).good());
}

// One fixture image through the CLI path, against llama.cpp's pooled vector
// (tanh reference, plan phase 4).
TEST(EmbedMode, EndToEndImageMatchesLlamaCpp) {
    const char* model_path = std::getenv("EMBEDDINGGEMMA2_MODEL_PATH");
    if (!model_path || !*model_path) GTEST_SKIP() << "EMBEDDINGGEMMA2_MODEL_PATH not set";
    const char* e = std::getenv("EMBEDDINGGEMMA2_MMPROJ_PATH");
    const std::string mm = (e && *e) ? e : "models/embeddinggemma-2-mmproj-BF16.gguf";
    if (!std::ifstream(mm).good()) GTEST_SKIP() << "mmproj not found at " << mm;

    std::ifstream ref(QINF_TEST_FIXTURES_DIR "/embed_image/reference_t280_gelu.txt");
    ASSERT_TRUE(ref);
    std::vector<float> want;
    std::string line;
    bool in_it = false;
    while (std::getline(ref, line)) {
        if (line == "image checker_512") in_it = true;
        else if (line.rfind("image ", 0) == 0 || line.rfind("caption ", 0) == 0) in_it = false;
        else if (in_it && line.rfind("embedding ", 0) == 0) {
            std::istringstream ss(line.substr(10));
            size_t n; ss >> n;
            want.resize(n);
            for (auto& x : want) ss >> x;
        }
    }
    ASSERT_EQ(want.size(), 768u);

    register_builtin_models();
    Model model;
    model.load_metadata(model_path);
    model.load_tensors();
    const std::string out_path = ::testing::TempDir() + "embed_mode_image.jsonl";
    CliArgs args;
    args.embed = true;
    args.image_path = QINF_TEST_FIXTURES_DIR "/embed_image/checker_512.png";
    args.mmproj_path = mm;
    args.embed_out_path = out_path;
    ASSERT_EQ(embed_mode_refusal(args), "");
    ASSERT_EQ(run_embed(model, args), 0);
    std::ifstream out(out_path);
    std::getline(out, line);
    EXPECT_EQ(line.rfind("{\"index\":0,\"n_tokens\":125,", 0), 0u) << line.substr(0, 80);
    EXPECT_GE(cosine(parse_vector(line), want), 0.9999);
    std::remove(out_path.c_str());
}

