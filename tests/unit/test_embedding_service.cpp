// test_embedding_service.cpp — co-located test for
// src/server/embedding_service.cpp (POST /v1/embeddings).
//
// Handler (always): the OpenAI contract through fake tokenize/embed callbacks —
//   shapes, refusals, base64, and that a refused request runs no embedding.
// Service (EMBEDDINGGEMMA2_MODEL_PATH): the loaded model end to end, matching
//   llama.cpp's reference vectors (tests/fixtures/embed/) and decoding base64
//   back to the float reply bit for bit.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "embedding_service.h"

using json = nlohmann::json;
using qinf::server::HttpReply;
using qinf::server::handle_embeddings;
using qinf::server::base64_f32;

namespace {

// Fake model: a text's tokens are its characters plus BOS/EOS; its vector is
// [len, 1, 0] normalised. Counts embed calls.
struct Fake {
    int embeds = 0;
    qinf::server::TokenizeFn tokenize = [](const std::string& t) {
        std::vector<int32_t> ids{2};
        for (char c : t) ids.push_back((int32_t)c);
        ids.push_back(1);
        return ids;
    };
    qinf::server::EmbedFn embed = [this](const std::vector<int32_t>& ids) {
        ++embeds;
        const float a = (float)ids.size(), n = std::sqrt(a * a + 1.0f);
        return std::vector<float>{a / n, 1.0f / n, 0.0f};
    };
    HttpReply call(const json& body, uint32_t max_tokens = 64) {
        return handle_embeddings(body.dump(), "fake-model", 3, max_tokens, tokenize, embed);
    }
};

std::vector<float> decode_b64_f32(const std::string& s) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        return c == '+' ? 62 : c == '/' ? 63 : -1;
    };
    std::vector<unsigned char> bytes;
    for (size_t i = 0; i + 3 < s.size(); i += 4) {
        const int a = val(s[i]), b = val(s[i + 1]), c = val(s[i + 2]), d = val(s[i + 3]);
        bytes.push_back((unsigned char)((a << 2) | (b >> 4)));
        if (s[i + 2] != '=') bytes.push_back((unsigned char)(((b & 15) << 4) | (c >> 2)));
        if (s[i + 3] != '=') bytes.push_back((unsigned char)(((c & 3) << 6) | d));
    }
    std::vector<float> v(bytes.size() / sizeof(float));
    std::memcpy(v.data(), bytes.data(), v.size() * sizeof(float));
    return v;
}

}  // namespace

// ── Handler: the contract ────────────────────────────────────────────────────

TEST(EmbeddingsHandler, StringInputGivesOneOpenAiShapedItem) {
    Fake f;
    const HttpReply r = f.call({{"input", "abc"}, {"model", "anything"}});
    ASSERT_EQ(r.status, 200) << r.body.dump();
    EXPECT_EQ(r.body["object"], "list");
    EXPECT_EQ(r.body["model"], "fake-model");
    ASSERT_EQ(r.body["data"].size(), 1u);
    EXPECT_EQ(r.body["data"][0]["object"], "embedding");
    EXPECT_EQ(r.body["data"][0]["index"], 0);
    EXPECT_EQ(r.body["data"][0]["embedding"].size(), 3u);
    EXPECT_EQ(r.body["usage"]["prompt_tokens"], 5);   // BOS + 3 + EOS
    EXPECT_EQ(r.body["usage"]["total_tokens"], 5);
}

TEST(EmbeddingsHandler, ArrayInputKeepsOrderAndSumsUsage) {
    Fake f;
    const HttpReply r = f.call({{"input", {"a", "bbbb", "cc"}}});
    ASSERT_EQ(r.status, 200);
    ASSERT_EQ(r.body["data"].size(), 3u);
    for (size_t i = 0; i < 3; ++i) EXPECT_EQ(r.body["data"][i]["index"], i);
    EXPECT_EQ(r.body["usage"]["prompt_tokens"], 3 + 6 + 4);
    // Different lengths → different fake vectors, in request order.
    EXPECT_NE(r.body["data"][0]["embedding"], r.body["data"][1]["embedding"]);
}

TEST(EmbeddingsHandler, Base64IsTheFloatBytes) {
    Fake f;
    const HttpReply flt = f.call({{"input", "abc"}});
    const HttpReply b64 = f.call({{"input", "abc"}, {"encoding_format", "base64"}});
    ASSERT_EQ(b64.status, 200);
    ASSERT_TRUE(b64.body["data"][0]["embedding"].is_string());
    EXPECT_EQ(decode_b64_f32(b64.body["data"][0]["embedding"].get<std::string>()),
              flt.body["data"][0]["embedding"].get<std::vector<float>>());
}

TEST(EmbeddingsHandler, Base64PadsEveryLength) {
    // 1, 2 and 3 floats = 4, 8, 12 bytes: remainders 1, 2, 0 mod 3.
    EXPECT_EQ(base64_f32({1.0f}), "AACAPw==");
    EXPECT_EQ(base64_f32({1.0f, 1.0f}), "AACAPwAAgD8=");
    EXPECT_EQ(base64_f32({1.0f, 1.0f, 1.0f}), "AACAPwAAgD8AAIA/");
}

TEST(EmbeddingsHandler, DimensionsEqualToTheModelIsAccepted) {
    Fake f;
    EXPECT_EQ(f.call({{"input", "a"}, {"dimensions", 3}}).status, 200);
}

// ── Handler: refusals ────────────────────────────────────────────────────────

TEST(EmbeddingsHandler, RefusesMalformedRequestsWithoutEmbedding) {
    Fake f;
    const json bad[] = {
        json{{"model", "x"}},                          // no input
        json{{"input", 42}},                           // not text
        json{{"input", json::array()}},                // empty array
        json{{"input", {"ok", ""}}},                   // empty text
        json{{"input", {1, 2, 3}}},                    // token ids
        json{{"input", {{1, 2}, {3}}}},                // token-id batches
        json{{"input", {"ok", 7}}},                    // mixed
        json{{"input", "a"}, {"encoding_format", "int8"}},
        json{{"input", "a"}, {"dimensions", 2}},       // no truncation
    };
    for (const json& b : bad) {
        const HttpReply r = f.call(b);
        EXPECT_EQ(r.status, 400) << b.dump();
        EXPECT_TRUE(r.body.contains("error")) << b.dump();
    }
    EXPECT_EQ(handle_embeddings("{not json", "m", 3, 64, f.tokenize, f.embed).status, 400);
    EXPECT_EQ(f.embeds, 0);
}

// One over-long text refuses the whole request, by index, before any compute.
TEST(EmbeddingsHandler, RefusesATextOverTheTokenBudgetByIndex) {
    Fake f;
    const HttpReply r = f.call({{"input", {"short", std::string(100, 'x')}}}, /*max_tokens=*/64);
    EXPECT_EQ(r.status, 413);
    EXPECT_NE(r.body["error"].get<std::string>().find("'input[1]'"), std::string::npos)
        << r.body.dump();
    EXPECT_EQ(f.embeds, 0);
}

TEST(EmbeddingsHandler, RefusesTooManyTexts) {
    Fake f;
    const HttpReply r = f.call({{"input", std::vector<std::string>(qinf::server::kMaxInputs + 1, "a")}});
    EXPECT_EQ(r.status, 413);
    EXPECT_EQ(f.embeds, 0);
}

TEST(EmbeddingsHandler, AnEngineFailureIsA500NotACrash) {
    Fake f;
    f.embed = [](const std::vector<int32_t>&) -> std::vector<float> {
        throw std::runtime_error("compute failed");
    };
    const HttpReply r = f.call({{"input", "a"}});
    EXPECT_EQ(r.status, 500);
    EXPECT_NE(r.body["error"].get<std::string>().find("compute failed"), std::string::npos);
}

// ── Service: the real model ──────────────────────────────────────────────────

TEST(EmbeddingService, MatchesLlamaCppAndBase64RoundTrips) {
    const char* path = std::getenv("EMBEDDINGGEMMA2_MODEL_PATH");
    if (!path || !*path) GTEST_SKIP() << "EMBEDDINGGEMMA2_MODEL_PATH not set";

    std::ifstream ref(QINF_TEST_FIXTURES_DIR "/embed/embeddinggemma2_llamacpp.txt");
    ASSERT_TRUE(ref);
    std::vector<std::string> texts;
    std::vector<std::vector<float>> want;
    std::string line;
    while (std::getline(ref, line) && want.size() < 3) {
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

    qinf::server::EmbeddingService svc(path, 4096, /*flash=*/false);
    EXPECT_EQ(svc.dim(), 768u);
    EXPECT_EQ(svc.model_id(), "embeddinggemma-2-BF16");

    const HttpReply r = svc.handle(json{{"input", texts}}.dump());
    ASSERT_EQ(r.status, 200) << r.body.dump();
    ASSERT_EQ(r.body["data"].size(), want.size());
    for (size_t i = 0; i < want.size(); ++i) {
        const auto got = r.body["data"][i]["embedding"].get<std::vector<float>>();
        double d = 0, ng = 0, nw = 0;
        for (size_t k = 0; k < got.size(); ++k) {
            d += (double)got[k] * want[i][k]; ng += (double)got[k] * got[k]; nw += (double)want[i][k] * want[i][k];
        }
        EXPECT_GE(d / std::sqrt(ng * nw), 0.9999) << "text " << i;
    }
    EXPECT_EQ(r.body["usage"]["prompt_tokens"], 18 + 23 + 18);   // fixture token counts

    const HttpReply b = svc.handle(json{{"input", texts[0]}, {"encoding_format", "base64"}}.dump());
    ASSERT_EQ(b.status, 200);
    EXPECT_EQ(decode_b64_f32(b.body["data"][0]["embedding"].get<std::string>()),
              r.body["data"][0]["embedding"].get<std::vector<float>>());
    EXPECT_EQ(svc.requests_served(), 2u);
}

TEST(EmbeddingService, RefusesAGenerationModel) {
    const char* path = std::getenv("QWEN3_MODEL_PATH");
    if (!path || !*path) GTEST_SKIP() << "QWEN3_MODEL_PATH not set";
    EXPECT_THROW(qinf::server::EmbeddingService(path, 512, false), std::runtime_error);
}
