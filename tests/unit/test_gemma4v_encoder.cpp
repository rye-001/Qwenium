// test_gemma4v_encoder.cpp — the gemma4v ViT (EmbeddingGemma-2 vision) against
// llama.cpp ceefd16da (docs/plan-embed-image.md phase 3).
//
// The reference is the encoder output llama-server dumped for each of the seven
// synthetic fixture images (tests/fixtures/embed_image/<name>.t280.soft.bin,
// MTMD_DEBUG_EMBEDDINGS). Our preprocessing is byte-identical to theirs (phase
// 2), so this isolates the encoder. Bar (proposed in the plan): rel-L2 ≤ 5e-3
// per image — not bit-identical, because encoder attention is always flash
// attention here (architecture §7).
//
// llama.cpp runs this file's FFN with GELU-quick (the mmproj has no
// clip.use_gelu key); the model's config.json says tanh GELU. The parity test
// therefore forces GeluQuick: it gates everything ELSE against the reference.
// The tanh default is a separate, user-decided question.
//
// Needs models/embeddinggemma-2-mmproj-BF16.gguf (or
// EMBEDDINGGEMMA2_MMPROJ_PATH); skips without it. Metal by default when built
// with it, QINF_TEST_CPU=1 for the CPU backend.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef QINF_TEST_HAS_METAL
#include "ggml-metal.h"
#endif

#include "../../src/image/image_loader.h"
#include "../../src/vision/bitmap.h"
#include "../../src/vision/gemma4v_encoder.h"
#include "../../src/vision/image_preprocess.h"
#include "../../src/vision/vision_loader.h"
#include "../../src/vision/vision_model.h"

namespace {

using qinf::vision::Gemma4vEncoder;
using qinf::vision::VisionFfnAct;
using qinf::vision::VisionLoader;
using qinf::vision::VisionModel;

std::string mmproj_path() {
    if (const char* e = std::getenv("EMBEDDINGGEMMA2_MMPROJ_PATH"); e && e[0]) return e;
    return "./models/embeddinggemma-2-mmproj-BF16.gguf";
}

std::string fixture_dir() {
    if (const char* e = std::getenv("QINF_EMBED_IMAGE_FIXTURE_DIR"); e && e[0]) return e;
    for (const char* d : {"tests/fixtures/embed_image", "../tests/fixtures/embed_image",
                          "../../tests/fixtures/embed_image"}) {
        std::ifstream f(std::string(d) + "/captions.txt");
        if (f.good()) return d;
    }
    return {};
}

#define SKIP_IF_NO_INPUTS()                                                          \
    do {                                                                             \
        FILE* _f = std::fopen(mmproj_path().c_str(), "rb");                          \
        if (!_f) GTEST_SKIP() << "mmproj not found at " << mmproj_path()             \
                              << " (set EMBEDDINGGEMMA2_MMPROJ_PATH)";               \
        std::fclose(_f);                                                             \
        if (fixture_dir().empty()) GTEST_SKIP() << "embed_image fixtures not found"; \
    } while (0)

// Metal by default when built with it: the reference ran on Metal and so does
// production, and there the encoder is BIT-identical to llama.cpp (rel-L2 0 on
// all seven images, 2026-10-16). QINF_TEST_CPU=1 forces the CPU backend, where
// BF16 matmuls round activations to BF16 and the worst image reads 5.4e-3 —
// over this bar; the CPU path is reported, not gated (plan §7 phase 3).
ggml_backend_t make_backend() {
#ifdef QINF_TEST_HAS_METAL
    if (const char* e = std::getenv("QINF_TEST_CPU"); !(e && e[0])) return ggml_backend_metal_init();
#endif
    return ggml_backend_cpu_init();
}

struct Fixture {
    VisionModel    model;
    VisionLoader   loader;
    ggml_backend_t backend = nullptr;
    Fixture() {
        backend = make_backend();
        loader.parse_metadata(mmproj_path(), model);
        loader.load_tensors(model, backend);
    }
    ~Fixture() { if (backend) ggml_backend_free(backend); }
};

const char* kImages[] = {"red_circle_224", "blue_squares_640x480", "green_triangle_1200x300",
                         "stripes_300x1200", "page_850x1100", "yellow_tiny_40", "checker_512"};

std::vector<float> read_soft(const std::string& path, int32_t& n_tok, int32_t& n_embd) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) throw std::runtime_error("cannot open " + path);
    f.read(reinterpret_cast<char*>(&n_tok), 4);
    f.read(reinterpret_cast<char*>(&n_embd), 4);
    std::vector<float> v(static_cast<size_t>(n_tok) * n_embd);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
    if (static_cast<size_t>(f.gcount()) != v.size() * 4) throw std::runtime_error("truncated " + path);
    return v;
}

struct Diff { double rel_l2 = 0, min_cos = 1; };

Diff compare(const std::vector<float>& got, const std::vector<float>& want, int64_t dim) {
    Diff d;
    double num = 0, den = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double e = static_cast<double>(got[i]) - want[i];
        num += e * e;
        den += static_cast<double>(want[i]) * want[i];
    }
    d.rel_l2 = std::sqrt(num / den);
    for (size_t t = 0; t < want.size() / dim; ++t) {
        double ab = 0, aa = 0, bb = 0;
        for (int64_t j = 0; j < dim; ++j) {
            const double a = got[t * dim + j], b = want[t * dim + j];
            ab += a * b; aa += a * a; bb += b * b;
        }
        d.min_cos = std::min(d.min_cos, ab / std::sqrt(aa * bb));
    }
    return d;
}

}  // namespace

TEST(Gemma4vEncoder, LoaderReadsTheTowerAndSkipsAudio) {
    SKIP_IF_NO_INPUTS();
    Fixture fx;
    const auto& c = fx.model.config();
    EXPECT_EQ(c.projector_type, qinf::vision::VisionProjectorType::Gemma4V);
    EXPECT_EQ(c.hidden_size, 768u);
    EXPECT_EQ(c.num_layers, 16u);
    EXPECT_EQ(c.num_attn_heads, 12u);
    EXPECT_EQ(c.intermediate_size, 3072u);
    EXPECT_EQ(c.projection_dim, 512u);
    EXPECT_EQ(c.n_merge, 3u);
    EXPECT_EQ(c.ffn_act, VisionFfnAct::GeluTanh) << "config.json: gelu_pytorch_tanh (user decision)";
    // 16 blocks × 13 tensors + patch_embd + position_embd + mm.input_projection.
    EXPECT_EQ(fx.model.tensors().size(), 16u * 13u + 3u);
    for (const auto& [name, t] : fx.model.tensors()) {
        (void)t;
        EXPECT_TRUE(name.rfind("v.", 0) == 0 || name == "mm.input_projection.weight")
            << "loaded a non-vision tensor: " << name;
    }
}

// One parity run: our encoder with `act` against the soft tokens llama.cpp
// dumped with the matching activation (`suffix` "" = GELU-quick, the unpatched
// mmproj; ".gelu" = tanh, the scratch mmproj copy with clip.use_gelu = true).
void check_parity(VisionFfnAct act, const std::string& suffix) {
    Fixture fx;
    fx.model.config().ffn_act = act;
    Gemma4vEncoder enc(fx.model, fx.backend, 512);
    const std::string dir = fixture_dir();
    for (const char* name : kImages) {
        SCOPED_TRACE(name);
        const auto bmp = qinf::image::load_image_to_bitmap(dir + "/" + name + ".png",
                                                           qinf::vision::gemma4v_preprocess());
        int32_t n_tok = 0, n_embd = 0;
        const auto want = read_soft(dir + "/" + name + ".t280" + suffix + ".soft.bin", n_tok, n_embd);
        ASSERT_EQ(n_embd, 512);
        ASSERT_EQ(enc.mm_tokens_for(bmp), static_cast<uint32_t>(n_tok));
        const auto got = enc.encode(bmp);
        ASSERT_EQ(got.size(), want.size());
        const Diff d = compare(got, want, n_embd);
        std::printf("    %-24s tokens %4d  rel-L2 %.3e  min token cosine %.6f\n", name, n_tok, d.rel_l2, d.min_cos);
        EXPECT_LE(d.rel_l2, 5e-3);
    }
}

TEST(Gemma4vEncoder, MatchesLlamaCppSoftTokens) {
    SKIP_IF_NO_INPUTS();
    check_parity(VisionFfnAct::GeluQuick, "");
}

TEST(Gemma4vEncoder, MatchesLlamaCppSoftTokensTanhGelu) {
    SKIP_IF_NO_INPUTS();
    check_parity(VisionFfnAct::GeluTanh, ".gelu");
}

TEST(Gemma4vEncoder, RefusesAMisalignedBitmapAndAWrongTextWidth) {
    SKIP_IF_NO_INPUTS();
    Fixture fx;
    EXPECT_THROW(Gemma4vEncoder(fx.model, fx.backend, 768), std::runtime_error);
    Gemma4vEncoder enc(fx.model, fx.backend, 512);
    qinf::vision::Bitmap bmp;
    bmp.channels = 3;
    bmp.width = 50;    // not a multiple of 48
    bmp.height = 48;
    bmp.pixels.assign(3u * 50u * 48u, 0.5f);
    EXPECT_THROW(enc.encode(bmp), std::runtime_error);
}
