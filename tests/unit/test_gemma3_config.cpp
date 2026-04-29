// test_gemma3_config.cpp
//
// Unit tests for Gemma 3 config and layer infrastructure (PRs G3.1, G3.2, G3.5).
//
// Three concerns:
//
// 1. 5:1 sliding-window alternation (PR G3.5):
//    Gemma3Config::from_metadata must produce:
//      layer i local  (window = sliding_window) when (i % 6 != 5)
//      layer i global (window = 0)              when (i % 6 == 5)
//    Verified against the known pattern for 26 layers (Gemma 3 1B).
//
// 2. Per-layer RoPE base frequency (PR G3.2):
//    layer_rope_base[i] == local_rope_base  when layer i is local
//    layer_rope_base[i] == global_rope_base when layer i is global
//    global_rope_base comes from GGUF (gemma3.rope.freq_base),
//    local_rope_base is the hardcoded 10000.0f.
//
// 3. QK-norm weight dimensionality (PR G3.1):
//    build_rms_norm applied to a [head_dim, n_head, n_tokens] tensor with a
//    [head_dim] weight broadcasts the weight across all heads — each head is
//    normalised independently but uses the same weight vector (Gemma 3 style).
//    The operation is identical to the Qwen 3 call site; this test confirms the
//    shared weight semantics are correct so no broadcast variant is needed.
//
// No model file required — all tests use synthetic metadata / config.

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

#include "../../src/core/model.h"
#include "../../src/models/gemma3.h"

// ── Helpers ───────────────────────────────────────────────────────────────────

static ModelMetadata make_gemma3_meta(uint32_t n_layers,
                                       uint32_t sliding_window,
                                       float    global_rope_base = 1000000.0f)
{
    ModelMetadata m;
    m.architecture             = "gemma3";
    m.block_count              = n_layers;
    m.attention_head_count     = 4;
    m.attention_head_count_kv  = 1;
    m.attention_key_length     = 256;
    m.attention_value_length   = 256;
    m.embedding_length         = 1152;
    m.context_length           = 32768;
    m.rms_norm_eps             = 1e-6f;
    m.rope_freq_base           = global_rope_base;

    m.raw_kv.set("gemma3.attention.sliding_window", sliding_window);
    return m;
}

// ── PR G3.5: 5:1 alternation pattern ─────────────────────────────────────────

// For 26 layers, global layers are at indices where (i % 6 == 5): 5,11,17,23.
TEST(Gemma3Config, LayerWindowFiveOnePattern) {
    const uint32_t n_layers = 26;
    const uint32_t window   = 512;

    auto cfg = Gemma3Config::from_metadata(make_gemma3_meta(n_layers, window));

    ASSERT_EQ(cfg.layer_window.size(), (size_t)n_layers);

    for (uint32_t i = 0; i < n_layers; ++i) {
        if (i % 6 == 5) {
            EXPECT_EQ(cfg.layer_window[i], 0u)
                << "layer " << i << " should be global (window=0)";
        } else {
            EXPECT_EQ(cfg.layer_window[i], window)
                << "layer " << i << " should be local (window=" << window << ")";
        }
    }
}

// Spot-check the exact global layer indices for the 1B checkpoint (26 layers).
TEST(Gemma3Config, GlobalLayerIndices26Layers) {
    auto cfg = Gemma3Config::from_metadata(make_gemma3_meta(26, 512));

    // Global layers: 5, 11, 17, 23
    for (uint32_t i : {5u, 11u, 17u, 23u}) {
        EXPECT_EQ(cfg.layer_window[i], 0u) << "layer " << i << " must be global";
    }
    // A sampling of local layers: 0, 4, 6, 10, 24, 25
    for (uint32_t i : {0u, 4u, 6u, 10u, 24u, 25u}) {
        EXPECT_GT(cfg.layer_window[i], 0u) << "layer " << i << " must be local";
    }
}

// Sliding window value is carried through correctly.
TEST(Gemma3Config, SlidingWindowValueFromMetadata) {
    auto cfg = Gemma3Config::from_metadata(make_gemma3_meta(6, 1024));
    EXPECT_EQ(cfg.sliding_window, 1024u);
}

// 6-layer model: exactly one global layer (index 5).
TEST(Gemma3Config, SixLayerModelExactlyOneGlobal) {
    auto cfg = Gemma3Config::from_metadata(make_gemma3_meta(6, 512));
    uint32_t global_count = 0;
    for (uint32_t i = 0; i < 6; ++i) {
        if (cfg.layer_window[i] == 0) ++global_count;
    }
    EXPECT_EQ(global_count, 1u);
    EXPECT_EQ(cfg.layer_window[5], 0u);
}

// 12-layer model: two global layers (indices 5 and 11).
TEST(Gemma3Config, TwelveLayerModelTwoGlobals) {
    auto cfg = Gemma3Config::from_metadata(make_gemma3_meta(12, 512));
    EXPECT_EQ(cfg.layer_window[5],  0u);
    EXPECT_EQ(cfg.layer_window[11], 0u);
    for (uint32_t i = 0; i < 12; ++i) {
        if (i != 5 && i != 11) {
            EXPECT_GT(cfg.layer_window[i], 0u) << "layer " << i;
        }
    }
}

// ── PR G3.2: per-layer RoPE base frequency ────────────────────────────────────

// Local layers get local_rope_base (10K); global layers get global_rope_base (1M).
TEST(Gemma3Config, LayerRopeBaseFrequencyAssignment) {
    const float global_base = 1000000.0f;
    auto cfg = Gemma3Config::from_metadata(make_gemma3_meta(26, 512, global_base));

    ASSERT_EQ(cfg.layer_rope_base.size(), 26u);

    for (uint32_t i = 0; i < 26; ++i) {
        if (i % 6 == 5) {
            EXPECT_FLOAT_EQ(cfg.layer_rope_base[i], global_base)
                << "global layer " << i << " must use global_rope_base";
        } else {
            EXPECT_FLOAT_EQ(cfg.layer_rope_base[i], cfg.local_rope_base)
                << "local layer " << i << " must use local_rope_base";
        }
    }
}

// local_rope_base is hardcoded to 10000.0f regardless of GGUF content.
TEST(Gemma3Config, LocalRopeBaseIsHardcoded10K) {
    auto cfg = Gemma3Config::from_metadata(make_gemma3_meta(6, 512, 500000.0f));
    EXPECT_FLOAT_EQ(cfg.local_rope_base, 10000.0f);
}

// global_rope_base comes from GGUF rope_freq_base.
TEST(Gemma3Config, GlobalRopeBaseFromGGUF) {
    const float base = 2000000.0f;
    auto cfg = Gemma3Config::from_metadata(make_gemma3_meta(6, 512, base));
    EXPECT_FLOAT_EQ(cfg.global_rope_base, base);
}

// When GGUF rope_freq_base is 0 or absent, global base defaults to 1e6.
TEST(Gemma3Config, GlobalRopeBaseDefaultsToOneMillion) {
    auto meta = make_gemma3_meta(6, 512);
    meta.rope_freq_base = 0.0f;  // simulate missing key
    auto cfg = Gemma3Config::from_metadata(meta);
    EXPECT_FLOAT_EQ(cfg.global_rope_base, 1000000.0f);
}

// ── PR G3.1: QK-norm weight dimensionality ────────────────────────────────────
//
// Gemma 3 uses a [head_dim]-shaped RMS-norm weight for both Q and K norms.
// The weight is applied to each [head_dim] row of the [head_dim, n_head, n_tokens]
// tensor, broadcasting across all heads. This is the same call as Qwen 3.
//
// We verify the semantics here without needing a model file: run build_rms_norm
// on a synthetic 2-head tensor and confirm that both heads receive the same
// normalisation weight, and that the result is numerically correct.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../../src/layers/norm.h"

class Gemma3QKNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        ASSERT_NE(backend_, nullptr);
    }
    void TearDown() override {
        if (backend_) ggml_backend_free(backend_);
    }
    ggml_backend_t backend_ = nullptr;
};

// A [head_dim=4, n_head=2, n_tokens=1] tensor normed with a [4] weight.
// Both heads must be normalised by the same weight — Gemma 3 "shared across heads".
TEST_F(Gemma3QKNormTest, BroadcastWeightAcrossHeads) {
    const int head_dim = 4;
    const int n_head   = 2;
    const int n_tokens = 1;
    const float eps    = 1e-6f;

    // head 0: [1, 2, 3, 4], head 1: [4, 3, 2, 1] (different inputs, same weight)
    std::vector<float> input = {1.f, 2.f, 3.f, 4.f,   // head 0
                                 4.f, 3.f, 2.f, 1.f};  // head 1
    // weight w = [1, 1, 1, 1] (identity) so output = x / rms(x)
    std::vector<float> weight(head_dim, 1.0f);

    const size_t ctx_bytes = 256 * 1024;
    ggml_init_params p{ctx_bytes, nullptr, true};
    ggml_context* ctx = ggml_init(p);
    ASSERT_NE(ctx, nullptr);

    ggml_cgraph* gf = ggml_new_graph(ctx);

    // Build [head_dim, n_head, n_tokens] input tensor.
    ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                         head_dim, n_head, n_tokens);
    ggml_set_input(x);
    ggml_tensor* w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
    ggml_set_input(w);

    ggml_tensor* out = build_rms_norm(ctx, x, w, eps, /*il=*/0);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend_));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(x, input.data(),  0, input.size()  * sizeof(float));
    ggml_backend_tensor_set(w, weight.data(), 0, weight.size() * sizeof(float));
    ggml_backend_graph_compute(backend_, gf);

    std::vector<float> result(head_dim * n_head * n_tokens);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    // Oracle: each 4-element row is divided by its own RMS, then multiplied by weight=1.
    auto rms = [](const float* v, int n) {
        float sum = 0.f;
        for (int i = 0; i < n; ++i) sum += v[i] * v[i];
        return std::sqrt(sum / n + 1e-6f);
    };

    // head 0
    const float rms0 = rms(input.data(),     head_dim);
    for (int i = 0; i < head_dim; ++i) {
        EXPECT_NEAR(result[i], input[i] / rms0, 1e-5f)
            << "head 0, elem " << i;
    }
    // head 1
    const float rms1 = rms(input.data() + head_dim, head_dim);
    for (int i = 0; i < head_dim; ++i) {
        EXPECT_NEAR(result[head_dim + i], input[head_dim + i] / rms1, 1e-5f)
            << "head 1, elem " << i;
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

// Scalar weight w=2 scales both heads by the same factor.
TEST_F(Gemma3QKNormTest, SharedWeightScalesBothHeadsEqually) {
    const int head_dim = 2;
    const int n_head   = 3;
    const float eps    = 1e-6f;

    // All three heads have the same input so their RMS is identical.
    std::vector<float> input(head_dim * n_head);
    for (int h = 0; h < n_head; ++h) {
        input[h * head_dim + 0] = 3.0f;
        input[h * head_dim + 1] = 4.0f;
    }
    std::vector<float> weight = {2.0f, 2.0f};  // scale by 2

    const size_t ctx_bytes = 128 * 1024;
    ggml_init_params p{ctx_bytes, nullptr, true};
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf = ggml_new_graph(ctx);

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, n_head);
    ggml_set_input(x);
    ggml_tensor* w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
    ggml_set_input(w);
    ggml_tensor* out = build_rms_norm(ctx, x, w, eps, /*il=*/1);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend_));
    ggml_gallocr_alloc_graph(alloc, gf);
    ggml_backend_tensor_set(x, input.data(),  0, input.size()  * sizeof(float));
    ggml_backend_tensor_set(w, weight.data(), 0, weight.size() * sizeof(float));
    ggml_backend_graph_compute(backend_, gf);

    std::vector<float> result(head_dim * n_head);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    // RMS([3,4]) = sqrt((9+16)/2 + eps) ≈ sqrt(12.5) ≈ 3.5355
    const float rms_val = std::sqrt(0.5f * (9.f + 16.f) + eps);
    for (int h = 0; h < n_head; ++h) {
        EXPECT_NEAR(result[h * head_dim + 0], 2.0f * 3.0f / rms_val, 1e-4f)
            << "head " << h << " elem 0";
        EXPECT_NEAR(result[h * head_dim + 1], 2.0f * 4.0f / rms_val, 1e-4f)
            << "head " << h << " elem 1";
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}
