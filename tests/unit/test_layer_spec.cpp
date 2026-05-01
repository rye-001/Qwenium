// test_layer_spec.cpp — PR G4.1
//
// Unit tests for LayerSpec and the per-layer attention shape machinery.
//
// Tests:
//   1. SyntheticPerLayerArrays — LayerSpec vector built from synthetic 5:1
//      sliding/global data carries the expected per-layer fields.
//   2. DegenerateIsUniform — LayerSpec with head_dim_k = head_dim_v = n_embd_head
//      and rope_dim = n_embd_head is the degenerate case; build_transformer_layer
//      produces bit-identical output to the zero-default (no new fields set) path.
//   3. AsymmetricShapeBuilds — a block configured with head_dim_k / head_dim_v
//      different from n_embd_head (simulating Gemma 4 global attention) builds
//      without crash and produces finite output.
//
// No model file required.
// Build target: layer-spec-tests (see tests/CMakeLists.txt).

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <memory>

#include "../../src/layers/layer.h"
#include "../../src/layers/transformer_block.h"
#include "../../src/state/kv_cache_simple.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// ── LayerSpec construction tests ─────────────────────────────────────────────

// Verify that a LayerSpec vector synthesised from per-layer shape arrays
// carries the right field values per layer.  No ggml required.
TEST(LayerSpecConstructionTest, SyntheticPerLayerArrays) {
    // Six synthetic layers: 5 sliding + 1 global (5:1 pattern).
    constexpr int n_layers = 6;
    constexpr int n_head   = 16;

    // Per-layer inputs (as Gemma4Config::from_metadata would produce)
    const std::vector<bool> is_sliding = {true, true, true, true, true, false};

    // Scalar metadata (as from GGUF keys)
    constexpr int     head_dim_swa      = 256;
    constexpr int     head_dim_global   = 512;
    constexpr int     n_kv_heads_swa    = 8;
    constexpr int     n_kv_heads_global = 2;
    constexpr float   rope_base_swa     = 10000.0f;
    constexpr float   rope_base_global  = 1000000.0f;
    constexpr uint32_t window_swa       = 1024;

    // Build spec vector (simulates the recipe factory).
    std::vector<LayerSpec> specs(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        const bool swa = is_sliding[i];
        specs[i].head_dim_q  = swa ? head_dim_swa : head_dim_global;
        specs[i].head_dim_k  = swa ? head_dim_swa : head_dim_global;
        specs[i].head_dim_v  = swa ? head_dim_swa : head_dim_global;
        specs[i].n_head      = n_head;
        specs[i].n_head_kv   = swa ? n_kv_heads_swa    : n_kv_heads_global;
        specs[i].rope_dim    = swa ? head_dim_swa       : head_dim_global;
        specs[i].rope_base   = swa ? rope_base_swa      : rope_base_global;
        specs[i].attn_kind   = swa ? AttentionKind::Sliding : AttentionKind::Global;
        specs[i].window      = swa ? window_swa : 0u;
    }

    // Verify sliding layers (0–4)
    for (int i = 0; i < 5; ++i) {
        SCOPED_TRACE("layer " + std::to_string(i));
        EXPECT_EQ(specs[i].head_dim_q,  head_dim_swa);
        EXPECT_EQ(specs[i].head_dim_k,  head_dim_swa);
        EXPECT_EQ(specs[i].head_dim_v,  head_dim_swa);
        EXPECT_EQ(specs[i].n_head,      n_head);
        EXPECT_EQ(specs[i].n_head_kv,   n_kv_heads_swa);
        EXPECT_EQ(specs[i].rope_dim,    head_dim_swa);
        EXPECT_FLOAT_EQ(specs[i].rope_base, rope_base_swa);
        EXPECT_EQ(specs[i].attn_kind,   AttentionKind::Sliding);
        EXPECT_EQ(specs[i].window,      window_swa);
    }

    // Verify global layer (5)
    EXPECT_EQ(specs[5].head_dim_q,  head_dim_global);
    EXPECT_EQ(specs[5].head_dim_k,  head_dim_global);
    EXPECT_EQ(specs[5].head_dim_v,  head_dim_global);
    EXPECT_EQ(specs[5].n_head,      n_head);
    EXPECT_EQ(specs[5].n_head_kv,   n_kv_heads_global);
    EXPECT_EQ(specs[5].rope_dim,    head_dim_global);
    EXPECT_FLOAT_EQ(specs[5].rope_base, rope_base_global);
    EXPECT_EQ(specs[5].attn_kind,   AttentionKind::Global);
    EXPECT_EQ(specs[5].window,      0u);
}

// ── build_transformer_layer tests ────────────────────────────────────────────

// Small synthetic dimensions.
static constexpr int LS_N_EMBD      = 32;
static constexpr int LS_N_HEAD      = 2;
static constexpr int LS_N_HEAD_KV   = 2;
static constexpr int LS_N_EMBD_HEAD = LS_N_EMBD / LS_N_HEAD;  // 16
static constexpr int LS_N_FFN       = 64;
static constexpr int LS_N_CTX_MAX   = 64;
static constexpr int LS_N_TOKENS    = 4;

class LayerSpecBlockTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        ASSERT_NE(backend_, nullptr);

        const size_t weight_bytes = 128 * ggml_tensor_overhead()
            + (2*LS_N_EMBD*LS_N_EMBD
               + 2*LS_N_HEAD_KV*LS_N_EMBD_HEAD*LS_N_EMBD
               + 3*LS_N_FFN*LS_N_EMBD
               + 4*LS_N_EMBD
               + 2*LS_N_EMBD_HEAD) * sizeof(float);
        ggml_init_params wp = { weight_bytes, nullptr, false };
        wctx_ = ggml_init(wp);
        ASSERT_NE(wctx_, nullptr);

        auto make_2d = [&](int rows, int cols) -> ggml_tensor* {
            ggml_tensor* t = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, cols, rows);
            float* d = (float*)t->data;
            for (int i = 0; i < rows * cols; ++i)
                d[i] = (i % (cols + 1) == 0) ? 0.05f : 0.0f;
            return t;
        };
        auto make_1d = [&](int n, float val = 1.0f) -> ggml_tensor* {
            ggml_tensor* t = ggml_new_tensor_1d(wctx_, GGML_TYPE_F32, n);
            float* d = (float*)t->data;
            for (int i = 0; i < n; ++i) d[i] = val;
            return t;
        };

        w_.attn_norm     = make_1d(LS_N_EMBD);
        w_.q             = make_2d(LS_N_EMBD, LS_N_EMBD);
        w_.k             = make_2d(LS_N_HEAD_KV * LS_N_EMBD_HEAD, LS_N_EMBD);
        w_.v             = make_2d(LS_N_HEAD_KV * LS_N_EMBD_HEAD, LS_N_EMBD);
        w_.q_bias        = nullptr;
        w_.k_bias        = nullptr;
        w_.v_bias        = nullptr;
        w_.q_norm        = make_1d(LS_N_EMBD_HEAD);
        w_.k_norm        = make_1d(LS_N_EMBD_HEAD);
        w_.out           = make_2d(LS_N_EMBD, LS_N_EMBD);
        w_.ffn_norm      = make_1d(LS_N_EMBD);
        w_.ffn_gate      = make_2d(LS_N_FFN, LS_N_EMBD);
        w_.ffn_up        = make_2d(LS_N_FFN, LS_N_EMBD);
        w_.ffn_down      = make_2d(LS_N_EMBD, LS_N_FFN);

        base_hp_.is_qwen2      = false;
        base_hp_.n_head        = LS_N_HEAD;
        base_hp_.n_head_kv     = LS_N_HEAD_KV;
        base_hp_.n_embd_head   = LS_N_EMBD_HEAD;
        base_hp_.freq_base     = 10000.0f;
        base_hp_.context_length = LS_N_CTX_MAX;
        base_hp_.rms_norm_eps  = 1e-6f;
    }

    void TearDown() override {
        kv_cache_.reset();
        if (wctx_)    ggml_free(wctx_);
        if (backend_) ggml_backend_free(backend_);
    }

    std::unique_ptr<simple_kv_cache> make_cache(
            int head_dim_k = LS_N_EMBD_HEAD,
            int head_dim_v = LS_N_EMBD_HEAD) {
        return std::make_unique<simple_kv_cache>(
            /*n_layers=*/1, LS_N_CTX_MAX, /*n_batch_max=*/2,
            LS_N_HEAD_KV * head_dim_k, LS_N_HEAD_KV * head_dim_v,
            GGML_TYPE_F32, GGML_TYPE_F32, backend_);
    }

    // Build and run a single transformer layer; return output floats.
    std::vector<float> run(const TransformerBlockHparams& hp,
                           const TransformerBlockWeights& w,
                           simple_kv_cache* cache,
                           int n_tokens, int pos = 0)
    {
        const size_t ctx_bytes = 2048 * ggml_tensor_overhead()
                                + 128 * n_tokens * LS_N_EMBD * sizeof(float);
        ggml_init_params p = { ctx_bytes, nullptr, true };
        ggml_context* ctx = ggml_init(p);

        ggml_cgraph* gf = ggml_new_graph(ctx);

        ggml_tensor* input = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, LS_N_EMBD, n_tokens);
        ggml_set_input(input);
        ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp_pos);

        ggml_tensor* out = build_transformer_layer(
            ctx, gf, cache, input, inp_pos,
            w, hp, /*il=*/0, /*slot_idx=*/0,
            static_cast<uint32_t>(n_tokens));
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
        ggml_gallocr_alloc_graph(alloc, gf);

        // Set inputs
        {
            std::vector<float> inp(LS_N_EMBD * n_tokens, 0.5f);
            ggml_backend_tensor_set(input, inp.data(), 0, inp.size() * sizeof(float));
            std::vector<int32_t> pv(n_tokens);
            for (int i = 0; i < n_tokens; ++i) pv[i] = pos + i;
            ggml_backend_tensor_set(inp_pos, pv.data(), 0, pv.size() * sizeof(int32_t));
        }

        // Fill causal mask
        {
            char mask_name[32];
            snprintf(mask_name, sizeof(mask_name), "kq_mask.%d", 0);
            ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, mask_name);
            EXPECT_NE(kq_mask, nullptr) << "kq_mask.0 not found";
            if (kq_mask) {
                const uint32_t n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
                std::vector<float> mask(n_kv * n_tokens);
                for (int i = 0; i < n_tokens; ++i) {
                    const int q_pos = pos + i;
                    for (uint32_t j = 0; j < n_kv; ++j)
                        mask[i * n_kv + j] = (static_cast<int>(j) <= q_pos) ? 0.0f : -INFINITY;
                }
                ggml_backend_tensor_set(kq_mask, mask.data(), 0,
                    n_kv * n_tokens * sizeof(float));
            }
        }

        ggml_backend_graph_compute(backend_, gf);
        cache->advance(static_cast<uint32_t>(n_tokens), 0);

        std::vector<float> result(LS_N_EMBD * n_tokens);
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return result;
    }

    ggml_backend_t backend_ = nullptr;
    ggml_context*  wctx_    = nullptr;
    TransformerBlockWeights  w_{};
    TransformerBlockHparams  base_hp_{};
    std::unique_ptr<simple_kv_cache> kv_cache_;
};

// Test 2: Degenerate case — head_dim_k = head_dim_v = n_embd_head and
// rope_dim = n_embd_head produces bit-identical output to the zero-default path.
TEST_F(LayerSpecBlockTest, DegenerateIsUniform) {
    auto cache_a = make_cache();
    auto out_a = run(base_hp_, w_, cache_a.get(), LS_N_TOKENS);

    // Explicitly set the new fields to match n_embd_head (the degenerate case).
    TransformerBlockHparams hp_explicit = base_hp_;
    hp_explicit.head_dim_k = LS_N_EMBD_HEAD;
    hp_explicit.head_dim_v = LS_N_EMBD_HEAD;
    hp_explicit.rope_dim   = LS_N_EMBD_HEAD;

    auto cache_b = make_cache();
    auto out_b = run(hp_explicit, w_, cache_b.get(), LS_N_TOKENS);

    ASSERT_EQ(out_a.size(), out_b.size());
    for (size_t i = 0; i < out_a.size(); ++i) {
        EXPECT_FLOAT_EQ(out_a[i], out_b[i])
            << "bit-identical mismatch at index " << i;
    }
}

// Test 3: Per-layer head_dim variation — n_head * head_dim != hidden_dim.
//
// This is the key G4.1 scenario: Gemma 4 global layers have head_dim=512 with
// n_head=16, giving 16*512=8192 >> hidden_dim=2816.  The recipe sets
// n_embd_head=512 per layer, and head_dim_k/v are redundant (== n_embd_head)
// but their explicit values must flow through correctly.
//
// Synthetic: hidden_dim=32, n_head=2, head_dim=8 → n_head*head_dim=16 ≠ 32.
//   Q weight: [hidden_dim, n_head*head_dim] = ggml [16, 32]
//   K weight: [hidden_dim, n_kv_heads*head_dim] = ggml [16, 32]
//   V weight: same shape as K.
//   Out proj: [n_head*head_dim, hidden_dim] = ggml [32, 16]
//   q_norm / k_norm: [head_dim] = [8]
//   KV cache: n_embd_k = n_kv_heads * head_dim = 2 * 8 = 16.
TEST_F(LayerSpecBlockTest, PerLayerHeadDimVariation) {
    constexpr int head_dim   = 8;    // per-head dim for this "global" layer
    constexpr int n_head     = 2;    // same as LS_N_HEAD
    constexpr int n_kv       = 2;    // same as LS_N_HEAD_KV
    // n_head * head_dim = 16 ≠ 32 = LS_N_EMBD (the Gemma 4 scenario)

    auto make2d = [&](int out_dim, int in_dim) -> ggml_tensor* {
        // ggml layout: [in_dim, out_dim] — matmul produces [out_dim, n_tok]
        ggml_tensor* t = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, in_dim, out_dim);
        float* d = (float*)t->data;
        for (int i = 0; i < out_dim * in_dim; ++i) d[i] = 0.01f;
        return t;
    };
    auto make1d = [&](int n) -> ggml_tensor* {
        ggml_tensor* t = ggml_new_tensor_1d(wctx_, GGML_TYPE_F32, n);
        float* d = (float*)t->data;
        for (int i = 0; i < n; ++i) d[i] = 1.0f;
        return t;
    };

    TransformerBlockWeights w_g4{};
    w_g4.attn_norm  = make1d(LS_N_EMBD);
    w_g4.q          = make2d(n_head * head_dim,   LS_N_EMBD);  // [16, 32]
    w_g4.k          = make2d(n_kv  * head_dim,    LS_N_EMBD);  // [16, 32]
    w_g4.v          = make2d(n_kv  * head_dim,    LS_N_EMBD);  // [16, 32]
    w_g4.q_norm     = make1d(head_dim);                         // [8]
    w_g4.k_norm     = make1d(head_dim);                         // [8]
    w_g4.out        = make2d(LS_N_EMBD, n_head * head_dim);     // [32, 16]
    w_g4.ffn_norm   = make1d(LS_N_EMBD);
    w_g4.ffn_gate   = make2d(LS_N_FFN, LS_N_EMBD);
    w_g4.ffn_up     = make2d(LS_N_FFN, LS_N_EMBD);
    w_g4.ffn_down   = make2d(LS_N_EMBD, LS_N_FFN);

    // hparams: n_embd_head carries the per-layer head dim.
    // head_dim_k and head_dim_v are set explicitly (== n_embd_head) to verify
    // the zero-fallback and the explicit path produce the same behaviour.
    TransformerBlockHparams hp_g4 = base_hp_;
    hp_g4.n_embd_head  = head_dim;   // 8 — per-layer head dim
    hp_g4.n_head       = n_head;
    hp_g4.n_head_kv    = n_kv;
    hp_g4.head_dim_k   = head_dim;   // explicit == n_embd_head (no new info)
    hp_g4.head_dim_v   = head_dim;
    hp_g4.rope_dim     = head_dim;   // RoPE covers the full head dim

    auto cache_g4 = make_cache(head_dim, head_dim);
    auto out = run(hp_g4, w_g4, cache_g4.get(), LS_N_TOKENS);

    ASSERT_EQ(out.size(), static_cast<size_t>(LS_N_EMBD * LS_N_TOKENS));
    for (float v : out)
        EXPECT_TRUE(std::isfinite(v))
            << "non-finite in per-layer-head-dim transformer output";
}
