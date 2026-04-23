// test_deltanet.cpp — PR 3.D.2-4
//
// Tests DeltaNetLayer: graph-building module for Gated DeltaNet layers.
// Uses the fused ggml_gated_delta_net op (Phase 3 fused path).
//
// Oracle: shape + finiteness checks. Logit-level comparison against llama.cpp
// is in the PR 3.I.2 integration gate.
//
// Run: ./qwen3-deltanet-tests --gtest_filter="DeltaNetLayer*"

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>

#include "../../src/layers/deltanet.h"
#include "../../src/state/deltanet_state.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

// Small dimensions safe for CPU unit tests
static constexpr int N_EMBD        = 32;
static constexpr int D_INNER       = 16;
static constexpr int HEAD_K_DIM    = 8;  // Must match HEAD_V_DIM for fused op
static constexpr int NUM_K_HEADS   = 2;
static constexpr int NUM_V_HEADS   = 2;
static constexpr int HEAD_V_DIM    = D_INNER / NUM_V_HEADS;  // 8
static constexpr int CONV_KERNEL   = 4;
static constexpr int CONV_CHANNELS = D_INNER + 2 * NUM_K_HEADS * HEAD_K_DIM;  // 16 + 2*2*8 = 48
static constexpr int N_SLOTS       = 1;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Make a DeltaNetState with one logical DeltaNet layer.
static DeltaNetState make_dn_state(ggml_backend_t be) {
    DeltaNetState::Hparams hp;
    hp.n_dn_layers   = 1;
    hp.n_slots       = N_SLOTS;
    hp.head_v_dim    = HEAD_V_DIM;
    hp.head_k_dim    = HEAD_K_DIM;
    hp.num_v_heads   = NUM_V_HEADS;
    hp.conv_channels = CONV_CHANNELS;
    hp.conv_kernel   = CONV_KERNEL;
    hp.backend       = be;
    return DeltaNetState(hp);
}

// Fill an inline-allocated (no_alloc=false) tensor with a constant.
static void fill_f32(ggml_tensor* t, float v) {
    float* d = (float*)t->data;
    size_t n = ggml_nelements(t);
    for (size_t i = 0; i < n; ++i) d[i] = v;
}

// Allocate weight tensors in an inline context.
// Returns ctx pointer — caller must ggml_free it.
struct Weights {
    ggml_context* ctx;
    ggml_tensor* qkv;
    ggml_tensor* gate;
    ggml_tensor* beta;
    ggml_tensor* a;
    ggml_tensor* dt_bias;
    ggml_tensor* a_log;
    ggml_tensor* conv;
    ggml_tensor* norm;
    ggml_tensor* out;
};

static Weights make_weights(float val = 0.01f) {
    // Need enough space for all weight tensors
    const size_t total_floats =
        static_cast<size_t>(N_EMBD) * CONV_CHANNELS +   // qkv
        static_cast<size_t>(N_EMBD) * D_INNER       +   // gate
        static_cast<size_t>(N_EMBD) * NUM_V_HEADS   +   // beta
        static_cast<size_t>(N_EMBD) * NUM_V_HEADS   +   // a
        NUM_V_HEADS                                  +   // dt_bias
        1 * NUM_V_HEADS                              +   // a_log
        CONV_CHANNELS * 1 * CONV_KERNEL              +   // conv
        HEAD_V_DIM                                   +   // norm (per-head, not d_inner)
        static_cast<size_t>(D_INNER) * N_EMBD;          // out

    const size_t ctx_size = 64 * ggml_tensor_overhead()
                          + total_floats * sizeof(float) + 4096;

    struct ggml_init_params p{ctx_size, nullptr, false};
    ggml_context* ctx = ggml_init(p);

    Weights w;
    w.ctx     = ctx;
    w.qkv     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, CONV_CHANNELS);
    w.gate    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, D_INNER);
    w.beta    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, NUM_V_HEADS);
    w.a       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, NUM_V_HEADS);
    w.dt_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NUM_V_HEADS);
    w.a_log   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NUM_V_HEADS);
    w.conv    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, CONV_KERNEL, CONV_CHANNELS);
    w.norm    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HEAD_V_DIM);
    w.out     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_INNER, N_EMBD);

    fill_f32(w.qkv, val);
    fill_f32(w.gate, val);
    fill_f32(w.beta, val);
    fill_f32(w.a, val);
    fill_f32(w.dt_bias, 0.0f);
    fill_f32(w.a_log, -1.0f);   // negative so decay < 1
    fill_f32(w.conv, 0.1f);
    fill_f32(w.norm, 1.0f);     // identity RMSNorm
    fill_f32(w.out, val);

    return w;
}

// Build and execute a DeltaNet prefill graph. Returns output data.
// Returns false on shape mismatch.
static bool run_prefill(ggml_backend_t be, int n_tokens,
                        std::vector<float>* out_data) {
    Weights w = make_weights(0.01f);
    DeltaNetState dn_state = make_dn_state(be);

    // Graph context: must be large enough for all intermediate tensors.
    // The gated_delta_net op can be memory-hungry — give 32 MB.
    const size_t g_sz = 4096 * ggml_tensor_overhead() + 32 * 1024 * 1024;
    struct ggml_init_params gp{g_sz, nullptr, false};
    ggml_context* g_ctx = ggml_init(gp);
    ggml_cgraph*  gf    = ggml_new_graph_custom(g_ctx, 8192, false);

    ggml_tensor* inp = ggml_new_tensor_2d(g_ctx, GGML_TYPE_F32, N_EMBD, n_tokens);
    ggml_set_input(inp);
    fill_f32(inp, 0.1f);

    DeltaNetLayer::Hparams hp;
    hp.n_embd        = N_EMBD;
    hp.d_inner       = D_INNER;
    hp.head_k_dim    = HEAD_K_DIM;
    hp.num_k_heads   = NUM_K_HEADS;
    hp.num_v_heads   = NUM_V_HEADS;
    hp.head_v_dim    = HEAD_V_DIM;
    hp.conv_channels = CONV_CHANNELS;
    hp.conv_kernel   = CONV_KERNEL;
    hp.rms_norm_eps  = 1e-6f;

    DeltaNetLayer layer(w.qkv, w.gate, w.beta, w.a, w.dt_bias, w.a_log,
                        w.conv, w.norm, w.out, &dn_state, hp);

    DeltaNetLayer::PrefillArgs args;
    args.n_tokens = static_cast<uint32_t>(n_tokens);
    args.slot_idx = 0;

    ggml_tensor* out = layer.build(g_ctx, gf, inp, /*dn_idx=*/0,
                                   Phase::Prefill, args, nullptr);

    bool ok = (out != nullptr
               && out->ne[0] == N_EMBD
               && out->ne[1] == n_tokens);

    if (ok) {
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(alloc, gf);
        ggml_backend_graph_compute(be, gf);

        size_t n = ggml_nelements(out);
        out_data->resize(n);
        // out is in the inline graph context — data pointer is valid
        std::memcpy(out_data->data(), out->data, n * sizeof(float));

        ggml_gallocr_free(alloc);
    }

    ggml_free(g_ctx);
    ggml_free(w.ctx);
    return ok;
}

// ── Test fixture ──────────────────────────────────────────────────────────────

class DeltaNetLayerTest : public ::testing::Test {
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

// ── Prefill path ──────────────────────────────────────────────────────────────

TEST_F(DeltaNetLayerTest, PrefillBuildSucceeds) {
    std::vector<float> out;
    EXPECT_TRUE(run_prefill(backend_, /*n_tokens=*/3, &out));
    EXPECT_EQ(out.size(), static_cast<size_t>(N_EMBD * 3));
}

TEST_F(DeltaNetLayerTest, PrefillOutputIsFinite) {
    std::vector<float> out;
    ASSERT_TRUE(run_prefill(backend_, 3, &out));
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_TRUE(std::isfinite(out[i])) << "NaN/Inf at i=" << i;
}

TEST_F(DeltaNetLayerTest, PrefillSingleTokenSucceeds) {
    std::vector<float> out;
    EXPECT_TRUE(run_prefill(backend_, 1, &out));
    EXPECT_EQ(out.size(), static_cast<size_t>(N_EMBD));
}

// ── Decode path ───────────────────────────────────────────────────────────────

TEST_F(DeltaNetLayerTest, DecodeBuildSucceeds) {
    Weights w = make_weights(0.01f);
    DeltaNetState dn_state = make_dn_state(backend_);

    const size_t g_sz = 4096 * ggml_tensor_overhead() + 32 * 1024 * 1024;
    struct ggml_init_params gp{g_sz, nullptr, false};
    ggml_context* g_ctx = ggml_init(gp);
    ggml_cgraph*  gf    = ggml_new_graph_custom(g_ctx, 8192, false);

    // Decode: 1 token per batch slot
    ggml_tensor* inp = ggml_new_tensor_2d(g_ctx, GGML_TYPE_F32, N_EMBD, 1);
    ggml_set_input(inp);
    fill_f32(inp, 0.1f);

    DeltaNetLayer::Hparams hp;
    hp.n_embd        = N_EMBD;
    hp.d_inner       = D_INNER;
    hp.head_k_dim    = HEAD_K_DIM;
    hp.num_k_heads   = NUM_K_HEADS;
    hp.num_v_heads   = NUM_V_HEADS;
    hp.head_v_dim    = HEAD_V_DIM;
    hp.conv_channels = CONV_CHANNELS;
    hp.conv_kernel   = CONV_KERNEL;
    hp.rms_norm_eps  = 1e-6f;

    DeltaNetLayer layer(w.qkv, w.gate, w.beta, w.a, w.dt_bias, w.a_log,
                        w.conv, w.norm, w.out, &dn_state, hp);

    DeltaNetLayer::DecodeArgs dargs;
    dargs.slots = {0};

    ggml_tensor* out = layer.build(g_ctx, gf, inp, 0, Phase::Decode, {}, &dargs);

    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->ne[0], N_EMBD);
    EXPECT_EQ(out->ne[1], 1);

    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend_));
    ggml_gallocr_alloc_graph(alloc, gf);
    ggml_backend_graph_compute(backend_, gf);

    float* result = (float*)out->data;
    for (int i = 0; i < N_EMBD; ++i)
        EXPECT_TRUE(std::isfinite(result[i])) << "NaN/Inf at i=" << i;

    ggml_gallocr_free(alloc);
    ggml_free(g_ctx);
    ggml_free(w.ctx);
}

// ── Qwen 3.6 GQA-style heads ──────────────────────────────────────────────────

TEST_F(DeltaNetLayerTest, UnequalKVHeadsSucceeds) {
    // Qwen3.6 uses num_k_heads = 16, num_v_heads = 32.
    // We scale down for the unit test: num_k_heads = 2, num_v_heads = 4.
    const int k_heads = 2;
    const int v_heads = 4;
    const int k_dim = 8;
    const int v_dim = 8;
    const int d_in = v_heads * v_dim; // 32
    const int embd = 64;
    const int channels = d_in + 2 * k_heads * k_dim; // 32 + 32 = 64
    const int seq_len = 3;

    DeltaNetState::Hparams shp;
    shp.n_dn_layers = 1;
    shp.n_slots = 1;
    shp.head_v_dim = v_dim;
    shp.head_k_dim = k_dim;
    shp.num_v_heads = v_heads;
    shp.conv_channels = channels;
    shp.conv_kernel = 4;
    shp.backend = backend_;
    DeltaNetState dn_state(shp);

    const size_t ctx_size = 1024 * 1024;
    struct ggml_init_params p{ctx_size, nullptr, false};
    ggml_context* ctx = ggml_init(p);

    ggml_tensor* w_qkv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embd, channels);
    ggml_tensor* w_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embd, d_in);
    ggml_tensor* w_beta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embd, v_heads);
    ggml_tensor* w_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embd, v_heads);
    ggml_tensor* w_dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, v_heads);
    ggml_tensor* w_alog = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, v_heads);
    ggml_tensor* w_conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, channels);
    ggml_tensor* w_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, v_dim);
    ggml_tensor* w_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_in, embd);

    fill_f32(w_qkv, 0.01f);
    fill_f32(w_gate, 0.01f);
    fill_f32(w_beta, 0.01f);
    fill_f32(w_a, 0.01f);
    fill_f32(w_dt, 0.0f);
    fill_f32(w_alog, -1.0f);
    fill_f32(w_conv, 0.1f);
    fill_f32(w_norm, 1.0f);
    fill_f32(w_out, 0.01f);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_tensor* inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embd, seq_len);
    ggml_set_input(inp);
    fill_f32(inp, 0.1f);

    DeltaNetLayer::Hparams hp;
    hp.n_embd = embd;
    hp.d_inner = d_in;
    hp.head_k_dim = k_dim;
    hp.num_k_heads = k_heads;
    hp.num_v_heads = v_heads;
    hp.head_v_dim = v_dim;
    hp.conv_channels = channels;
    hp.conv_kernel = 4;
    hp.rms_norm_eps = 1e-6f;

    DeltaNetLayer layer(w_qkv, w_gate, w_beta, w_a, w_dt, w_alog,
                        w_conv, w_norm, w_out, &dn_state, hp);

    DeltaNetLayer::PrefillArgs args;
    args.n_tokens = seq_len;
    args.slot_idx = 0;

    ggml_tensor* out = layer.build(ctx, gf, inp, 0, Phase::Prefill, args, nullptr);

    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->ne[0], embd);
    EXPECT_EQ(out->ne[1], seq_len);

    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    ggml_gallocr_alloc_graph(alloc, gf);
    ggml_backend_graph_compute(backend_, gf);

    float* result = (float*)out->data;
    for (int i = 0; i < embd * seq_len; ++i)
        EXPECT_TRUE(std::isfinite(result[i]));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

// ── State mutation: recurrent state is updated after a forward pass ───────────

TEST_F(DeltaNetLayerTest, PrefillMutatesRecurrentState) {
    Weights w = make_weights(0.05f);
    DeltaNetState dn_state = make_dn_state(backend_);

    const size_t rec_floats = HEAD_V_DIM * HEAD_K_DIM * NUM_V_HEADS;

    // Verify initial state is zeroed
    std::vector<float> before(rec_floats);
    dn_state.get_recurrent(0, 0, before.data());
    for (size_t i = 0; i < rec_floats; ++i)
        ASSERT_EQ(before[i], 0.0f) << "initial state not zero at i=" << i;

    // Run a prefill pass with non-zero input
    {
        const size_t g_sz = 4096 * ggml_tensor_overhead() + 32 * 1024 * 1024;
        struct ggml_init_params gp{g_sz, nullptr, false};
        ggml_context* g_ctx = ggml_init(gp);
        ggml_cgraph*  gf    = ggml_new_graph_custom(g_ctx, 8192, false);

        ggml_tensor* inp = ggml_new_tensor_2d(g_ctx, GGML_TYPE_F32, N_EMBD, 1);
        ggml_set_input(inp);
        fill_f32(inp, 0.5f);

        DeltaNetLayer::Hparams hp;
        hp.d_inner       = D_INNER;
        hp.head_k_dim    = HEAD_K_DIM;
        hp.num_k_heads   = NUM_K_HEADS;
        hp.num_v_heads   = NUM_V_HEADS;
        hp.head_v_dim    = HEAD_V_DIM;
        hp.conv_channels = CONV_CHANNELS;
        hp.conv_kernel   = CONV_KERNEL;
        hp.rms_norm_eps  = 1e-6f;

        DeltaNetLayer layer(w.qkv, w.gate, w.beta, w.a, w.dt_bias, w.a_log,
                            w.conv, w.norm, w.out, &dn_state, hp);

        DeltaNetLayer::PrefillArgs args;
        args.n_tokens = 1;
        args.slot_idx = 0;

        ggml_tensor* out = layer.build(g_ctx, gf, inp, 0, Phase::Prefill, args, nullptr);
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
        ggml_gallocr_alloc_graph(alloc, gf);
        ggml_backend_graph_compute(backend_, gf);
        ggml_gallocr_free(alloc);
        ggml_free(g_ctx);
    }

    // State should have been updated (non-zero)
    std::vector<float> after(rec_floats);
    dn_state.get_recurrent(0, 0, after.data());

    bool any_nonzero = false;
    for (size_t i = 0; i < rec_floats; ++i) {
        if (after[i] != 0.0f) { any_nonzero = true; break; }
    }
    EXPECT_TRUE(any_nonzero) << "Recurrent state was not updated after prefill";

    ggml_free(w.ctx);
}
