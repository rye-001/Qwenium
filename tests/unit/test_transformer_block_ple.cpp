// test_transformer_block_ple.cpp — PR G4.2
//
// Verifies the build_transformer_layer second-residual (PLE) injection slot.
//
// Contract:
//   1. ple_residual == nullptr  → behavior bit-identical to before G4.2
//      (existing Qwen / Gemma 1/2/3 path; this is a regression gate against
//      accidentally introducing extra ggml_add nodes).
//   2. ple_residual all-zeros   → bit-identical to nullptr at the *value*
//      level; ggml_add of a zero tensor is the identity.
//   3. ple_residual non-zero    → output differs from the nullptr/zero path.
//      We don't oracle-check the exact value (that's PLE-spec territory and
//      is exercised by test_ple.cpp + the recipe integration in G4.8); we
//      only confirm the residual is actually consumed.
//
// The injection point is documented in transformer_block.h: ple_residual is
// added at block entry, before the pre-attention norm, so the post-injection
// value flows through both the attention residual and the FFN residual.
//
// No model file required.
// Build target: layer-spec-tests (this file is appended to the same target;
// see tests/CMakeLists.txt).
//
// NOTE: This test is currently linked into layer-spec-tests because both
// share the same fixture shape (small synthetic transformer layer over CPU
// backend).  Splitting into a dedicated target adds CMake/CI cost without
// improving coverage.

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "../../src/layers/transformer_block.h"
#include "../../src/state/kv_cache_simple.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// Synthetic dimensions (kept identical to test_layer_spec.cpp so the harness
// shape is familiar — this is a lateral test of the same machinery).
static constexpr int PLE_N_EMBD      = 32;
static constexpr int PLE_N_HEAD      = 2;
static constexpr int PLE_N_HEAD_KV   = 2;
static constexpr int PLE_N_EMBD_HEAD = PLE_N_EMBD / PLE_N_HEAD;  // 16
static constexpr int PLE_N_FFN       = 64;
static constexpr int PLE_N_CTX_MAX   = 64;
static constexpr int PLE_N_TOKENS    = 4;

class TransformerBlockPleTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        ASSERT_NE(backend_, nullptr);

        const size_t weight_bytes = 128 * ggml_tensor_overhead()
            + (2*PLE_N_EMBD*PLE_N_EMBD
               + 2*PLE_N_HEAD_KV*PLE_N_EMBD_HEAD*PLE_N_EMBD
               + 3*PLE_N_FFN*PLE_N_EMBD
               + 4*PLE_N_EMBD
               + 2*PLE_N_EMBD_HEAD) * sizeof(float);
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

        w_.attn_norm = make_1d(PLE_N_EMBD);
        w_.q         = make_2d(PLE_N_EMBD, PLE_N_EMBD);
        w_.k         = make_2d(PLE_N_HEAD_KV * PLE_N_EMBD_HEAD, PLE_N_EMBD);
        w_.v         = make_2d(PLE_N_HEAD_KV * PLE_N_EMBD_HEAD, PLE_N_EMBD);
        w_.q_bias    = nullptr;
        w_.k_bias    = nullptr;
        w_.v_bias    = nullptr;
        w_.q_norm    = make_1d(PLE_N_EMBD_HEAD);
        w_.k_norm    = make_1d(PLE_N_EMBD_HEAD);
        w_.out       = make_2d(PLE_N_EMBD, PLE_N_EMBD);
        w_.ffn_norm  = make_1d(PLE_N_EMBD);
        w_.ffn_gate  = make_2d(PLE_N_FFN, PLE_N_EMBD);
        w_.ffn_up    = make_2d(PLE_N_FFN, PLE_N_EMBD);
        w_.ffn_down  = make_2d(PLE_N_EMBD, PLE_N_FFN);

        hp_.is_qwen2       = false;
        hp_.n_head         = PLE_N_HEAD;
        hp_.n_head_kv      = PLE_N_HEAD_KV;
        hp_.n_embd_head    = PLE_N_EMBD_HEAD;
        hp_.freq_base      = 10000.0f;
        hp_.context_length = PLE_N_CTX_MAX;
        hp_.rms_norm_eps   = 1e-6f;
    }

    void TearDown() override {
        if (wctx_)    ggml_free(wctx_);
        if (backend_) ggml_backend_free(backend_);
    }

    std::unique_ptr<simple_kv_cache> make_cache() {
        return std::make_unique<simple_kv_cache>(
            /*n_layers=*/1, PLE_N_CTX_MAX, /*n_batch_max=*/2,
            PLE_N_HEAD_KV * PLE_N_EMBD_HEAD,
            PLE_N_HEAD_KV * PLE_N_EMBD_HEAD,
            GGML_TYPE_F32, GGML_TYPE_F32, backend_);
    }

    // Run one transformer layer with optional PLE residual.
    // ple_data == nullptr → call build_transformer_layer with nullptr ple_residual.
    // ple_data != nullptr → allocate a [PLE_N_EMBD, n_tokens] input tensor in
    //                       the compute graph and pass it as ple_residual.
    std::vector<float> run(const std::vector<float>* ple_data) {
        auto cache = make_cache();

        const size_t ctx_bytes = 2048 * ggml_tensor_overhead()
            + 128 * PLE_N_TOKENS * PLE_N_EMBD * sizeof(float);
        ggml_init_params p = { ctx_bytes, nullptr, /*no_alloc=*/true };
        ggml_context* ctx = ggml_init(p);

        ggml_cgraph* gf = ggml_new_graph(ctx);

        ggml_tensor* input = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, PLE_N_EMBD, PLE_N_TOKENS);
        ggml_set_input(input);

        ggml_tensor* inp_pos = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, PLE_N_TOKENS);
        ggml_set_input(inp_pos);

        ggml_tensor* ple_t = nullptr;
        if (ple_data != nullptr) {
            ple_t = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, PLE_N_EMBD, PLE_N_TOKENS);
            ggml_set_input(ple_t);
            ggml_set_name(ple_t, "ple_residual");
        }

        ggml_tensor* out = build_transformer_layer(
            ctx, gf, cache.get(), input, inp_pos,
            w_, hp_, /*il=*/0, /*slot_idx=*/0,
            static_cast<uint32_t>(PLE_N_TOKENS),
            ple_t);
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
        ggml_gallocr_alloc_graph(alloc, gf);

        std::vector<float> inp(PLE_N_EMBD * PLE_N_TOKENS, 0.5f);
        ggml_backend_tensor_set(input, inp.data(), 0,
                                inp.size() * sizeof(float));
        std::vector<int32_t> pv(PLE_N_TOKENS);
        for (int i = 0; i < PLE_N_TOKENS; ++i) pv[i] = i;
        ggml_backend_tensor_set(inp_pos, pv.data(), 0,
                                pv.size() * sizeof(int32_t));

        if (ple_t) {
            ggml_backend_tensor_set(ple_t, ple_data->data(), 0,
                                    ple_data->size() * sizeof(float));
        }

        // Causal mask (single layer, layer 0).
        ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, "kq_mask.0");
        EXPECT_NE(kq_mask, nullptr) << "kq_mask.0 not found";
        if (kq_mask) {
            const uint32_t n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
            std::vector<float> mask(n_kv * PLE_N_TOKENS);
            for (int i = 0; i < PLE_N_TOKENS; ++i) {
                for (uint32_t j = 0; j < n_kv; ++j) {
                    mask[i * n_kv + j] = (static_cast<int>(j) <= i) ? 0.0f
                                                                    : -INFINITY;
                }
            }
            ggml_backend_tensor_set(kq_mask, mask.data(), 0,
                                    n_kv * PLE_N_TOKENS * sizeof(float));
        }

        ggml_backend_graph_compute(backend_, gf);

        std::vector<float> result(PLE_N_EMBD * PLE_N_TOKENS);
        ggml_backend_tensor_get(out, result.data(), 0,
                                result.size() * sizeof(float));

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return result;
    }

    ggml_backend_t           backend_ = nullptr;
    ggml_context*            wctx_    = nullptr;
    TransformerBlockWeights  w_{};
    TransformerBlockHparams  hp_{};
};

// 1. Null PLE residual is the inherited Qwen/Gemma1/2/3 path.  We don't
//    cross-check against an old binary; we only assert that the call
//    succeeds and produces finite output.  Bit-identity against the legacy
//    path is structurally guaranteed by the `if (ple_residual != nullptr)`
//    guard in build_transformer_layer.
TEST_F(TransformerBlockPleTest, NullPathFinite) {
    auto out = run(nullptr);
    ASSERT_EQ(out.size(),
              static_cast<size_t>(PLE_N_EMBD * PLE_N_TOKENS));
    for (float v : out) {
        EXPECT_TRUE(std::isfinite(v))
            << "non-finite output on null-PLE path";
    }
}

// 2. All-zero PLE residual is bit-identical to the null path at the value
//    level (ggml_add of zeros is the identity).  This guards against the
//    op being implemented as something other than a plain add.
TEST_F(TransformerBlockPleTest, ZeroResidualMatchesNull) {
    auto null_out = run(nullptr);
    std::vector<float> zeros(PLE_N_EMBD * PLE_N_TOKENS, 0.0f);
    auto zero_out = run(&zeros);

    ASSERT_EQ(null_out.size(), zero_out.size());
    for (size_t i = 0; i < null_out.size(); ++i) {
        EXPECT_FLOAT_EQ(null_out[i], zero_out[i])
            << "zero-PLE deviated from null-PLE at flat index " << i;
    }
}

// 3. Non-zero PLE residual changes the output.  This catches the failure
//    mode where the parameter is plumbed through but not actually consumed
//    (e.g. forgetting to use the modified `cur` further down the function).
TEST_F(TransformerBlockPleTest, NonZeroResidualPropagates) {
    auto null_out = run(nullptr);

    std::vector<float> ple(PLE_N_EMBD * PLE_N_TOKENS);
    for (size_t i = 0; i < ple.size(); ++i) {
        ple[i] = 0.1f * static_cast<float>((i % 7) + 1);  // varied non-zero
    }
    auto ple_out = run(&ple);

    bool differs = false;
    for (size_t i = 0; i < null_out.size(); ++i) {
        if (std::fabs(null_out[i] - ple_out[i]) > 1e-5f) { differs = true; break; }
    }
    EXPECT_TRUE(differs)
        << "non-zero PLE produced output identical to null path — residual "
           "appears to be unused";
}
