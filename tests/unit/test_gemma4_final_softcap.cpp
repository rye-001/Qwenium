// test_gemma4_final_softcap.cpp — PR G4.6
//
// Re-enables the Gemma 2 final-logit soft-cap path for Gemma 4 and exercises
// it against a hand-rolled oracle.  Behaviorally identical to G2.4, only the
// cap value differs (30 for G4 vs. 50 for G2).
//
// Production wiring lives in the G4.8 recipe (src/models/gemma4.cpp): after
// the LM head matmul, build_softcap(ctx, logits, 30.0f) is applied.  This
// unit test isolates the op on an LM-head-shaped tensor so the oracle is
// independent of any recipe.
//
// Oracle: 30 * tanh(x / 30)
//   x=  0   → 0
//   x= 30   → 30 * tanh(1)        ≈ 22.84782
//   x=300   → 30 * tanh(10)       ≈ 30 - ε  (saturates at +cap)
//   x=-30   → -30 * tanh(1)       ≈ -22.84782
//
// Null-cap path: callers gate the call (cap > 0); we don't exercise cap == 0
// here because the contract is "skip the call if cap == 0" — see the comment
// on build_softcap in attention.h.
//
// No model file required.
// Build target: gemma4-final-softcap-tests (see tests/CMakeLists.txt).

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "../../src/layers/attention.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

static constexpr float G4_FINAL_CAP = 30.0f;
static constexpr float TOL          = 1e-4f;

namespace {

// Run build_softcap on an LM-head-shaped 2D tensor [vocab, n_tokens] with the
// given cap, return a flat vector of outputs in row-major order.
std::vector<float> run_softcap_2d(int vocab, int n_tokens,
                                  const std::vector<float>& values,
                                  float cap)
{
    EXPECT_EQ(values.size(), static_cast<size_t>(vocab * n_tokens));
    ggml_backend_t backend = ggml_backend_cpu_init();
    EXPECT_NE(backend, nullptr);

    const size_t ctx_bytes = 256 * 1024;
    ggml_init_params p{ ctx_bytes, nullptr, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(p);

    ggml_cgraph* gf = ggml_new_graph(ctx);

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab, n_tokens);
    ggml_set_input(x);

    ggml_tensor* y = build_softcap(ctx, x, cap);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(x, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> out(values.size());
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return out;
}

} // namespace

// 1. Zero input → zero output: tanh(0)=0 ⇒ cap·tanh(0/cap)=0 for any cap.
TEST(Gemma4FinalSoftcapTest, ZeroVector) {
    std::vector<float> in(8, 0.0f);
    auto out = run_softcap_2d(/*vocab=*/8, /*n_tokens=*/1, in, G4_FINAL_CAP);
    for (float v : out) EXPECT_NEAR(v, 0.0f, TOL);
}

// 2. Curated value oracle on a [vocab=4, n_tokens=2] tensor.
// Rows are token-major in row-major flattening; ggml stores the tensor with
// vocab as the contiguous (ne[0]) dim, so flat index i*vocab+j gives token i,
// vocab j.  We pick values that exercise the small-x linear regime, the unit
// scale (x = cap), the saturating regime (x = 10·cap), and signed values.
TEST(Gemma4FinalSoftcapTest, CuratedOracle) {
    const std::vector<float> in = {
        // token 0, vocab 0..3
         0.0f,    1.0f,   30.0f,   300.0f,
        // token 1, vocab 0..3
        -1.0f,  -30.0f, -300.0f,    15.0f,
    };
    auto out = run_softcap_2d(/*vocab=*/4, /*n_tokens=*/2, in, G4_FINAL_CAP);

    auto oracle = [](float x) {
        return G4_FINAL_CAP * std::tanh(x / G4_FINAL_CAP);
    };

    ASSERT_EQ(out.size(), in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        EXPECT_NEAR(out[i], oracle(in[i]), TOL)
            << "mismatch at flat index " << i << " (x=" << in[i] << ")";
    }
}

// 3. Saturation: |x| ≫ cap ⇒ output approaches ±cap.
// At x = 10·cap, |30·tanh(10)| > 29.9999999 — well within fp32 epsilon of 30.
TEST(Gemma4FinalSoftcapTest, SaturatesAtCap) {
    const std::vector<float> in = { 300.0f, -300.0f, 30000.0f, -30000.0f };
    auto out = run_softcap_2d(/*vocab=*/4, /*n_tokens=*/1, in, G4_FINAL_CAP);
    EXPECT_NEAR(out[0],  G4_FINAL_CAP, 1e-3f);
    EXPECT_NEAR(out[1], -G4_FINAL_CAP, 1e-3f);
    EXPECT_NEAR(out[2],  G4_FINAL_CAP, 1e-3f);
    EXPECT_NEAR(out[3], -G4_FINAL_CAP, 1e-3f);
}

// 4. Sign symmetry: f(-x) = -f(x), since tanh is odd.
TEST(Gemma4FinalSoftcapTest, SignSymmetric) {
    const std::vector<float> pos_in = { 0.5f, 5.0f, 30.0f, 100.0f };
    std::vector<float> neg_in(pos_in.size());
    for (size_t i = 0; i < pos_in.size(); ++i) neg_in[i] = -pos_in[i];

    auto pos_out = run_softcap_2d(/*vocab=*/4, /*n_tokens=*/1, pos_in,
                                  G4_FINAL_CAP);
    auto neg_out = run_softcap_2d(/*vocab=*/4, /*n_tokens=*/1, neg_in,
                                  G4_FINAL_CAP);
    ASSERT_EQ(pos_out.size(), neg_out.size());
    for (size_t i = 0; i < pos_out.size(); ++i) {
        EXPECT_NEAR(pos_out[i], -neg_out[i], TOL)
            << "asymmetry at index " << i;
    }
}
