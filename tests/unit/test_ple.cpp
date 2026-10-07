// test_ple.cpp — co-located test for src/layers/ple.cpp (Per-Layer Embeddings).
//
// Checks build_ple_inputs + build_ple_inject on the CPU backend against the
// same formula written out in doubles (ple.h states it). Small synthetic
// shapes; no model file. The real-model gate is EmbeddingGemma-2 against
// llama.cpp (test_embed.cpp).
//
// The older build_ple (token table + projection at block entry) was replaced
// on 2026-10-07 — see ple.h for why.
//
// Build target: ple-tests (see tests/CMakeLists.txt).

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "../../src/layers/ple.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

namespace {

constexpr int   kEmbd = 6, kPle = 3, kLayers = 2, kTok = 4;
constexpr float kEps  = 1e-6f;

std::vector<float> fill(size_t n, float seed) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = std::sin(seed + 0.71f * (float)i);
    return v;
}

struct Weights {
    std::vector<float> x         = fill((size_t)kEmbd * kTok, 0.2f);          // [n_embd, n_tok]
    std::vector<float> cur       = fill((size_t)kEmbd * kTok, 1.1f);          // residual
    std::vector<float> model_proj= fill((size_t)kEmbd * kPle * kLayers, 2.3f);// [n_embd, ple*L]
    std::vector<float> proj_norm = fill(kPle, 3.7f);
    std::vector<float> inp_gate  = fill((size_t)kEmbd * kPle, 4.1f);          // [n_embd, ple]
    std::vector<float> proj      = fill((size_t)kPle * kEmbd, 5.3f);          // [ple, n_embd]
    std::vector<float> post_norm = fill(kEmbd, 6.9f);
};

double gelu_tanh(double x) {
    return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

// ggml mul_mat weight [ne0=in, ne1=out]: out[o] = sum_i w[o*in + i] * x[i].
std::vector<double> matvec(const std::vector<float>& w, int in, int out, const double* x) {
    std::vector<double> y(out, 0.0);
    for (int o = 0; o < out; ++o)
        for (int i = 0; i < in; ++i) y[o] += (double)w[(size_t)o * in + i] * x[i];
    return y;
}

void rms_norm(std::vector<double>& v, const std::vector<float>& w) {
    double ss = 0;
    for (double a : v) ss += a * a;
    const double inv = 1.0 / std::sqrt(ss / (double)v.size() + kEps);
    for (size_t i = 0; i < v.size(); ++i) v[i] = v[i] * inv * (double)w[i];
}

// The ple.h formula for layer `il`, in doubles. Returns [n_embd, n_tok].
std::vector<double> reference(const Weights& w, int il) {
    std::vector<double> out((size_t)kEmbd * kTok);
    for (int t = 0; t < kTok; ++t) {
        std::vector<double> x(kEmbd), c(kEmbd);
        for (int e = 0; e < kEmbd; ++e) { x[e] = w.x[(size_t)t * kEmbd + e]; c[e] = w.cur[(size_t)t * kEmbd + e]; }
        std::vector<double> all = matvec(w.model_proj, kEmbd, kPle * kLayers, x.data());
        std::vector<double> in(all.begin() + il * kPle, all.begin() + (il + 1) * kPle);
        for (double& a : in) a /= std::sqrt((double)kEmbd);
        rms_norm(in, w.proj_norm);
        std::vector<double> g = matvec(w.inp_gate, kEmbd, kPle, c.data());
        for (int p = 0; p < kPle; ++p) g[p] = gelu_tanh(g[p]) * in[p];
        std::vector<double> u = matvec(w.proj, kPle, kEmbd, g.data());
        rms_norm(u, w.post_norm);
        for (int e = 0; e < kEmbd; ++e) out[(size_t)t * kEmbd + e] = c[e] + u[e];
    }
    return out;
}

std::vector<float> run_graph(const Weights& w, int il) {
    ggml_init_params p{ 128 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(p);
    auto t2 = [&](int a, int b) { ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a, b); ggml_set_input(t); return t; };
    ggml_tensor* x   = t2(kEmbd, kTok);
    ggml_tensor* cur = t2(kEmbd, kTok);
    ggml_tensor* mp  = t2(kEmbd, kPle * kLayers);
    ggml_tensor* pn  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kPle);
    ggml_tensor* ig  = t2(kEmbd, kPle);
    ggml_tensor* pr  = t2(kPle, kEmbd);
    ggml_tensor* po  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kEmbd);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_tensor* ins = build_ple_inputs(ctx, gf, x, mp, pn, kPle, kLayers, kEps);
    ggml_tensor* out = build_ple_inject(ctx, gf, cur, ins, ig, pr, po, kEps, il);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    auto set = [](ggml_tensor* t, const std::vector<float>& v) {
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    };
    set(x, w.x); set(cur, w.cur); set(mp, w.model_proj); set(pn, w.proj_norm);
    set(ig, w.inp_gate); set(pr, w.proj); set(po, w.post_norm);
    EXPECT_EQ(ggml_backend_graph_compute(be, gf), GGML_STATUS_SUCCESS);
    std::vector<float> got((size_t)kEmbd * kTok);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return got;
}

}  // namespace

// Both layers, so the per-layer slice of the shared inputs is exercised: layer 1
// reads the second ple_dim block of model_proj's output, not the first.
// Tolerance covers ggml-cpu's F16 GELU table.
TEST(Ple, InjectMatchesTheFormulaOnEveryLayer) {
    const Weights w;
    for (int il = 0; il < kLayers; ++il) {
        const std::vector<float>  got  = run_graph(w, il);
        const std::vector<double> want = reference(w, il);
        for (size_t i = 0; i < got.size(); ++i)
            EXPECT_NEAR(got[i], want[i], 5e-3) << "layer " << il << " i=" << i;
    }
}

// Different layers inject different updates — a slice that ignored `il`
// would make these equal.
TEST(Ple, LayersReadTheirOwnSlice) {
    const Weights w;
    EXPECT_NE(run_graph(w, 0), run_graph(w, 1));
}

TEST(Ple, RefusesAModelProjOfTheWrongWidth) {
    ggml_init_params p{ 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ggml_tensor* x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kEmbd, kTok);
    ggml_tensor* mp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kEmbd, kPle * kLayers + 1);
    ggml_tensor* pn = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kPle);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    EXPECT_THROW(build_ple_inputs(ctx, gf, x, mp, pn, kPle, kLayers, kEps), std::runtime_error);
    ggml_free(ctx);
}

TEST(Ple, RefusesALayerOutsideTheInputs) {
    ggml_init_params p{ 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ggml_tensor* cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kEmbd, kTok);
    ggml_tensor* ins = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kPle, kTok, kLayers);
    ggml_tensor* ig  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kEmbd, kPle);
    ggml_tensor* pr  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kPle, kEmbd);
    ggml_tensor* po  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kEmbd);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    EXPECT_THROW(build_ple_inject(ctx, gf, cur, ins, ig, pr, po, kEps, kLayers), std::runtime_error);
    ggml_free(ctx);
}
