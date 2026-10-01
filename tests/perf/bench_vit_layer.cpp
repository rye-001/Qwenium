// One Qwen3-VL vision-tower layer at a document page's shape, timed in parts
// on Metal — where do the encoder's ~10 s per page go?
//
// Shape (Qwen3.6 mmproj): width 1152, 16 heads x 72, FFN 4304, BF16 weights;
// 1024 x 1440 px at patch 16 = 64 x 90 = 5760 patches. The tower has 27 such
// layers. Parts, each its own graph, median of REPS computes:
//   dense  — QKV, attention-out, FFN up/GELU/down (+ biases), as the encoder
//   attn   — the encoder's attention: K·Q, soft_max, V·KQ, materialized
//   fa32   — ggml_flash_attn_ext, K/V F32
//   fa16   — ggml_flash_attn_ext, K/V cast to F16
// and the max |diff| of fa32 / fa16 against attn on the same random Q/K/V.
// Weights and activations are random: this measures time, not the model.
//
//   ./build-metal/bin/bench-vit-layer            (N_POS=5760 REPS=5 by default)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace {

const int64_t D = 1152, H = 16, DH = 72, FF = 4304;

void fill_f32(ggml_tensor* t, std::mt19937& rng, float scale) {
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> v((size_t)ggml_nelements(t));
    for (float& x : v) x = nd(rng);
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}

void fill_bf16(ggml_tensor* t, std::mt19937& rng, float scale) {   // any weight dtype (WTYPE)
    std::normal_distribution<float> nd(0.0f, scale);
    const size_t n = (size_t)ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) { fill_f32(t, rng, scale); return; }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> v(n);
        for (ggml_fp16_t& x : v) x = ggml_fp32_to_fp16(nd(rng));
        ggml_backend_tensor_set(t, v.data(), 0, n * sizeof(ggml_fp16_t));
        return;
    }
    std::vector<ggml_bf16_t> v(n);
    for (ggml_bf16_t& x : v) x = ggml_fp32_to_bf16(nd(rng));
    ggml_backend_tensor_set(t, v.data(), 0, n * sizeof(ggml_bf16_t));
}

struct Timed { double median_ms; std::vector<float> out; };

// Build with `make`, allocate, compute REPS times; return the median and the output.
template <typename Make>
Timed run(ggml_backend_t be, ggml_context* wctx_unused, Make make, int reps) {
    (void)wctx_unused;
    ggml_init_params ip = {ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_tensor* out = make(ctx);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(ga, gf)) throw std::runtime_error("bench-vit-layer: gallocr_alloc_graph failed");
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i)
        if (!ggml_backend_supports_op(be, ggml_graph_node(gf, i)))
            throw std::runtime_error(std::string("bench-vit-layer: op expected on Metal, actual unsupported: ") +
                                     ggml_op_desc(ggml_graph_node(gf, i)));
    std::vector<double> ms;
    ggml_backend_graph_compute(be, gf);   // warm-up (pipeline compile)
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("bench-vit-layer: compute failed");
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    Timed t;
    t.median_ms = ms[ms.size() / 2];
    t.out.resize((size_t)ggml_nelements(out));
    ggml_backend_tensor_get(out, t.out.data(), 0, t.out.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return t;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace

int main() {
    const int64_t N = std::getenv("N_POS") ? std::atoll(std::getenv("N_POS")) : 5760;
    const int reps = std::getenv("REPS") ? std::atoi(std::getenv("REPS")) : 5;
    const std::string wt_name = std::getenv("WTYPE") ? std::getenv("WTYPE") : "bf16";
    const ggml_type WT = wt_name == "f16" ? GGML_TYPE_F16 : wt_name == "f32" ? GGML_TYPE_F32 : GGML_TYPE_BF16;

    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!be) throw std::runtime_error("bench-vit-layer: GPU backend expected, actual none");
    std::printf("backend %s, N_POS=%lld, REPS=%d, weights %s\n", ggml_backend_name(be), (long long)N, reps,
                ggml_type_name(WT));

    // Persistent tensors: inputs and one layer's weights.
    ggml_init_params wp = {ggml_tensor_overhead() * 32, nullptr, true};
    ggml_context* w = ggml_init(wp);
    ggml_tensor* x    = ggml_new_tensor_2d(w, GGML_TYPE_F32, D, N);
    ggml_tensor* wqkv = ggml_new_tensor_2d(w, WT, D, 3 * D);
    ggml_tensor* bqkv = ggml_new_tensor_1d(w, GGML_TYPE_F32, 3 * D);
    ggml_tensor* wo   = ggml_new_tensor_2d(w, WT, D, D);
    ggml_tensor* bo   = ggml_new_tensor_1d(w, GGML_TYPE_F32, D);
    ggml_tensor* wup  = ggml_new_tensor_2d(w, WT, D, FF);
    ggml_tensor* bup  = ggml_new_tensor_1d(w, GGML_TYPE_F32, FF);
    ggml_tensor* wdn  = ggml_new_tensor_2d(w, WT, FF, D);
    ggml_tensor* bdn  = ggml_new_tensor_1d(w, GGML_TYPE_F32, D);
    ggml_tensor* qkv  = ggml_new_tensor_2d(w, GGML_TYPE_F32, 3 * D, N);   // a fixed QKV for the attention parts
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(w, be);
    std::mt19937 rng(42);
    fill_f32(x, rng, 1.0f);
    fill_bf16(wqkv, rng, 0.03f); fill_f32(bqkv, rng, 0.01f);
    fill_bf16(wo, rng, 0.03f);   fill_f32(bo, rng, 0.01f);
    fill_bf16(wup, rng, 0.03f);  fill_f32(bup, rng, 0.01f);
    fill_bf16(wdn, rng, 0.015f); fill_f32(bdn, rng, 0.01f);
    fill_f32(qkv, rng, 1.0f);
    const float scale = 1.0f / std::sqrt((float)DH);

    auto views = [&](ggml_context* ctx, ggml_tensor*& Q, ggml_tensor*& K, ggml_tensor*& V) {
        Q = ggml_view_3d(ctx, qkv, DH, H, N, ggml_row_size(qkv->type, DH), qkv->nb[1], 0);
        K = ggml_view_3d(ctx, qkv, DH, H, N, ggml_row_size(qkv->type, DH), qkv->nb[1], ggml_row_size(qkv->type, D));
        V = ggml_view_3d(ctx, qkv, DH, H, N, ggml_row_size(qkv->type, DH), qkv->nb[1], ggml_row_size(qkv->type, 2 * D));
    };

    const Timed dense = run(be, w, [&](ggml_context* ctx) {
        ggml_tensor* c = ggml_add(ctx, ggml_mul_mat(ctx, wqkv, x), bqkv);
        ggml_tensor* a = ggml_view_2d(ctx, c, D, N, c->nb[1], 0);        // stand-in for the attention output
        ggml_tensor* o = ggml_add(ctx, ggml_mul_mat(ctx, wo, ggml_cont(ctx, a)), bo);
        ggml_tensor* f = ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, wup, o), bup));
        return ggml_add(ctx, ggml_mul_mat(ctx, wdn, f), bdn);
    }, reps);

    const Timed attn = run(be, w, [&](ggml_context* ctx) {    // the encoder's exact attention
        ggml_tensor *Q, *K, *V; views(ctx, Q, K, V);
        ggml_tensor* q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        ggml_tensor* k = ggml_permute(ctx, K, 0, 2, 1, 3);
        ggml_tensor* v = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));
        ggml_tensor* kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, k, q), nullptr, scale, 0.0f);
        ggml_tensor* kqv = ggml_permute(ctx, ggml_mul_mat(ctx, v, kq), 0, 2, 1, 3);
        return ggml_cont_2d(ctx, kqv, D, N);
    }, reps);

    auto flash = [&](bool f16) {
        return run(be, w, [&, f16](ggml_context* ctx) {
            ggml_tensor *Q, *K, *V; views(ctx, Q, K, V);
            ggml_tensor* q = ggml_permute(ctx, Q, 0, 2, 1, 3);
            ggml_tensor* k = ggml_permute(ctx, K, 0, 2, 1, 3);
            ggml_tensor* v = ggml_permute(ctx, V, 0, 2, 1, 3);
            if (f16) { k = ggml_cast(ctx, k, GGML_TYPE_F16); v = ggml_cast(ctx, v, GGML_TYPE_F16); }
            else     { k = ggml_cont(ctx, k); v = ggml_cont(ctx, v); }
            ggml_tensor* o = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
            return ggml_reshape_2d(ctx, o, D, N);                        // FA output is [DH, H, N]
        }, reps);
    };
    const Timed fa32 = flash(false);
    const Timed fa16 = flash(true);

    const double total = dense.median_ms + attn.median_ms;
    std::printf("\npart   ms/layer  x27 layers  share\n");
    std::printf("dense  %8.1f  %9.0f  %4.0f%%\n", dense.median_ms, 27 * dense.median_ms, 100 * dense.median_ms / total);
    std::printf("attn   %8.1f  %9.0f  %4.0f%%   (materialized, as the encoder)\n", attn.median_ms,
                27 * attn.median_ms, 100 * attn.median_ms / total);
    std::printf("fa32   %8.1f  %9.0f          max|diff| vs attn %.2e\n", fa32.median_ms, 27 * fa32.median_ms,
                max_abs_diff(fa32.out, attn.out));
    std::printf("fa16   %8.1f  %9.0f          max|diff| vs attn %.2e\n", fa16.median_ms, 27 * fa16.median_ms,
                max_abs_diff(fa16.out, attn.out));
    std::printf("\nlayer as today (dense + attn) x27 = %.0f ms; with fa32 = %.0f ms; with fa16 = %.0f ms\n",
                27 * total, 27 * (dense.median_ms + fa32.median_ms), 27 * (dense.median_ms + fa16.median_ms));

    ggml_backend_buffer_free(wbuf);
    ggml_free(w);
    ggml_backend_free(be);
    return 0;
}
