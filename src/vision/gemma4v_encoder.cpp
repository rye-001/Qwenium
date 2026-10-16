#include "gemma4v_encoder.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "bitmap.h"
#include "vision_model.h"

namespace qinf::vision {

namespace {

// Per layer ≈ 45 nodes (3 norms + 4 projections + QK-norm + 2×2 ropes + V-norm
// + attention + gated FFN + 2 residuals); 16 layers ≈ 720, plus the prologue,
// pooler and embedder. Same headroom style as the other encoders.
constexpr size_t VIT_GRAPH_SIZE = 4096;

// The 2-D RoPE's frequency base (config.json rope_parameters.rope_theta;
// llama.cpp hardcodes the same 100 for gemma4v).
constexpr float kRopeTheta = 100.0f;

ggml_tensor* require(const VisionModel& model, const std::string& name) {
    const auto& tensors = model.tensors();
    auto it = tensors.find(name);
    if (it == tensors.end() || it->second == nullptr)
        throw std::runtime_error(
            "gemma4v_encoder: slot \"tensor:" + name +
            "\" expected present-in-mmproj, actual: absent");
    return it->second;
}

// RMS norm with a weight (llama.cpp build_norm, NORM_TYPE_RMS).
ggml_tensor* rms_w(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// The 2-D NeoX RoPE over one head-major tensor [d_head, n_head, n_pos]: the
// first d_head/2 dims rotate by column (pos_x), the second d_head/2 by row
// (pos_y). The second rope's dims are shifted by d_head/2 with
// ggml_rope_set_offset — op for op what gemma4v.cpp's add_pos does.
ggml_tensor* rope_2d(ggml_context* ctx, ggml_tensor* cur,
                     ggml_tensor* pos_x, ggml_tensor* pos_y) {
    const int n_dim = static_cast<int>(cur->ne[0]);
    cur = ggml_rope_ext(ctx, cur, pos_x, nullptr, n_dim / 2, GGML_ROPE_TYPE_NEOX, 0,
                        kRopeTheta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    cur = ggml_rope_ext(ctx, cur, pos_y, nullptr, n_dim / 2, GGML_ROPE_TYPE_NEOX, 0,
                        kRopeTheta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    return ggml_rope_set_offset(cur, n_dim / 2);
}

}  // namespace

Gemma4vEncoder::Gemma4vEncoder(const VisionModel& model,
                               ggml_backend_t     backend,
                               uint32_t           text_embed_dim)
    : model_(model), backend_(backend), text_embed_dim_(text_embed_dim) {
    if (backend_ == nullptr)
        throw std::runtime_error(
            "gemma4v_encoder: slot \"backend\" expected non-null ggml_backend_t "
            "(shared with the text engine), actual: nullptr");
    if (model_.config().projector_type != VisionProjectorType::Gemma4V)
        throw std::runtime_error(
            "gemma4v_encoder: slot \"projector_type\" expected Gemma4V, actual: another "
            "projector (mmproj '" + model_.mmproj_path() + "')");
    if (model_.config().projection_dim != text_embed_dim_)
        throw std::runtime_error(
            "gemma4v_encoder: slot \"projection_dim\" expected " +
            std::to_string(text_embed_dim_) + " (host text model width), actual: " +
            std::to_string(model_.config().projection_dim) +
            " (mmproj clip.vision.projection_dim). The mmproj does not match this text model.");

    const bool primary_is_cpu =
        ggml_backend_dev_type(ggml_backend_get_device(backend_)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_backend_t backends[2];
    int n_backends = 0;
    backends[n_backends++] = backend_;
    if (!primary_is_cpu) {
        cpu_fallback_ = ggml_backend_cpu_init();
        if (cpu_fallback_ == nullptr)
            throw std::runtime_error(
                "gemma4v_encoder: slot \"cpu_fallback\" expected non-null from "
                "ggml_backend_cpu_init, actual: nullptr");
        backends[n_backends++] = cpu_fallback_;
    }
    scheduler_ = ggml_backend_sched_new(backends, nullptr, n_backends, VIT_GRAPH_SIZE, false, false);
    if (scheduler_ == nullptr)
        throw std::runtime_error(
            "gemma4v_encoder: slot \"scheduler\" expected non-null from "
            "ggml_backend_sched_new, actual: nullptr");
}

Gemma4vEncoder::~Gemma4vEncoder() {
    if (scheduler_ != nullptr)    { ggml_backend_sched_free(scheduler_); scheduler_ = nullptr; }
    if (graph_ctx_ != nullptr)    { ggml_free(graph_ctx_); graph_ctx_ = nullptr; }
    if (cpu_fallback_ != nullptr) { ggml_backend_free(cpu_fallback_); cpu_fallback_ = nullptr; }
}

void Gemma4vEncoder::patch_grid(const Bitmap& bitmap, int64_t& px, int64_t& py) const {
    const SigLIPConfig& cfg = model_.config();
    if (bitmap.channels != static_cast<int>(cfg.num_channels))
        throw std::runtime_error(
            "gemma4v_encoder: slot \"bitmap.channels\" expected " +
            std::to_string(cfg.num_channels) + ", actual: " + std::to_string(bitmap.channels));
    const int64_t eff = static_cast<int64_t>(cfg.patch_size) * cfg.n_merge;
    if (eff <= 0 || bitmap.width % eff != 0 || bitmap.height % eff != 0)
        throw std::runtime_error(
            "gemma4v_encoder: slot \"bitmap.{width,height}\" expected multiples of "
            "patch_size*n_merge (" + std::to_string(eff) + "), actual: " +
            std::to_string(bitmap.width) + "x" + std::to_string(bitmap.height) +
            ". Preprocess with gemma4v_preprocess.");
    const size_t expect = static_cast<size_t>(cfg.num_channels) * bitmap.width * bitmap.height;
    if (bitmap.pixels.size() != expect)
        throw std::runtime_error(
            "gemma4v_encoder: slot \"bitmap.pixels.size\" expected " + std::to_string(expect) +
            " (C*H*W), actual: " + std::to_string(bitmap.pixels.size()));
    px = bitmap.width  / static_cast<int64_t>(cfg.patch_size);
    py = bitmap.height / static_cast<int64_t>(cfg.patch_size);
}

uint32_t Gemma4vEncoder::mm_tokens_for(const Bitmap& bitmap) const {
    uint32_t nx = 0, ny = 0;
    mm_grid_for(bitmap, nx, ny);
    return nx * ny;
}

void Gemma4vEncoder::mm_grid_for(const Bitmap& bitmap, uint32_t& nx, uint32_t& ny) const {
    int64_t px = 0, py = 0;
    patch_grid(bitmap, px, py);
    const int64_t k = static_cast<int64_t>(model_.config().n_merge);
    nx = static_cast<uint32_t>(px / k);
    ny = static_cast<uint32_t>(py / k);
}

uint32_t Gemma4vEncoder::projection_dim() const { return model_.config().projection_dim; }

std::vector<float> Gemma4vEncoder::encode(const Bitmap& bitmap) {
    const SigLIPConfig& cfg = model_.config();
    int64_t px = 0, py = 0;
    patch_grid(bitmap, px, py);

    const int64_t n_embd  = static_cast<int64_t>(cfg.hidden_size);
    const int64_t n_head  = static_cast<int64_t>(cfg.num_attn_heads);
    const int64_t d_head  = n_embd / n_head;
    const int64_t patch   = static_cast<int64_t>(cfg.patch_size);
    const int64_t k       = static_cast<int64_t>(cfg.n_merge);
    const int64_t n_pos   = px * py;
    const int64_t out_x   = px / k, out_y = py / k;
    const int64_t n_out   = out_x * out_y;
    const float   eps     = cfg.layer_norm_eps;
    const int     n_layer = static_cast<int>(cfg.num_layers);

    ggml_tensor* pos_tbl = require(model_, "v.position_embd.weight");
    const int64_t pos_size = pos_tbl->ne[1];
    if (pos_tbl->ne[0] != n_embd || pos_tbl->ne[2] != 2)
        throw std::runtime_error(
            "gemma4v_encoder: slot \"v.position_embd.weight.shape\" expected [" +
            std::to_string(n_embd) + ", N, 2], actual: [" + std::to_string(pos_tbl->ne[0]) +
            ", " + std::to_string(pos_tbl->ne[1]) + ", " + std::to_string(pos_tbl->ne[2]) + "]");
    if (px > pos_size || py > pos_size)
        throw std::runtime_error(
            "gemma4v_encoder: slot \"patch grid\" expected each side <= " +
            std::to_string(pos_size) + " (position table), actual: " +
            std::to_string(px) + "x" + std::to_string(py));
    if (d_head % 2 != 0)
        throw std::runtime_error(
            "gemma4v_encoder: slot \"d_head\" expected even (2-D RoPE halves), actual: " +
            std::to_string(d_head));

    if (graph_ctx_ != nullptr) { ggml_free(graph_ctx_); graph_ctx_ = nullptr; }
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * VIT_GRAPH_SIZE + ggml_graph_overhead_custom(VIT_GRAPH_SIZE, false);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    graph_ctx_ = ggml_init(ip);
    if (graph_ctx_ == nullptr)
        throw std::runtime_error(
            "gemma4v_encoder: slot \"graph_ctx\" expected non-null from ggml_init, actual: nullptr");
    ggml_context* ctx = graph_ctx_;
    ggml_cgraph*  gf  = ggml_new_graph_custom(ctx, VIT_GRAPH_SIZE, false);

    // ── Patchify: pixels in [0, 1] → 2x − 1, conv (no bias) ─────────────────
    // ggml conv2d reads [W, H, C]: the Bitmap's channel-planar memory order.
    ggml_tensor* inp_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, bitmap.width, bitmap.height,
                                              cfg.num_channels);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);
    ggml_tensor* x = ggml_scale_bias(ctx, inp_raw, 2.0f, -1.0f);

    ggml_tensor* pw = require(model_, "v.patch_embd.weight");
    // Metal's im2col emits only F16/F32 (see SiglipEncoder); BF16 → F32 is exact.
    if (pw->type != GGML_TYPE_F32 && pw->type != GGML_TYPE_F16) pw = ggml_cast(ctx, pw, GGML_TYPE_F32);
    ggml_tensor* inp = ggml_conv_2d(ctx, pw, x, patch, patch, 0, 0, 1, 1);  // [px, py, n_embd, 1]
    inp = ggml_reshape_2d(ctx, inp, n_pos, n_embd);
    inp = ggml_cont(ctx, ggml_transpose(ctx, inp));                         // [n_embd, n_pos], x fastest

    // ── Learned 2-D positions: table x then table y ─────────────────────────
    ggml_tensor* pos_x = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_x, "pos_x");
    ggml_set_input(pos_x);
    ggml_tensor* pos_y = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_y, "pos_y");
    ggml_set_input(pos_y);
    {
        const size_t nb1 = ggml_row_size(pos_tbl->type, n_embd);
        ggml_tensor* tbl_x = ggml_view_2d(ctx, pos_tbl, n_embd, pos_size, nb1, 0);
        ggml_tensor* tbl_y = ggml_view_2d(ctx, pos_tbl, n_embd, pos_size, nb1, pos_size * nb1);
        inp = ggml_add(ctx, inp, ggml_get_rows(ctx, tbl_x, pos_x));
        inp = ggml_add(ctx, inp, ggml_get_rows(ctx, tbl_y, pos_y));
    }

    // ── 16 Gemma-style blocks (no pre/post LN on this tower) ────────────────
    ggml_tensor* inpL = inp;
    for (int il = 0; il < n_layer; ++il) {
        const std::string p = "v.blk." + std::to_string(il) + ".";

        ggml_tensor* cur = rms_w(ctx, inpL, require(model_, p + "ln1.weight"), eps);

        ggml_tensor* Q = ggml_mul_mat(ctx, require(model_, p + "attn_q.weight"), cur);
        ggml_tensor* K = ggml_mul_mat(ctx, require(model_, p + "attn_k.weight"), cur);
        ggml_tensor* V = ggml_mul_mat(ctx, require(model_, p + "attn_v.weight"), cur);
        Q = ggml_reshape_3d(ctx, Q, d_head, n_head, n_pos);
        K = ggml_reshape_3d(ctx, K, d_head, n_head, n_pos);
        V = ggml_reshape_3d(ctx, V, d_head, n_head, n_pos);
        // Per-head QK-norm (the norm weight is d_head wide).
        Q = rms_w(ctx, Q, require(model_, p + "attn_q_norm.weight"), eps);
        K = rms_w(ctx, K, require(model_, p + "attn_k_norm.weight"), eps);
        Q = rope_2d(ctx, Q, pos_x, pos_y);
        K = rope_2d(ctx, K, pos_x, pos_y);
        // V-norm: gemma4v only — an RMS norm per head with no weight.
        V = ggml_rms_norm(ctx, V, eps);

        // Bidirectional, unmasked, kq_scale 1 (QK-norm replaces 1/√d). Flash
        // attention: K/V in F16, F32 accumulation (architecture §7).
        ggml_tensor* q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        ggml_tensor* kk = ggml_cast(ctx, ggml_permute(ctx, K, 0, 2, 1, 3), GGML_TYPE_F16);
        ggml_tensor* v  = ggml_cast(ctx, ggml_permute(ctx, V, 0, 2, 1, 3), GGML_TYPE_F16);
        cur = ggml_flash_attn_ext(ctx, q, kk, v, /*mask=*/nullptr, 1.0f, 0.0f, 0.0f);
        ggml_prec_set_acc(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx, cur, n_embd, n_pos);
        cur = ggml_mul_mat(ctx, require(model_, p + "attn_out.weight"), cur);
        cur = rms_w(ctx, cur, require(model_, p + "attn_post_norm.weight"), eps);
        inpL = ggml_add(ctx, cur, inpL);                                     // residual 1

        ggml_tensor* f  = rms_w(ctx, inpL, require(model_, p + "ln2.weight"), eps);
        ggml_tensor* up = ggml_mul_mat(ctx, require(model_, p + "ffn_up.weight"), f);
        ggml_tensor* gt = ggml_mul_mat(ctx, require(model_, p + "ffn_gate.weight"), f);
        f = cfg.ffn_act == VisionFfnAct::GeluTanh ? ggml_geglu_split(ctx, gt, up)
                                                  : ggml_geglu_quick_split(ctx, gt, up);
        f = ggml_mul_mat(ctx, require(model_, p + "ffn_down.weight"), f);
        f = rms_w(ctx, f, require(model_, p + "ffn_post_norm.weight"), eps);
        inpL = ggml_add(ctx, inpL, f);                                       // residual 2
    }

    // ── Gemma4VisionPooler: k×k average pool on the patch grid, × √n_embd ───
    ggml_tensor* out = ggml_cont_4d(ctx, ggml_transpose(ctx, inpL), px, py, n_embd, 1);
    out = ggml_pool_2d(ctx, out, GGML_OP_POOL_AVG, static_cast<int>(k), static_cast<int>(k),
                       static_cast<int>(k), static_cast<int>(k), 0, 0);
    out = ggml_reshape_2d(ctx, out, n_out, n_embd);
    out = ggml_cont(ctx, ggml_transpose(ctx, out));                          // [n_embd, n_out]
    out = ggml_scale(ctx, out, std::sqrt(static_cast<float>(n_embd)));

    // ── Gemma4MultimodalEmbedder: RMS (no weight), project to the text width ─
    out = ggml_rms_norm(ctx, out, eps);
    out = ggml_mul_mat(ctx, require(model_, "mm.input_projection.weight"), out);

    ggml_set_name(out, "image_embeddings");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_backend_sched_reset(scheduler_);
    if (!ggml_backend_sched_alloc_graph(scheduler_, gf))
        throw std::runtime_error(
            "gemma4v_encoder: slot \"alloc_graph\" expected success from "
            "ggml_backend_sched_alloc_graph, actual: failure");

    ggml_backend_tensor_set(inp_raw, bitmap.pixels.data(), 0, bitmap.pixels.size() * sizeof(float));
    {
        // (column, row) of each patch, x fastest — gemma4v set_input in clip.cpp.
        std::vector<int32_t> xs(static_cast<size_t>(n_pos)), ys(static_cast<size_t>(n_pos));
        for (int64_t i = 0; i < n_pos; ++i) {
            xs[i] = static_cast<int32_t>(i % px);
            ys[i] = static_cast<int32_t>(i / px);
        }
        ggml_backend_tensor_set(pos_x, xs.data(), 0, xs.size() * sizeof(int32_t));
        ggml_backend_tensor_set(pos_y, ys.data(), 0, ys.size() * sizeof(int32_t));
    }

    const ggml_status st = ggml_backend_sched_graph_compute(scheduler_, gf);
    if (st != GGML_STATUS_SUCCESS)
        throw std::runtime_error(
            "gemma4v_encoder: slot \"graph_compute\" expected GGML_STATUS_SUCCESS, actual: status " +
            std::to_string(st));

    const int64_t n_elem = ggml_nelements(out);
    const int64_t expect = static_cast<int64_t>(projection_dim()) * n_out;
    if (n_elem != expect)
        throw std::runtime_error(
            "gemma4v_encoder: slot \"image_embeddings.nelements\" expected " +
            std::to_string(expect) + " (projection_dim * mm_tokens), actual: " + std::to_string(n_elem));
    std::vector<float> host(static_cast<size_t>(n_elem));
    ggml_backend_tensor_get(out, host.data(), 0, host.size() * sizeof(float));
    return host;
}

}  // namespace qinf::vision
