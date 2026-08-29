#include "qwen3vl_encoder.h"

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

// Node budget. Per layer: ln1 (3) + qkv (2) + 3 views + 2 ropes + attn
// (~8) + o_proj (2) + residual + ln2 (3) + ffn (5) + residual ≈ 30.
// 27 layers ≈ 810, plus the patch/merge/pos-embed prologue and the merger.
// 4096 matches the headroom style of SiglipEncoder's VIT_GRAPH_SIZE.
constexpr size_t VIT_GRAPH_SIZE = 4096;

ggml_tensor* require(const VisionModel& model, const std::string& name) {
    const auto& tensors = model.tensors();
    auto it = tensors.find(name);
    if (it == tensors.end() || it->second == nullptr) {
        throw std::runtime_error(
            "qwen3vl_encoder: slot \"tensor:" + name +
            "\" expected present-in-mmproj, actual: absent");
    }
    return it->second;
}

// The 2×2 spatial-merge reorder, applied identically to the patch embeddings
// and to the resized position embeddings (llama.cpp qwen3vl.cpp lines 18–51).
//
// It rewrites [n_embd, px, py] so that each consecutive run of four rows is
// one 2×2 patch block. That is what lets the merger's closing
// reshape(n_embd*4, n_pos/4) fold a spatial block rather than four unrelated
// patches — the single step that makes the projector's arithmetic meaningful.
ggml_tensor* apply_spatial_merge(ggml_context* ctx, ggml_tensor* t,
                                 int64_t n_embd, int64_t px, int64_t py) {
    t = ggml_cont_4d(ctx, t, n_embd * 2, px / 2, py, 1);
    t = ggml_reshape_4d(ctx, t, n_embd * 2, px / 2, 2, py / 2);
    t = ggml_permute(ctx, t, 0, 2, 1, 3);
    t = ggml_cont_3d(ctx, t, n_embd, px * py, 1);
    return t;
}

}  // namespace

Qwen3VlEncoder::Qwen3VlEncoder(const VisionModel& model,
                               ggml_backend_t     backend,
                               uint32_t           text_embed_dim)
    : model_(model), backend_(backend), text_embed_dim_(text_embed_dim) {
    if (backend_ == nullptr)
        throw std::runtime_error(
            "qwen3vl_encoder: slot \"backend\" expected non-null "
            "ggml_backend_t (shared with the text engine), actual: nullptr");

    // Seam A invariant. Note llama.cpp computes projection_dim × merge² here
    // and refuses to load our pair (ggml-org/llama.cpp#20899, closed "not
    // planned"); mm.2.weight is [n_embd·4, projection_dim], so projection_dim
    // is the merger's OUTPUT width and this comparison is the correct one.
    if (model_.config().projection_dim != 0 &&
        model_.config().projection_dim != text_embed_dim_) {
        throw std::runtime_error(
            std::string("qwen3vl_encoder: slot \"projection_dim\" expected ") +
            std::to_string(text_embed_dim_) +
            " (host text model embedding_length), actual: " +
            std::to_string(model_.config().projection_dim) +
            " (mmproj clip.vision.projection_dim). The mmproj does not match "
            "this text model.");
    }

    const bool primary_is_cpu =
        ggml_backend_dev_type(ggml_backend_get_device(backend_)) ==
        GGML_BACKEND_DEVICE_TYPE_CPU;

    ggml_backend_t backends[2];
    int n_backends = 0;
    backends[n_backends++] = backend_;
    if (!primary_is_cpu) {
        cpu_fallback_ = ggml_backend_cpu_init();
        if (cpu_fallback_ == nullptr)
            throw std::runtime_error(
                "qwen3vl_encoder: slot \"cpu_fallback\" expected non-null from "
                "ggml_backend_cpu_init, actual: nullptr");
        backends[n_backends++] = cpu_fallback_;
    }

    scheduler_ = ggml_backend_sched_new(backends, nullptr, n_backends,
                                        VIT_GRAPH_SIZE, false, false);
    if (scheduler_ == nullptr)
        throw std::runtime_error(
            "qwen3vl_encoder: slot \"scheduler\" expected non-null from "
            "ggml_backend_sched_new, actual: nullptr");
}

Qwen3VlEncoder::~Qwen3VlEncoder() {
    if (scheduler_ != nullptr)    { ggml_backend_sched_free(scheduler_); scheduler_ = nullptr; }
    if (graph_ctx_ != nullptr)    { ggml_free(graph_ctx_); graph_ctx_ = nullptr; }
    if (cpu_fallback_ != nullptr) { ggml_backend_free(cpu_fallback_); cpu_fallback_ = nullptr; }
    // backend_ is not owned.
}

void Qwen3VlEncoder::patch_grid(const Bitmap& bitmap,
                                int64_t& px, int64_t& py) const {
    const SigLIPConfig& cfg = model_.config();

    if (bitmap.channels != static_cast<int>(cfg.num_channels))
        throw std::runtime_error(
            "qwen3vl_encoder: slot \"bitmap.channels\" expected " +
            std::to_string(cfg.num_channels) + ", actual: " +
            std::to_string(bitmap.channels));

    // The reference asserts nx and ny are multiples of patch·2 — the merge
    // step halves the grid, so an odd grid would silently drop a row/column.
    const int64_t eff = static_cast<int64_t>(cfg.patch_size) * cfg.n_merge;
    if (eff <= 0)
        throw std::runtime_error(
            "qwen3vl_encoder: slot \"patch_size*n_merge\" expected > 0, actual: " +
            std::to_string(eff));
    if (bitmap.width % eff != 0 || bitmap.height % eff != 0)
        throw std::runtime_error(
            "qwen3vl_encoder: slot \"bitmap.{width,height}\" expected multiples "
            "of patch_size*n_merge (" + std::to_string(eff) + "), actual: " +
            std::to_string(bitmap.width) + "x" + std::to_string(bitmap.height) +
            ". Preprocessing must align to the merged patch grid.");

    const size_t expect =
        static_cast<size_t>(cfg.num_channels) * bitmap.width * bitmap.height;
    if (bitmap.pixels.size() != expect)
        throw std::runtime_error(
            "qwen3vl_encoder: slot \"bitmap.pixels.size\" expected " +
            std::to_string(expect) + " (C*H*W), actual: " +
            std::to_string(bitmap.pixels.size()));

    px = bitmap.width  / static_cast<int64_t>(cfg.patch_size);
    py = bitmap.height / static_cast<int64_t>(cfg.patch_size);
}

uint32_t Qwen3VlEncoder::mm_tokens_for(const Bitmap& bitmap) const {
    int64_t px = 0, py = 0;
    patch_grid(bitmap, px, py);
    const int64_t merge = static_cast<int64_t>(model_.config().n_merge);
    return static_cast<uint32_t>((px / merge) * (py / merge));
}

void Qwen3VlEncoder::mm_grid_for(const Bitmap& bitmap,
                                 uint32_t& nx, uint32_t& ny) const {
    int64_t px = 0, py = 0;
    patch_grid(bitmap, px, py);
    const int64_t merge = static_cast<int64_t>(model_.config().n_merge);
    nx = static_cast<uint32_t>(px / merge);
    ny = static_cast<uint32_t>(py / merge);
}

uint32_t Qwen3VlEncoder::projection_dim() const {
    return model_.config().projection_dim != 0 ? model_.config().projection_dim
                                               : text_embed_dim_;
}

std::vector<float> Qwen3VlEncoder::encode(const Bitmap& bitmap) {
    const SigLIPConfig& cfg = model_.config();

    int64_t px = 0, py = 0;
    patch_grid(bitmap, px, py);

    const int64_t n_embd = static_cast<int64_t>(cfg.hidden_size);
    const int64_t n_head = static_cast<int64_t>(cfg.num_attn_heads);
    if (n_head == 0 || n_embd % n_head != 0)
        throw std::runtime_error(
            "qwen3vl_encoder: slot \"hidden_size/num_attn_heads\" expected "
            "hidden_size divisible by head count, actual: " +
            std::to_string(n_embd) + " / " + std::to_string(n_head));

    const int64_t d_head  = n_embd / n_head;             // 1152/16 = 72
    const int64_t patch   = static_cast<int64_t>(cfg.patch_size);
    const int64_t merge   = static_cast<int64_t>(cfg.n_merge);
    const int64_t n_pos   = px * py;                     // patch tokens
    const int64_t n_out   = n_pos / (merge * merge);     // soft tokens out
    const float   eps     = cfg.layer_norm_eps;
    const float   kq_scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    const int     n_layer = static_cast<int>(cfg.num_layers);

    // ggml_rope_multi in VISION mode asserts n_dims == ne0/2, and
    // ggml_mrope_cache_init asserts the section widths fit ne0. Both hold for
    // {d_head/4}×4 against ne0 = d_head, but only if d_head is a multiple of 4.
    if (d_head % 4 != 0)
        throw std::runtime_error(
            "qwen3vl_encoder: slot \"d_head\" expected a multiple of 4 "
            "(M-RoPE vision sections are d_head/4), actual: " +
            std::to_string(d_head));

    // ── Fresh metadata-only graph context ────────────────────────────────────
    if (graph_ctx_ != nullptr) { ggml_free(graph_ctx_); graph_ctx_ = nullptr; }
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * VIT_GRAPH_SIZE +
                    ggml_graph_overhead_custom(VIT_GRAPH_SIZE, false);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    graph_ctx_ = ggml_init(ip);
    if (graph_ctx_ == nullptr)
        throw std::runtime_error(
            "qwen3vl_encoder: slot \"graph_ctx\" expected non-null from "
            "ggml_init, actual: nullptr");
    ggml_context* ctx = graph_ctx_;
    ggml_cgraph*  gf  = ggml_new_graph_custom(ctx, VIT_GRAPH_SIZE, false);

    // ── Patch embedding: the temporal-merge conv PAIR, summed ────────────────
    // ggml conv2d reads [W, H, C]; the Bitmap is channel-planar [C, H, W]
    // contiguous, which is that exact memory order.
    ggml_tensor* inp_raw = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, bitmap.width, bitmap.height, cfg.num_channels);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    ggml_tensor* pw0 = require(model_, "v.patch_embd.weight");
    ggml_tensor* pw1 = require(model_, "v.patch_embd.weight.1");
    // Same Metal constraint SiglipEncoder documents: ggml_conv_2d lowers to
    // IM2COL + MUL_MAT and im2col takes the KERNEL's dtype, but Metal's im2col
    // emits only F16/F32 — a BF16 kernel silently splits the graph to CPU at
    // the very first op. BF16→F32 is an exact widening, so this does not move
    // us off the reference.
    if (pw0->type != GGML_TYPE_F32 && pw0->type != GGML_TYPE_F16)
        pw0 = ggml_cast(ctx, pw0, GGML_TYPE_F32);
    if (pw1->type != GGML_TYPE_F32 && pw1->type != GGML_TYPE_F16)
        pw1 = ggml_cast(ctx, pw1, GGML_TYPE_F32);

    // A still image feeds BOTH convs (llama.cpp qwen2vl.cpp, n_batch == 1).
    ggml_tensor* inp = ggml_add(
        ctx,
        ggml_conv_2d(ctx, pw0, inp_raw, patch, patch, 0, 0, 1, 1),
        ggml_conv_2d(ctx, pw1, inp_raw, patch, patch, 0, 0, 1, 1));

    // [px, py, n_embd, 1] → [n_embd, px, py, 1], then the 2×2 merge reorder.
    inp = ggml_permute(ctx, inp, 1, 2, 0, 3);
    inp = apply_spatial_merge(ctx, inp, n_embd, px, py);

    inp = ggml_add(ctx, inp, require(model_, "v.patch_embd.bias"));

    // ── Learned position embeddings, bilinearly resized to THIS grid ─────────
    // image_size is the pos-embed grid side (768/16 = 48 ⇒ 2304 entries), not
    // an input-size requirement.
    {
        ggml_tensor* pos_embd = require(model_, "v.position_embd.weight");
        const int64_t n_per_side = static_cast<int64_t>(cfg.image_size) / patch;
        if (n_per_side * n_per_side != pos_embd->ne[1])
            throw std::runtime_error(
                "qwen3vl_encoder: slot \"v.position_embd.weight.ne[1]\" expected " +
                std::to_string(n_per_side * n_per_side) +
                " ((image_size/patch_size)^2), actual: " +
                std::to_string(pos_embd->ne[1]));

        if (px != n_per_side || py != n_per_side) {
            pos_embd = ggml_reshape_3d(ctx, pos_embd, n_embd, n_per_side, n_per_side);
            pos_embd = ggml_permute(ctx, pos_embd, 2, 0, 1, 3);
            pos_embd = ggml_interpolate(
                ctx, pos_embd, px, py, n_embd, 1,
                GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
            pos_embd = ggml_permute(ctx, pos_embd, 1, 2, 0, 3);
            pos_embd = ggml_cont_2d(ctx, pos_embd, n_embd, px * py);
        }
        // The position embeddings get the SAME merge reorder as the patches,
        // so they stay aligned row-for-row after the block shuffle.
        pos_embd = apply_spatial_merge(ctx, pos_embd, n_embd, px, py);
        inp = ggml_add(ctx, inp, pos_embd);
    }

    ggml_tensor* inpL = inp;

    // ── ViT M-RoPE positions: 4 components per patch, component-major ────────
    ggml_tensor* positions =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pos * 4);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    int mrope_sections[GGML_MROPE_SECTIONS] = {
        static_cast<int>(d_head / 4), static_cast<int>(d_head / 4),
        static_cast<int>(d_head / 4), static_cast<int>(d_head / 4),
    };

    // ── Transformer layers (pre-norm; NO pre_ln on this tower) ───────────────
    for (int il = 0; il < n_layer; ++il) {
        const std::string p = "v.blk." + std::to_string(il) + ".";

        ggml_tensor* cur = ggml_norm(ctx, inpL, eps);
        cur = ggml_mul(ctx, cur, require(model_, p + "ln1.weight"));
        cur = ggml_add(ctx, cur, require(model_, p + "ln1.bias"));

        // Fused QKV, then three strided views — one projection, not three.
        ggml_tensor* qkv = ggml_mul_mat(ctx, require(model_, p + "attn_qkv.weight"), cur);
        qkv = ggml_add(ctx, qkv, require(model_, p + "attn_qkv.bias"));

        ggml_tensor* Qcur = ggml_view_3d(ctx, qkv, d_head, n_head, n_pos,
            ggml_row_size(qkv->type, d_head), qkv->nb[1], 0);
        ggml_tensor* Kcur = ggml_view_3d(ctx, qkv, d_head, n_head, n_pos,
            ggml_row_size(qkv->type, d_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd));
        ggml_tensor* Vcur = ggml_view_3d(ctx, qkv, d_head, n_head, n_pos,
            ggml_row_size(qkv->type, d_head), qkv->nb[1],
            ggml_row_size(qkv->type, 2 * n_embd));

        // VISION-mode M-RoPE: n_dims = d_head/2 (the mode asserts
        // n_dims == ne0/2), freq_base 10000, n_ctx_orig 32768 — the reference's
        // literals, which are ViT constants and not read from metadata.
        Qcur = ggml_rope_multi(ctx, Qcur, positions, nullptr,
                               static_cast<int>(d_head / 2), mrope_sections,
                               GGML_ROPE_TYPE_VISION, 32768, 10000.0f,
                               1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        Kcur = ggml_rope_multi(ctx, Kcur, positions, nullptr,
                               static_cast<int>(d_head / 2), mrope_sections,
                               GGML_ROPE_TYPE_VISION, 32768, 10000.0f,
                               1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // Attention: bidirectional over the whole image, no mask.
        ggml_tensor* q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
        ggml_tensor* k = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
        ggml_tensor* v = ggml_cont(ctx, ggml_permute(ctx, Vcur, 1, 2, 0, 3));

        ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
        kq = ggml_soft_max_ext(ctx, kq, /*mask=*/nullptr, kq_scale, 0.0f);

        ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);
        cur = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);

        cur = ggml_mul_mat(ctx, require(model_, p + "attn_out.weight"), cur);
        cur = ggml_add(ctx, cur, require(model_, p + "attn_out.bias"));

        inpL = ggml_add(ctx, cur, inpL);   // residual 1

        ggml_tensor* f = ggml_norm(ctx, inpL, eps);
        f = ggml_mul(ctx, f, require(model_, p + "ln2.weight"));
        f = ggml_add(ctx, f, require(model_, p + "ln2.bias"));

        // Gate-less GELU MLP (clip.use_gelu = true; no ffn_gate tensors exist).
        f = ggml_mul_mat(ctx, require(model_, p + "ffn_up.weight"), f);
        f = ggml_add(ctx, f, require(model_, p + "ffn_up.bias"));
        f = ggml_gelu(ctx, f);
        f = ggml_mul_mat(ctx, require(model_, p + "ffn_down.weight"), f);
        f = ggml_add(ctx, f, require(model_, p + "ffn_down.bias"));

        inpL = ggml_add(ctx, inpL, f);     // residual 2
    }

    // ── post_ln (this tower has post only) ───────────────────────────────────
    ggml_tensor* out = ggml_norm(ctx, inpL, eps);
    out = ggml_mul(ctx, out, require(model_, "v.post_ln.weight"));
    out = ggml_add(ctx, out, require(model_, "v.post_ln.bias"));

    // ── The 2×2 merger ───────────────────────────────────────────────────────
    // Folds four spatially-adjacent patches into one row (the merge reorder
    // above is what makes those four adjacent), then a GELU MLP down to the
    // text embedding width.
    out = ggml_reshape_3d(ctx, out, n_embd * merge * merge, n_out, 1);
    out = ggml_mul_mat(ctx, require(model_, "mm.0.weight"), out);
    out = ggml_add(ctx, out, require(model_, "mm.0.bias"));
    out = ggml_gelu(ctx, out);
    out = ggml_mul_mat(ctx, require(model_, "mm.2.weight"), out);
    out = ggml_add(ctx, out, require(model_, "mm.2.bias"));

    ggml_set_name(out, "image_embeddings");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    // ── Allocate, upload inputs, compute ─────────────────────────────────────
    ggml_backend_sched_reset(scheduler_);
    if (!ggml_backend_sched_alloc_graph(scheduler_, gf))
        throw std::runtime_error(
            "qwen3vl_encoder: slot \"alloc_graph\" expected success from "
            "ggml_backend_sched_alloc_graph, actual: failure");

    ggml_backend_tensor_set(inp_raw, bitmap.pixels.data(), 0,
                            bitmap.pixels.size() * sizeof(float));

    // Position ids, emitted in the SAME 2×2 block order as the merge reorder
    // (llama.cpp clip.cpp, PROJECTOR_TYPE_QWEN3VL set_input). Components are
    // (y, x, y, x): t and w carry the row, h and e carry the column.
    {
        std::vector<int32_t> pos(static_cast<size_t>(n_pos) * 4);
        int64_t ptr = 0;
        for (int64_t y = 0; y < py; y += merge) {
            for (int64_t x = 0; x < px; x += merge) {
                for (int64_t dy = 0; dy < merge; ++dy) {
                    for (int64_t dx = 0; dx < merge; ++dx) {
                        pos[0 * n_pos + ptr] = static_cast<int32_t>(y + dy);
                        pos[1 * n_pos + ptr] = static_cast<int32_t>(x + dx);
                        pos[2 * n_pos + ptr] = static_cast<int32_t>(y + dy);
                        pos[3 * n_pos + ptr] = static_cast<int32_t>(x + dx);
                        ++ptr;
                    }
                }
            }
        }
        if (ptr != n_pos)
            throw std::runtime_error(
                "qwen3vl_encoder: slot \"positions\" expected " +
                std::to_string(n_pos) + " entries, actual: " +
                std::to_string(ptr));
        ggml_backend_tensor_set(positions, pos.data(), 0,
                                pos.size() * sizeof(int32_t));
    }

    const ggml_status st = ggml_backend_sched_graph_compute(scheduler_, gf);
    if (st != GGML_STATUS_SUCCESS)
        throw std::runtime_error(
            "qwen3vl_encoder: slot \"graph_compute\" expected "
            "GGML_STATUS_SUCCESS, actual: status " + std::to_string(st));

    // ── Copy out [projection_dim, n_out] ─────────────────────────────────────
    const int64_t n_elem = ggml_nelements(out);
    const int64_t expect = static_cast<int64_t>(projection_dim()) * n_out;
    if (n_elem != expect)
        throw std::runtime_error(
            "qwen3vl_encoder: slot \"image_embeddings.nelements\" expected " +
            std::to_string(expect) + " (projection_dim * mm_tokens), actual: " +
            std::to_string(n_elem));

    std::vector<float> host(static_cast<size_t>(n_elem));
    ggml_backend_tensor_get(out, host.data(), 0, host.size() * sizeof(float));
    return host;
}

}  // namespace qinf::vision
