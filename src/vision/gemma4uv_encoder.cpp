#include "gemma4uv_encoder.h"

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

// Blockless graph: ~15 nodes. 256 is generous headroom (matches the style of
// SiglipEncoder's VIT_GRAPH_SIZE without its 27-layer cost).
constexpr size_t UV_GRAPH_SIZE = 256;

// pytorch-default LayerNorm eps, hardcoded in gemma4uv.cpp (NOT a mmproj kv).
constexpr float UV_LN_EPS = 1e-5f;

ggml_tensor* require(const VisionModel& model, const std::string& name) {
    const auto& tensors = model.tensors();
    auto it = tensors.find(name);
    if (it == tensors.end() || it->second == nullptr) {
        throw std::runtime_error(
            "gemma4uv_encoder: slot \"tensor:" + name +
            "\" expected present-in-mmproj, got absent. The mmproj does not "
            "carry the gemma4uv weight this graph requires "
            "(see docs/plan-gemma4-vision-impl.md §4).");
    }
    return it->second;
}

// build_norm(NORM_TYPE_NORMAL) from clip.cpp: LayerNorm over ne0, weight+bias.
ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* w, ggml_tensor* b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

}  // namespace

Gemma4UvEncoder::Gemma4UvEncoder(const VisionModel& model,
                                 ggml_backend_t     backend,
                                 uint32_t           text_embed_dim)
    : model_(model),
      backend_(backend),
      text_embed_dim_(text_embed_dim) {
    if (backend_ == nullptr) {
        throw std::runtime_error(
            "gemma4uv_encoder: slot \"backend\" expected non-null "
            "ggml_backend_t (shared with the text engine), got nullptr.");
    }

    const SigLIPConfig& cfg = model_.config();
    if (cfg.projector_type != VisionProjectorType::Gemma4Uv) {
        throw std::runtime_error(
            "gemma4uv_encoder: slot \"projector_type\" expected Gemma4Uv, got "
            "a mmproj loaded as a different projector type. Construct a "
            "SiglipEncoder for the gemma3 projector instead.");
    }

    // Loose-coupling sanity check: the mmproj's projection_dim must equal the
    // caller's text_embed_dim. For gemma4uv the projection output dim is the
    // weight's ne[1] (== projection_dim == embedding_length here).
    if (cfg.projection_dim != 0 && cfg.projection_dim != text_embed_dim_) {
        throw std::runtime_error(
            std::string("gemma4uv_encoder: slot \"projection_dim\" expected ") +
            std::to_string(text_embed_dim_) +
            " (host text model embedding_length), got " +
            std::to_string(cfg.projection_dim) +
            " (mmproj clip.vision.projection_dim). The mmproj does not match "
            "this text model.");
    }

    eff_patch_ = cfg.patch_size * cfg.n_merge;  // 16 · 3 = 48
    if (eff_patch_ == 0) {
        throw std::runtime_error(
            "gemma4uv_encoder: slot \"eff_patch\" expected non-zero "
            "(patch_size · n_merge), got 0.");
    }

    // Scheduler over the shared backend, with an owned CPU fallback when the
    // primary is a device backend (ggml_backend_sched_new requires the last
    // backend to be CPU). Identical contract to SiglipEncoder.
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
                "gemma4uv_encoder: slot \"cpu_fallback\" expected non-null from "
                "ggml_backend_cpu_init, got nullptr.");
        backends[n_backends++] = cpu_fallback_;
    }

    scheduler_ = ggml_backend_sched_new(backends, /*bufts=*/nullptr, n_backends,
                                        /*graph_size=*/UV_GRAPH_SIZE,
                                        /*parallel=*/false, /*op_offload=*/false);
    if (scheduler_ == nullptr) {
        throw std::runtime_error(
            "gemma4uv_encoder: slot \"scheduler\" expected non-null from "
            "ggml_backend_sched_new, got nullptr.");
    }
}

Gemma4UvEncoder::~Gemma4UvEncoder() {
    if (scheduler_ != nullptr) {
        ggml_backend_sched_free(scheduler_);
        scheduler_ = nullptr;
    }
    if (graph_ctx_ != nullptr) {
        ggml_free(graph_ctx_);
        graph_ctx_ = nullptr;
    }
    if (cpu_fallback_ != nullptr) {
        ggml_backend_free(cpu_fallback_);
        cpu_fallback_ = nullptr;
    }
    // backend_ is not owned — do not free.
}

uint32_t Gemma4UvEncoder::mm_tokens_for(const Bitmap& bitmap) const {
    // (W/P)·(H/P): the dynamic-size preprocessor guarantees W,H are multiples
    // of the effective patch P. Pure — no encode.
    const uint32_t P = eff_patch_;
    if (P == 0 || bitmap.width <= 0 || bitmap.height <= 0)
        return 0;
    return static_cast<uint32_t>(bitmap.width / static_cast<int>(P)) *
           static_cast<uint32_t>(bitmap.height / static_cast<int>(P));
}

void Gemma4UvEncoder::mm_grid_for(const Bitmap& bitmap,
                                  uint32_t& nx, uint32_t& ny) const {
    const uint32_t P = eff_patch_;
    if (P == 0 || bitmap.width <= 0 || bitmap.height <= 0) { nx = 0; ny = 0; return; }
    nx = static_cast<uint32_t>(bitmap.width  / static_cast<int>(P));
    ny = static_cast<uint32_t>(bitmap.height / static_cast<int>(P));
}

uint32_t Gemma4UvEncoder::projection_dim() const {
    const uint32_t pd = model_.config().projection_dim;
    return pd != 0 ? pd : text_embed_dim_;
}

std::vector<float> Gemma4UvEncoder::encode(const Bitmap& bitmap) {
    const SigLIPConfig& cfg = model_.config();

    const int64_t c       = static_cast<int64_t>(cfg.num_channels);   // 3
    const int64_t P       = static_cast<int64_t>(eff_patch_);          // 48
    const int64_t n_embd  = static_cast<int64_t>(cfg.hidden_size);     // 3840
    const float   rms_eps = cfg.layer_norm_eps;                        // 1e-6 (gemma4uv.cpp hparams.eps)

    // ── 1. Validate the preprocessed bitmap (fail-loud) ──────────────────────
    if (bitmap.channels != static_cast<int>(cfg.num_channels))
        throw std::runtime_error(
            "gemma4uv_encoder: slot \"bitmap.channels\" expected " +
            std::to_string(cfg.num_channels) + ", got " +
            std::to_string(bitmap.channels) + ".");
    if (bitmap.width <= 0 || bitmap.height <= 0)
        throw std::runtime_error(
            "gemma4uv_encoder: slot \"bitmap.{width,height}\" expected positive, "
            "got " + std::to_string(bitmap.width) + "x" +
            std::to_string(bitmap.height) + ".");
    if (bitmap.width % P != 0 || bitmap.height % P != 0)
        throw std::runtime_error(
            "gemma4uv_encoder: slot \"bitmap.{width,height}\" expected multiples "
            "of the effective patch " + std::to_string(P) + ", got " +
            std::to_string(bitmap.width) + "x" + std::to_string(bitmap.height) +
            ". The dynamic-size preprocessor must round to multiples of P.");
    const size_t expect_pixels =
        static_cast<size_t>(c) * bitmap.height * bitmap.width;
    if (bitmap.pixels.size() != expect_pixels)
        throw std::runtime_error(
            "gemma4uv_encoder: slot \"bitmap.pixels.size\" expected " +
            std::to_string(expect_pixels) + " (C·H·W), got " +
            std::to_string(bitmap.pixels.size()) + ".");

    const int64_t W        = bitmap.width;
    const int64_t H        = bitmap.height;
    const int64_t n_cols   = W / P;
    const int64_t n_rows   = H / P;
    const int64_t n_patch  = n_cols * n_rows;

    // ── 2. Fresh metadata-only graph context ─────────────────────────────────
    if (graph_ctx_ != nullptr) { ggml_free(graph_ctx_); graph_ctx_ = nullptr; }
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * UV_GRAPH_SIZE +
                    ggml_graph_overhead_custom(UV_GRAPH_SIZE, false);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    graph_ctx_ = ggml_init(ip);
    if (graph_ctx_ == nullptr)
        throw std::runtime_error(
            "gemma4uv_encoder: slot \"graph_ctx\" expected non-null from "
            "ggml_init, got nullptr.");
    ggml_context* ctx = graph_ctx_;
    ggml_cgraph*  gf  = ggml_new_graph_custom(ctx, UV_GRAPH_SIZE, false);

    // Raw input: ggml im2col reads [W, H, C] (ne0 fastest = width). Bitmap is
    // channel-planar [C, H, W] contiguous = exactly that memory order.
    ggml_tensor* inp_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, W, H, c);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    // ── 3. Patch embed via im2col (norm sits between im2col and the matmul,
    //       which conv2d can't express — hence im2col, not ggml_conv_2d). ─────
    // kernel tensor supplies only the [P,P,c] shape + dtype to im2col.
    ggml_tensor* kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, P, P, c);
    ggml_tensor* inp = ggml_im2col(ctx, kernel, inp_raw,
                                   /*s0=*/static_cast<int>(P), /*s1=*/static_cast<int>(P),
                                   /*p0=*/0, /*p1=*/0, /*d0=*/1, /*d1=*/1,
                                   /*is_2D=*/true, /*dst_type=*/GGML_TYPE_F32);
    // inp: [P·P·c, n_cols, n_rows] → [P·P·c, n_patch]
    inp = ggml_reshape_2d(ctx, inp, inp->ne[0], n_patch);
    inp = layer_norm(ctx, inp, require(model_, "v.patch_norm.1.weight"),
                     require(model_, "v.patch_norm.1.bias"), UV_LN_EPS);

    // patch_embd.weight [P·P·c, n_embd] → mul_mat → [n_embd, n_patch], +bias
    inp = ggml_mul_mat(ctx, require(model_, "v.patch_embd.weight"), inp);
    inp = ggml_add(ctx, inp, require(model_, "v.patch_embd.bias"));
    inp = layer_norm(ctx, inp, require(model_, "v.patch_norm.2.weight"),
                     require(model_, "v.patch_norm.2.bias"), UV_LN_EPS);

    // ── 4. 2D learned positional embeddings (x/y lookup tables) ───────────────
    ggml_tensor* pos_x = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_patch);
    ggml_set_name(pos_x, "pos_x");
    ggml_set_input(pos_x);
    ggml_tensor* pos_y = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_patch);
    ggml_set_name(pos_y, "pos_y");
    ggml_set_input(pos_y);

    ggml_tensor* position_embd = require(model_, "v.position_embd.weight");
    const int64_t pos_size = position_embd->ne[1];
    if (n_cols > pos_size || n_rows > pos_size)
        throw std::runtime_error(
            "gemma4uv_encoder: slot \"position table size\" expected pos_size (" +
            std::to_string(pos_size) + ") >= n_cols/n_rows (" +
            std::to_string(n_cols) + "/" + std::to_string(n_rows) +
            "); the image exceeds the learned positional grid.");
    const size_t nb1 = ggml_row_size(position_embd->type, n_embd);
    // tbl_x = the ne2=0 plane (rows [0, pos_size)); tbl_y = the ne2=1 plane.
    ggml_tensor* tbl_x = ggml_view_2d(ctx, position_embd, n_embd, pos_size, nb1, 0);
    ggml_tensor* tbl_y = ggml_view_2d(ctx, position_embd, n_embd, pos_size, nb1,
                                      static_cast<size_t>(pos_size) * nb1);
    ggml_tensor* emb_x = ggml_get_rows(ctx, tbl_x, pos_x);
    ggml_tensor* emb_y = ggml_get_rows(ctx, tbl_y, pos_y);
    inp = ggml_add(ctx, inp, emb_x);
    inp = ggml_add(ctx, inp, emb_y);
    inp = layer_norm(ctx, inp, require(model_, "v.patch_norm.3.weight"),
                     require(model_, "v.patch_norm.3.bias"), UV_LN_EPS);

    // ── 5. Gemma4UnifiedMultimodalEmbedder: RMSNorm (no weight) → projection ──
    ggml_tensor* cur = ggml_rms_norm(ctx, inp, rms_eps);
    // mm.input_projection.weight [n_embd, proj_dim] → DIRECT mul_mat (no transpose)
    // → [proj_dim, n_patch].
    ggml_tensor* out = ggml_mul_mat(ctx, require(model_, "mm.input_projection.weight"), cur);
    ggml_set_name(out, "image_embeddings");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    // ── 6. Allocate, upload inputs, compute ──────────────────────────────────
    ggml_backend_sched_reset(scheduler_);
    if (!ggml_backend_sched_alloc_graph(scheduler_, gf))
        throw std::runtime_error(
            "gemma4uv_encoder: slot \"alloc_graph\" expected success from "
            "ggml_backend_sched_alloc_graph, got failure.");

    ggml_backend_tensor_set(inp_raw, bitmap.pixels.data(), 0,
                            bitmap.pixels.size() * sizeof(float));
    // pos_x[i] = i % n_cols, pos_y[i] = i / n_cols (clip.cpp GEMMA4UV branch).
    std::vector<int32_t> px(static_cast<size_t>(n_patch));
    std::vector<int32_t> py(static_cast<size_t>(n_patch));
    for (int64_t i = 0; i < n_patch; ++i) {
        px[static_cast<size_t>(i)] = static_cast<int32_t>(i % n_cols);
        py[static_cast<size_t>(i)] = static_cast<int32_t>(i / n_cols);
    }
    ggml_backend_tensor_set(pos_x, px.data(), 0, px.size() * sizeof(int32_t));
    ggml_backend_tensor_set(pos_y, py.data(), 0, py.size() * sizeof(int32_t));

    const ggml_status st = ggml_backend_sched_graph_compute(scheduler_, gf);
    if (st != GGML_STATUS_SUCCESS)
        throw std::runtime_error(
            "gemma4uv_encoder: slot \"graph_compute\" expected "
            "GGML_STATUS_SUCCESS, got status " + std::to_string(st) + ".");

    // ── 7. Copy out [projection_dim, n_patch], projection_dim fastest ─────────
    const int64_t n_out = ggml_nelements(out);
    std::vector<float> host(static_cast<size_t>(n_out));
    ggml_backend_tensor_get(out, host.data(), 0, host.size() * sizeof(float));
    return host;
}

}  // namespace qinf::vision
