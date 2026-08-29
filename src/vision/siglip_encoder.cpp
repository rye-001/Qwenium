#include "siglip_encoder.h"

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

// Node-count budget for the SigLIP ViT graph. Per layer the build emits on
// the order of ~25 nodes (ln1: norm+mul+add; q/k/v: 3×(mm+add); attn:
// permutes + 2×mul_mat + softmax + cont; out: mm+add; residual; ln2:
// norm+mul+add; ffn: mm+add+gelu+mm+add; residual). 27 layers + patch embed
// + post_ln + slack. 4096 is comfortably above the ~700 actually built and
// matches the headroom style of FP_GRAPH_SIZE on the text side.
constexpr size_t VIT_GRAPH_SIZE = 4096;

// Fetch a required weight tensor by mmproj GGUF name. Fail-loud per the
// CLAUDE.md contract (slot → expected → actual) rather than returning a
// null that would segfault deep inside graph build.
ggml_tensor* require(const VisionModel& model, const std::string& name) {
    const auto& tensors = model.tensors();
    auto it = tensors.find(name);
    if (it == tensors.end() || it->second == nullptr) {
        throw std::runtime_error(
            "vision_encoder: slot \"tensor:" + name +
            "\" expected present-in-mmproj, got absent. The mmproj does not "
            "carry the SigLIP weight this graph requires "
            "(see docs/plan-gemma-vision-impl.md P2.3).");
    }
    return it->second;
}

}  // namespace

SiglipEncoder::SiglipEncoder(const VisionModel& model,
                             ggml_backend_t      backend,
                             uint32_t            text_embed_dim)
    : model_(model),
      backend_(backend),
      text_embed_dim_(text_embed_dim) {
    if (backend_ == nullptr) {
        throw std::runtime_error(
            "vision_encoder: slot \"backend\" expected non-null "
            "ggml_backend_t (shared with the text engine), got nullptr.");
    }

    // Loose-coupling sanity check (P2.1+): the mmproj's declared
    // projection_dim must equal the caller's stated text_embed_dim. A
    // mismatch means the mmproj does not match the host text model —
    // refuse loudly rather than encode and produce embeddings of the
    // wrong dim that would silently corrupt the text-model residual
    // stream.
    if (model_.config().projection_dim != 0 &&
        model_.config().projection_dim != text_embed_dim_) {
        throw std::runtime_error(
            std::string("vision_encoder: slot \"projection_dim\" expected ") +
            std::to_string(text_embed_dim_) +
            " (host text model embedding_length), got " +
            std::to_string(model_.config().projection_dim) +
            " (mmproj clip.vision.projection_dim). The mmproj does not "
            "match this text model.");
    }

    // Own a scheduler over the shared backend. One device, two schedulers
    // (text + vision) — see the C3 ownership note in the header. measure=false:
    // real buffers are allocated at alloc_graph time.
    //
    // ggml_backend_sched_new requires the LAST backend to be a CPU backend
    // (it is the scheduler's fallback for ops the primary device can't run).
    // When the shared backend is already CPU (the unit-test path), one backend
    // suffices. When it is a device backend (Metal — the real engine path), we
    // add an owned CPU fallback as the last backend, mirroring how the text
    // engine builds its [device, cpu] scheduler. Without it, sched_new aborts.
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
                "vision_encoder: slot \"cpu_fallback\" expected non-null from "
                "ggml_backend_cpu_init (scheduler requires a CPU fallback "
                "backend), got nullptr.");
        backends[n_backends++] = cpu_fallback_;
    }

    scheduler_ = ggml_backend_sched_new(backends, /*bufts=*/nullptr,
                                        n_backends,
                                        /*graph_size=*/VIT_GRAPH_SIZE,
                                        /*parallel=*/false,
                                        /*op_offload=*/false);
    if (scheduler_ == nullptr) {
        throw std::runtime_error(
            "vision_encoder: slot \"scheduler\" expected non-null from "
            "ggml_backend_sched_new, got nullptr.");
    }
}

SiglipEncoder::~SiglipEncoder() {
    if (scheduler_ != nullptr) {
        ggml_backend_sched_free(scheduler_);
        scheduler_ = nullptr;
    }
    if (graph_ctx_ != nullptr) {
        ggml_free(graph_ctx_);
        graph_ctx_ = nullptr;
    }
    if (cpu_fallback_ != nullptr) {
        ggml_backend_free(cpu_fallback_);  // owned — only when backend_ isn't CPU
        cpu_fallback_ = nullptr;
    }
    // backend_ is not owned — do not free.
}

std::vector<float> SiglipEncoder::encode(const Bitmap& bitmap) {
    const SigLIPConfig& cfg = model_.config();

    // ── 1. Validate the bitmap against the config (fail-loud) ─────────────────
    if (bitmap.channels != static_cast<int>(cfg.num_channels)) {
        throw std::runtime_error(
            "vision_encoder: slot \"bitmap.channels\" expected " +
            std::to_string(cfg.num_channels) + ", got " +
            std::to_string(bitmap.channels) + ".");
    }
    if (bitmap.height != static_cast<int>(cfg.image_size)) {
        throw std::runtime_error(
            "vision_encoder: slot \"bitmap.height\" expected " +
            std::to_string(cfg.image_size) + ", got " +
            std::to_string(bitmap.height) + ".");
    }
    if (bitmap.width != static_cast<int>(cfg.image_size)) {
        throw std::runtime_error(
            "vision_encoder: slot \"bitmap.width\" expected " +
            std::to_string(cfg.image_size) + ", got " +
            std::to_string(bitmap.width) + ".");
    }
    const size_t expect_pixels =
        static_cast<size_t>(cfg.num_channels) * cfg.image_size * cfg.image_size;
    if (bitmap.pixels.size() != expect_pixels) {
        throw std::runtime_error(
            "vision_encoder: slot \"bitmap.pixels.size\" expected " +
            std::to_string(expect_pixels) + " (C*H*W), got " +
            std::to_string(bitmap.pixels.size()) + ".");
    }

    // Derived dimensions. SigLIP-So400M @ Gemma 3: patch 14, 64×64 grid,
    // 4096 patch tokens, hidden 1152, 16 heads → d_head 72.
    const int64_t n_embd          = static_cast<int64_t>(cfg.hidden_size);
    const int64_t n_head          = static_cast<int64_t>(cfg.num_attn_heads);
    const int64_t d_head          = n_embd / n_head;
    const int64_t patch           = static_cast<int64_t>(cfg.patch_size);
    const int64_t img_sz          = static_cast<int64_t>(cfg.image_size);
    const int64_t patches_per_side = img_sz / patch;          // 64
    const int64_t n_pos           = patches_per_side * patches_per_side; // 4096
    const float   eps             = cfg.layer_norm_eps;
    const float   kq_scale        = 1.0f / std::sqrt(static_cast<float>(d_head));
    const int     n_layer         = static_cast<int>(cfg.num_layers);

    if (n_head == 0 || n_embd % n_head != 0) {
        throw std::runtime_error(
            "vision_encoder: slot \"hidden_size/num_attn_heads\" expected "
            "hidden_size divisible by head count, got " +
            std::to_string(n_embd) + " / " + std::to_string(n_head) + ".");
    }

    // ── 2. (Re)build a fresh metadata-only graph context ──────────────────────
    if (graph_ctx_ != nullptr) {
        ggml_free(graph_ctx_);
        graph_ctx_ = nullptr;
    }
    const size_t mem_size =
        ggml_tensor_overhead() * VIT_GRAPH_SIZE +
        ggml_graph_overhead_custom(VIT_GRAPH_SIZE, false);
    ggml_init_params ip{};
    ip.mem_size   = mem_size;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;   // data lives in scheduler-managed backend buffers
    graph_ctx_ = ggml_init(ip);
    if (graph_ctx_ == nullptr) {
        throw std::runtime_error(
            "vision_encoder: slot \"graph_ctx\" expected non-null from "
            "ggml_init, got nullptr.");
    }
    ggml_context* ctx = graph_ctx_;
    ggml_cgraph*  gf  = ggml_new_graph_custom(ctx, VIT_GRAPH_SIZE, false);

    // ── 3. Patch embedding: conv2d → [n_embd, n_patches] + bias ───────────────
    // Raw input tensor: ggml conv2d expects [W, H, C] (ne0 fastest = width).
    // Bitmap is channel-planar [C, H, W] contiguous, which is exactly the
    // memory order ggml reads for an [W, H, C] tensor.
    ggml_tensor* inp_raw =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, img_sz, img_sz, cfg.num_channels);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    ggml_tensor* patch_w = require(model_, "v.patch_embd.weight");
    ggml_tensor* patch_b = require(model_, "v.patch_embd.bias");

    // Metal backend constraint (ggml-metal supports_op): ggml_conv_2d lowers
    // to IM2COL + MUL_MAT, and IM2COL builds its output tensor with the
    // kernel's dtype. The mmproj patch weight is BF16, but Metal's IM2COL
    // kernel accepts only F16/F32 output — a BF16 im2col node is scheduled to
    // CPU, splitting the graph at the very first op of every image encode
    // (exactly the silent-CPU-fallback trap CLAUDE.md warns about). Upcast the
    // weight to F32 so im2col's output is Metal-native. BF16→F32 is an exact
    // widening, so this does not move the encoder off the llama.cpp reference
    // (the conv accumulates in F32 regardless). The weight is tiny
    // (patch²·C·n_embd ≈ 675K elements), so the per-encode cast is negligible.
    if (patch_w->type != GGML_TYPE_F32 && patch_w->type != GGML_TYPE_F16) {
        patch_w = ggml_cast(ctx, patch_w, GGML_TYPE_F32);
    }

    // conv2d: stride = patch, no padding, dilation 1 → [64, 64, n_embd]
    ggml_tensor* cur = ggml_conv_2d(ctx, patch_w, inp_raw,
                                    patch, patch, 0, 0, 1, 1);
    // → [n_patches, n_embd], then transpose to [n_embd, n_patches]
    cur = ggml_reshape_2d(ctx, cur, n_pos, n_embd);
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
    cur = ggml_add(ctx, cur, patch_b);

    // Learned position embedding (no CLS token in SigLIP): [n_embd, n_pos].
    ggml_tensor* pos_embd = require(model_, "v.position_embd.weight");
    cur = ggml_add(ctx, cur, pos_embd);

    ggml_tensor* inpL = cur;  // residual stream

    // ── 4. Transformer layers (pre-norm ViT; SigLIP has no pre_ln) ────────────
    for (int il = 0; il < n_layer; ++il) {
        const std::string p = "v.blk." + std::to_string(il) + ".";

        // layernorm 1
        ggml_tensor* h = ggml_norm(ctx, inpL, eps);
        h = ggml_mul(ctx, h, require(model_, p + "ln1.weight"));
        h = ggml_add(ctx, h, require(model_, p + "ln1.bias"));

        // self-attention (separate q/k/v, each with bias)
        ggml_tensor* Q = ggml_add(ctx,
            ggml_mul_mat(ctx, require(model_, p + "attn_q.weight"), h),
            require(model_, p + "attn_q.bias"));
        ggml_tensor* K = ggml_add(ctx,
            ggml_mul_mat(ctx, require(model_, p + "attn_k.weight"), h),
            require(model_, p + "attn_k.bias"));
        ggml_tensor* V = ggml_add(ctx,
            ggml_mul_mat(ctx, require(model_, p + "attn_v.weight"), h),
            require(model_, p + "attn_v.bias"));

        Q = ggml_reshape_3d(ctx, Q, d_head, n_head, n_pos);
        K = ggml_reshape_3d(ctx, K, d_head, n_head, n_pos);
        V = ggml_reshape_3d(ctx, V, d_head, n_head, n_pos);

        // [d_head, n_pos, n_head]
        ggml_tensor* q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        ggml_tensor* k = ggml_permute(ctx, K, 0, 2, 1, 3);
        // kq = k^T q → [n_pos(k), n_pos(q), n_head]; scaled softmax, no mask
        ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
        kq = ggml_soft_max_ext(ctx, kq, /*mask=*/nullptr, kq_scale, 0.0f);
        // v: [n_pos, d_head, n_head] so mul_mat(v, kq) → [d_head, n_pos, n_head]
        ggml_tensor* v = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));
        ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);
        // → [d_head, n_head, n_pos] → [n_embd, n_pos]
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx, kqv, n_embd, n_pos);

        // output projection (+bias)
        cur = ggml_add(ctx,
            ggml_mul_mat(ctx, require(model_, p + "attn_out.weight"), cur),
            require(model_, p + "attn_out.bias"));

        // residual 1
        inpL = ggml_add(ctx, cur, inpL);

        // layernorm 2 (pre-ffn)
        ggml_tensor* f = ggml_norm(ctx, inpL, eps);
        f = ggml_mul(ctx, f, require(model_, p + "ln2.weight"));
        f = ggml_add(ctx, f, require(model_, p + "ln2.bias"));

        // ffn: up (+bias) → GELU (tanh approx; clip.use_gelu=true) → down (+bias)
        f = ggml_add(ctx,
            ggml_mul_mat(ctx, require(model_, p + "ffn_up.weight"), f),
            require(model_, p + "ffn_up.bias"));
        f = ggml_gelu(ctx, f);
        f = ggml_add(ctx,
            ggml_mul_mat(ctx, require(model_, p + "ffn_down.weight"), f),
            require(model_, p + "ffn_down.bias"));

        // residual 2
        inpL = ggml_add(ctx, inpL, f);
    }

    // ── 5. Final post-layernorm → [n_embd, n_patches] ────────────────────────
    ggml_tensor* cur_out = ggml_norm(ctx, inpL, eps);
    cur_out = ggml_mul(ctx, cur_out, require(model_, "v.post_ln.weight"));
    cur_out = ggml_add(ctx, cur_out, require(model_, "v.post_ln.bias"));

    // ── 6. Gemma 3 projection head (P2.4) ────────────────────────────────────
    // 4×4 average pool on the 64×64 patch grid → 256 tokens, then RMS
    // soft-emb-norm and the linear projection to the text-embedding dim.
    // Mirrors clip_graph_siglip::build() PROJECTOR_TYPE_GEMMA3 branch.
    const int64_t kernel          = static_cast<int64_t>(cfg.pool_factor);   // 4
    const int64_t tokens_per_side = patches_per_side / kernel;               // 16
    if (tokens_per_side * kernel != patches_per_side) {
        throw std::runtime_error(
            "vision_encoder: slot \"pool_factor\" expected patches_per_side "
            "divisible by pool_factor, got " +
            std::to_string(patches_per_side) + " / " +
            std::to_string(kernel) + ".");
    }
    const int64_t n_tokens = tokens_per_side * tokens_per_side;              // 256

    // [n_embd, n_pos] → [n_pos, n_embd] → [64, 64, n_embd, 1]
    cur = ggml_transpose(ctx, cur_out);
    cur = ggml_cont_4d(ctx, cur, patches_per_side, patches_per_side, n_embd, 1);

    // average pool 4×4 stride 4 → [16, 16, n_embd, 1]
    cur = ggml_pool_2d(ctx, cur, GGML_OP_POOL_AVG,
                       kernel, kernel, kernel, kernel, 0, 0);

    // → [256, n_embd, 1] → [n_embd, 256]
    cur = ggml_reshape_3d(ctx, cur, n_tokens, n_embd, 1);
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));

    // RMS soft-emb-norm (NB: RMS here, not the LayerNorm used inside the ViT)
    cur = ggml_rms_norm(ctx, cur, eps);
    cur = ggml_mul(ctx, cur, require(model_, "mm.soft_emb_norm.weight"));

    // linear projection: mm.input_projection.weight is [proj_dim, n_embd];
    // transpose to [n_embd, proj_dim] so mul_mat maps [n_embd,256] → [proj_dim,256].
    ggml_tensor* proj_w = require(model_, "mm.input_projection.weight");
    ggml_tensor* out = ggml_mul_mat(
        ctx, ggml_cont(ctx, ggml_transpose(ctx, proj_w)), cur);
    ggml_set_name(out, "image_embeddings");
    ggml_set_output(out);

    ggml_build_forward_expand(gf, out);

    // ── 7. Allocate, upload input, compute on the shared backend ──────────────
    ggml_backend_sched_reset(scheduler_);
    if (!ggml_backend_sched_alloc_graph(scheduler_, gf)) {
        throw std::runtime_error(
            "vision_encoder: slot \"alloc_graph\" expected success from "
            "ggml_backend_sched_alloc_graph, got failure.");
    }
    ggml_backend_tensor_set(inp_raw, bitmap.pixels.data(), 0,
                            bitmap.pixels.size() * sizeof(float));

    const ggml_status st = ggml_backend_sched_graph_compute(scheduler_, gf);
    if (st != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(
            "vision_encoder: slot \"graph_compute\" expected "
            "GGML_STATUS_SUCCESS, got status " + std::to_string(st) + ".");
    }

    // ── 8. Copy the projected image embeddings out to host ────────────────────
    // Shape: [projection_dim=2560, mm_tokens_per_image=256], projection_dim-
    // fastest (row r = the embedding of image-soft-token r at the text-embed
    // dim). This is the final encoder output that Phase 3's ImageEmbeddingsInput
    // uploads into the gemma3 prefill graph. Matches the llama.cpp reference
    // fixture layout (tests/fixtures/vision/siglip_gray_896.bin: [256, 2560]
    // row-major, n_embd-fastest).
    const int64_t n_out = ggml_nelements(out);
    std::vector<float> host(static_cast<size_t>(n_out));
    ggml_backend_tensor_get(out, host.data(), 0,
                            host.size() * sizeof(float));
    return host;
}

uint32_t SiglipEncoder::mm_tokens_per_image() const {
    return model_.config().mm_tokens_per_image;
}

uint32_t SiglipEncoder::mm_tokens_for(const Bitmap& /*bitmap*/) const {
    // SigLIP's token count is fixed (896 tile → 4×4 pool → 256), independent of
    // the bitmap. The IVisionEncoder seam takes a bitmap because Gemma 4's count
    // is per-image; here it is the constant.
    return model_.config().mm_tokens_per_image;
}

void SiglipEncoder::mm_grid_for(const Bitmap& /*bitmap*/,
                                uint32_t& nx, uint32_t& ny) const {
    // Constant, like mm_tokens_for: the 896 tile always pools to 16×16.
    const auto& cfg = model_.config();
    const uint32_t per_side =
        (cfg.patch_size && cfg.pool_factor)
            ? (cfg.image_size / cfg.patch_size) / cfg.pool_factor : 0;
    nx = per_side;
    ny = per_side;
}

uint32_t SiglipEncoder::projection_dim() const {
    return model_.config().projection_dim != 0
        ? model_.config().projection_dim
        : text_embed_dim_;
}

}  // namespace qinf::vision
