#pragma once
// qwen3vl_encoder.h — Qwen 3.5-family ViT + 2×2 spatial merger
// (`clip.projector_type = "qwen3vl_merger"`). The third concrete
// IVisionEncoder behind Seam A (see i_vision_encoder.h), after SiglipEncoder
// (Gemma 3) and Gemma4UvEncoder (Gemma 4).
//
// P3 of docs/plan-qwen35-vision-impl.md. Ported against llama.cpp's
// `clip_graph_qwen3vl::build()` (tools/mtmd/models/qwen3vl.cpp) and the
// matching `set_input` position construction in clip.cpp, both of which are
// vendored under build-*/_deps/ggml-src — this is a port from the reference,
// not a reconstruction from the weight inventory.
//
// Dimensionally this tower equals the Gemma 3 SigLIP one we already run
// (1152 / 27 layers / 16 heads / 4304 FFN). It is NOT SigLIP. The differences
// that matter, each verified against models/Qwen3.6-mtp-mmproj-BF16.gguf:
//
//   - TWO patch-embed convs (`v.patch_embd.weight` and `.weight.1`), summed.
//     They are the temporal-merge pair; a still image feeds both.
//   - A 2×2 SPATIAL MERGE applied to the patch grid BEFORE the transformer,
//     so the merger's closing reshape folds contiguous 2×2 blocks.
//   - Learned position embeddings on a fixed 48×48 grid, BILINEARLY RESIZED
//     (align-corners) to this image's patch grid. `image_size` is the
//     pos-embed grid, not a required input size — resolution is dynamic.
//   - FUSED QKV (one [n_embd, 3·n_embd] projection, viewed into Q/K/V).
//   - M-RoPE INSIDE the ViT: ggml_rope_multi with GGML_ROPE_TYPE_VISION and
//     sections {d_head/4}×4, over a 4-component position tensor. Unrelated to
//     the text-side M-RoPE from P2 beyond sharing the kernel.
//   - LayerNorm WITH biases (not RMSNorm), gate-less GELU MLP.
//   - post_ln only — this tower has NO pre_ln.
//   - Projection is the 2×2 merger: [n_embd·4 → n_embd·4] → GELU →
//     [n_embd·4 → projection_dim], i.e. `mm.0` then `mm.2`.
//
// DeepStack is deliberately absent: our mmproj declares
// `is_deepstack_layers` all-false and ships no `deepstack_*` tensors, and
// VisionLoader refuses a file that enables it (§3.4). If a DeepStack variant
// ever ships, it is a new feature here, not a silent degradation.
//
// Ownership / threading identical to SiglipEncoder: owns its ggml context,
// graph and scheduler; shares (does not own) the engine backend; borrows
// weights from VisionModel; single-threaded.

#include <cstdint>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "i_vision_encoder.h"

namespace qinf::vision {

struct Bitmap;
class  VisionModel;

class Qwen3VlEncoder : public IVisionEncoder {
public:
    // `model` must already have been loaded by VisionLoader with a
    // Qwen3VlMerger projector. `backend` is shared, not owned.
    // `text_embed_dim` is the host text model's embedding_length and is
    // fail-loud checked against the mmproj's projection_dim (Seam A).
    Qwen3VlEncoder(const VisionModel& model,
                   ggml_backend_t     backend,
                   uint32_t           text_embed_dim);
    ~Qwen3VlEncoder() override;

    Qwen3VlEncoder(const Qwen3VlEncoder&) = delete;
    Qwen3VlEncoder& operator=(const Qwen3VlEncoder&) = delete;

    // Encode one preprocessed bitmap. Returns [projection_dim, n_tokens]
    // row-major (projection_dim fastest), n_tokens = mm_tokens_for(bitmap).
    //
    // Fail-loud: throws if the bitmap's dimensions are not a multiple of
    // patch_size·n_merge (32), if the channel count or buffer size disagree
    // with the config, or if graph compute fails.
    std::vector<float> encode(const Bitmap& bitmap) override;

    // (W / (patch·merge)) · (H / (patch·merge)) — per image, not a constant.
    // Dynamic resolution means the count is decided by preprocessing.
    uint32_t mm_tokens_for(const Bitmap& bitmap) const override;

    // The MERGED grid: (W/(patch·merge)) × (H/(patch·merge)). This is the
    // shape M-RoPE indexes — llama.cpp's clip_n_output_tokens_{x,y} for
    // qwen3vl compute exactly (nx/patch)/2 and (ny/patch)/2.
    void mm_grid_for(const Bitmap& bitmap,
                     uint32_t& nx, uint32_t& ny) const override;

    uint32_t projection_dim() const override;

private:
    // Shared by encode() and mm_tokens_for(): validates the bitmap against the
    // config and returns the patch-grid dimensions. Fail-loud.
    void patch_grid(const Bitmap& bitmap, int64_t& px, int64_t& py) const;

    const VisionModel&   model_;
    ggml_backend_t       backend_      = nullptr;  // shared, not owned
    ggml_backend_t       cpu_fallback_ = nullptr;  // owned; only if backend_ isn't CPU
    ggml_backend_sched_t scheduler_    = nullptr;  // owned
    ggml_context*        graph_ctx_    = nullptr;  // owned

    uint32_t             text_embed_dim_ = 0;
};

}  // namespace qinf::vision
