#pragma once
// gemma4v_encoder.h — the Gemma 4 / EmbeddingGemma-2 ViT (`gemma4v`). The
// fourth concrete IVisionEncoder behind Seam A, after SiglipEncoder (Gemma 3),
// Gemma4UvEncoder (Gemma 4 12B, blockless) and Qwen3VlEncoder.
//
// docs/plan-embed-image.md, phase 3 (decision G1: a new file, not a parameter
// on SigLIP — the block is a different signal flow). Ported from llama.cpp
// ceefd16da `clip_graph_gemma4v::build()` (tools/mtmd/models/gemma4v.cpp), the
// shared `clip_graph::build_vit` it calls, and the matching gemma4v set_input
// in clip.cpp. Checked against models/embeddinggemma-2-mmproj-BF16.gguf:
//
//   - pixels arrive in [0, 1] (mean 0 / std 1) and are mapped to 2x − 1 in
//     the graph; one conv patchify (patch 16), NO patch bias;
//   - positions: two LEARNED lookup tables (v.position_embd [n_embd, 10240, 2],
//     x then y) added to the patches, AND a 2-D NeoX RoPE inside attention
//     (θ = 100, first half of each head's dims by column, second half by row);
//   - 16 Gemma-style blocks: RMS ln1 → q/k/v (no bias) → per-head RMS
//     QK-norm → 2-D RoPE → V-norm (RMS, no weight) → attention (kq_scale 1)
//     → out → RMS attn_post_norm → residual → RMS ln2 → gated FFN (up, gate,
//     act(gate)·up, down) → RMS ffn_post_norm → residual; no pre/post LN;
//   - Gemma4VisionPooler: n_merge × n_merge average pool on the patch grid,
//     then × √n_embd;
//   - Gemma4MultimodalEmbedder: RMS norm (no weight), then
//     mm.input_projection (n_embd → projection_dim = the text model's width).
//
// Attention is flash attention, always (architecture §7, a standing decision):
// not bit-identical to the materialized reference, gated by rel-L2.
//
// The FFN activation comes from the config (VisionFfnAct): the model's
// config.json says tanh GELU, llama.cpp runs GELU-quick for this file. Which
// is the default is a user decision; the encoder implements both.
//
// Ownership / threading as the other encoders: owns its graph context and
// scheduler, shares (does not own) the engine backend, borrows weights from
// VisionModel, single-threaded.

#include <cstdint>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "i_vision_encoder.h"

namespace qinf::vision {

struct Bitmap;
class  VisionModel;

class Gemma4vEncoder : public IVisionEncoder {
public:
    // `model` must have been loaded by VisionLoader with a Gemma4V projector.
    // `text_embed_dim` is the host text model's embedding width and is checked
    // fail-loud against the mmproj's projection_dim (Seam A).
    Gemma4vEncoder(const VisionModel& model,
                   ggml_backend_t     backend,
                   uint32_t           text_embed_dim);
    ~Gemma4vEncoder() override;

    Gemma4vEncoder(const Gemma4vEncoder&) = delete;
    Gemma4vEncoder& operator=(const Gemma4vEncoder&) = delete;

    // [projection_dim, n_tokens] row-major (projection_dim fastest), tokens in
    // pooled-grid order (x fastest). Fail-loud: throws if the bitmap is not a
    // multiple of patch_size·n_merge (48), on a channel/size mismatch, or if
    // compute fails.
    std::vector<float> encode(const Bitmap& bitmap) override;

    // (W / (patch·merge)) · (H / (patch·merge)).
    uint32_t mm_tokens_for(const Bitmap& bitmap) const override;
    void mm_grid_for(const Bitmap& bitmap, uint32_t& nx, uint32_t& ny) const override;
    uint32_t projection_dim() const override;

private:
    void patch_grid(const Bitmap& bitmap, int64_t& px, int64_t& py) const;

    const VisionModel&   model_;
    ggml_backend_t       backend_      = nullptr;  // shared, not owned
    ggml_backend_t       cpu_fallback_ = nullptr;  // owned; only if backend_ isn't CPU
    ggml_backend_sched_t scheduler_    = nullptr;  // owned
    ggml_context*        graph_ctx_    = nullptr;  // owned
    uint32_t             text_embed_dim_ = 0;
};

}  // namespace qinf::vision
