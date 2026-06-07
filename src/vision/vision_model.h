#pragma once
// vision_model.h — weights + config for a Gemma 3 mmproj (SigLIP encoder
// plus 4×4 pool + linear projection + soft-emb-norm).
//
// Mirrors the text-side Model / ModelMetadata split:
//   - SigLIPConfig is the metadata (dimensions, layer count, etc.).
//   - VisionModel owns the loaded weight tensors (ggml_context + buffer).
// Mirrors text-side Model RAII: destructor frees the ggml_context and the
// backing buffer.
//
// VisionModel is populated by VisionLoader (see vision_loader.h) and
// consumed by VisionEncoder. It does NOT depend on the text-side Model
// or ForwardPassBase — the architectural separation (C3 deliverable) is
// directory-enforced.
//
// Single-recipe scope in v1: Gemma 3 SigLIP-So400M (image_size=896,
// patch=14, 27 vision layers, hidden=1152, mm_tokens_per_image=256 after
// 4×4 average pool on the 64×64 patch grid). Gemma 3n (MobileNetV5) and
// Gemma 4V are deferred under separate decisions
// (docs/plan-gemma-vision-impl.md → Open Questions). When a second
// vision recipe lands, the working hypothesis ("pre-decoder pipeline,
// not 6th layer-module type") is re-evaluated against it.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "ggml.h"
#include "ggml-backend.h"

namespace qinf::vision {

// Which projector head the mmproj carries. The vision tower config is shared
// infrastructure; the projector type is the discriminant that selects which
// IVisionEncoder implementation consumes it and which tensor set the loader
// requires. (Parameterize-don't-split: one config, a type knob — NOT a parallel
// config struct per projector.)
enum class VisionProjectorType {
    Gemma3Siglip,  // 27-layer SigLIP ViT + 4×4 pool + projection + soft-emb-norm
    Gemma4Uv,      // blockless im2col projector (gemma4uv); variable token count
};

// SigLIP / Gemma mmproj config. Fields named after the upstream GGUF
// metadata keys where they exist, so the loader can do a direct mapping.
// Defaults describe the Gemma 3 SigLIP-So400M tower; the loader overwrites
// per projector type (see VisionLoader::parse_metadata).
struct SigLIPConfig {
    VisionProjectorType projector_type = VisionProjectorType::Gemma3Siglip;

    // ── Vision tower (SigLIP ViT-So400M) ─────────────────────────────────────
    uint32_t image_size      = 896;   // single-tile only in v1
    uint32_t patch_size      = 14;    // 896/14 = 64×64 patch grid
    uint32_t num_channels    = 3;
    uint32_t hidden_size     = 1152;  // SigLIP-So400M
    uint32_t num_layers      = 27;
    uint32_t num_attn_heads  = 16;
    uint32_t intermediate_size = 4304;  // ViT-So400M FFN dim
    float    layer_norm_eps  = 1e-6f;

    // Token-merge factor. Gemma 4 `gemma4uv` folds an n_merge×n_merge patch
    // merge directly onto the conv: the effective patch size is patch_size *
    // n_merge (16·3 = 48). SigLIP does its merge as a separate 4×4 pool, so its
    // n_merge here is 1 (the pool lives in pool_factor below).
    uint32_t n_merge         = 1;

    // ── Pool + projection + soft-emb-norm ────────────────────────────────────
    uint32_t pool_factor          = 4;    // 4×4 avg pool on the patch grid
    uint32_t mm_tokens_per_image  = 256;  // (64/4)² = 16² = 256
    uint32_t projection_dim       = 0;    // = text_embed_dim of the host text
                                          // model. Filled from the mmproj's
                                          // projection-weight column count.

    // ── Token markers (informational; the text-side tokenizer owns them) ─────
    int32_t image_soft_token_id = -1;
    int32_t boi_token_id        = -1;
    int32_t eoi_token_id        = -1;
};

class VisionModel {
public:
    VisionModel();
    ~VisionModel();

    // Non-copyable, non-movable: holds ggml_context + backend buffer.
    VisionModel(const VisionModel&) = delete;
    VisionModel& operator=(const VisionModel&) = delete;

    // Accessors — populated by VisionLoader.
    const SigLIPConfig& config() const { return config_; }
    SigLIPConfig&       config()       { return config_; }

    // Weight tensors keyed by mmproj GGUF tensor name (e.g.
    // "v.blk.0.attn_q.weight", "mm.input_projection.weight",
    // "mm.soft_emb_norm.weight"). Owned by ctx_; lifetime = VisionModel's.
    const std::unordered_map<std::string, ggml_tensor*>& tensors() const {
        return tensors_;
    }
    std::unordered_map<std::string, ggml_tensor*>& tensors() {
        return tensors_;
    }

    // ggml_context that owns the weight tensors. VisionEncoder borrows
    // these tensors when building its forward graph; it does NOT take
    // ownership.
    ggml_context* weight_context() const { return ctx_; }

    // Backend buffer holding the actual tensor data on the chosen device.
    // VisionLoader allocates this and copies weights into it once at load
    // time. Lifetime = VisionModel's.
    ggml_backend_buffer_t backend_buffer() const { return buffer_; }

    // Source path the mmproj was loaded from. Diagnostic only.
    const std::string& mmproj_path() const { return mmproj_path_; }

    // ── Mutators used by VisionLoader only ────────────────────────────────────
    // (Public for now because the loader is a separate type; if this surface
    // turns out to be wider than needed, lock it down via `friend class
    // VisionLoader` in P2.1. Deferred.)
    void set_context(ggml_context* ctx) { ctx_ = ctx; }
    void set_buffer(ggml_backend_buffer_t buf) { buffer_ = buf; }
    void set_mmproj_path(const std::string& p) { mmproj_path_ = p; }

private:
    SigLIPConfig                                       config_;
    std::unordered_map<std::string, ggml_tensor*>      tensors_;
    ggml_context*                                      ctx_     = nullptr;
    ggml_backend_buffer_t                              buffer_  = nullptr;
    std::string                                        mmproj_path_;
};

}  // namespace qinf::vision
