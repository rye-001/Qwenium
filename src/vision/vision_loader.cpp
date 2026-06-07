#include "vision_loader.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "ggml.h"
#include "ggml-backend.h"

#include "../core/model.h"
#include "../loader/gguf_loader.h"
#include "vision_model.h"

namespace qinf::vision {

namespace {

// CLAUDE.md fail-loud contract: name the slot, the expected value, the
// actual value, in that order.
[[noreturn]] void throw_slot(const std::string& slot,
                              const std::string& expected,
                              const std::string& actual) {
    throw GGUFLoadError(
        "vision_loader: slot \"" + slot + "\" expected " +
        expected + ", got " + actual);
}

// Verify a required tensor exists with the expected 2-D shape
// (shape[0]=dim0, shape[1]=dim1). For ggml-conventional matmul weights
// like mm.input_projection.weight where ne = [out_dim, in_dim].
void require_tensor_shape_2d(const ModelMetadata& meta,
                              const std::string&   name,
                              uint64_t             dim0,
                              uint64_t             dim1) {
    auto it = meta.tensor_inventory.find(name);
    if (it == meta.tensor_inventory.end()) {
        throw_slot("tensor:" + name, "present in mmproj inventory", "absent");
    }
    const auto& shape = it->second.shape;
    if (shape.size() != 2) {
        throw_slot("tensor:" + name + ".rank",
                   "2", std::to_string(shape.size()));
    }
    if (shape[0] != dim0 || shape[1] != dim1) {
        throw_slot("tensor:" + name + ".shape",
                   "[" + std::to_string(dim0) + ", " + std::to_string(dim1) + "]",
                   "[" + std::to_string(shape[0]) + ", " +
                       std::to_string(shape[1]) + "]");
    }
}

// Verify a required tensor exists as a rank-1 vector of length `dim0`.
// GGUF norm / bias tensors store as shape={dim0} (rank-1), not
// {dim0, 1, 1, 1}.
void require_tensor_shape_1d(const ModelMetadata& meta,
                              const std::string&   name,
                              uint64_t             dim0) {
    auto it = meta.tensor_inventory.find(name);
    if (it == meta.tensor_inventory.end()) {
        throw_slot("tensor:" + name, "present in mmproj inventory", "absent");
    }
    const auto& shape = it->second.shape;
    if (shape.size() != 1) {
        throw_slot("tensor:" + name + ".rank",
                   "1", std::to_string(shape.size()));
    }
    if (shape[0] != dim0) {
        throw_slot("tensor:" + name + ".shape",
                   "[" + std::to_string(dim0) + "]",
                   "[" + std::to_string(shape[0]) + "]");
    }
}

}  // namespace

VisionLoader::VisionLoader() = default;
VisionLoader::~VisionLoader() = default;

void VisionLoader::parse_metadata(const std::string& path,
                                  VisionModel&       model) {
    gguf_  = std::make_unique<GGUFLoader>();
    path_  = path;

    // Skip text-model arch + inventory validation: mmproj declares
    // general.architecture="clip" with vision-only tensor namespace.
    gguf_->load_model(path, /*validate_as_text_model=*/false);

    ModelMetadata meta;
    gguf_->extract_metadata(meta);

    // ── Architecture sanity: this must be an mmproj, not an arbitrary
    //    non-text-model GGUF. We gate on general.architecture=="clip"
    //    AND clip.has_vision_encoder. Both must be present.
    if (meta.architecture != "clip") {
        throw_slot("general.architecture",
                   "clip (mmproj)", "'" + meta.architecture + "'");
    }
    {
        auto has_v = meta.raw_kv.get_bool_opt("clip.has_vision_encoder");
        if (!has_v.has_value() || !*has_v) {
            throw_slot("clip.has_vision_encoder",
                       "present and true",
                       has_v.has_value() ? "false" : "absent");
        }
    }

    // ── Projector type. Two are supported, each with its own IVisionEncoder
    //    and its own required tensor set:
    //      "gemma3"   — 27-layer SigLIP ViT + 4×4 pool + projection + soft-emb-norm.
    //      "gemma4uv" — blockless im2col projector (Gemma 4 unified vision).
    //    Any other projector ships a structurally different head we don't
    //    implement; refuse fail-loud rather than encode and produce garbage.
    auto& cfg = model.config();
    {
        // The newer (Gemma 4) mmproj writer namespaces this under
        // clip.vision.*; the older Gemma 3 mmproj uses the bare clip.* key.
        // Prefer the namespaced key, fall back to the legacy one.
        auto proj_type = meta.raw_kv.get_string_opt("clip.vision.projector_type");
        if (!proj_type.has_value())
            proj_type = meta.raw_kv.get_string_opt("clip.projector_type");
        if (!proj_type.has_value()) {
            throw_slot("clip(.vision).projector_type",
                       "'gemma3' or 'gemma4uv'", "absent");
        } else if (*proj_type == "gemma3") {
            cfg.projector_type = qinf::vision::VisionProjectorType::Gemma3Siglip;
        } else if (*proj_type == "gemma4uv") {
            cfg.projector_type = qinf::vision::VisionProjectorType::Gemma4Uv;
        } else {
            throw_slot("clip.projector_type",
                       "'gemma3' or 'gemma4uv'", "'" + *proj_type + "'");
        }
    }

    // ── Config population + required tensor set, branched by projector type.
    //    Common clip.vision.* numeric keys are uint32 per llama.cpp's writer.
    if (cfg.projector_type == qinf::vision::VisionProjectorType::Gemma3Siglip) {
        cfg.image_size        = meta.raw_kv.get_uint32("clip.vision.image_size");
        cfg.patch_size        = meta.raw_kv.get_uint32("clip.vision.patch_size");
        cfg.num_channels      = 3;  // mmproj is RGB; not exposed as a KV
        cfg.hidden_size       = meta.raw_kv.get_uint32("clip.vision.embedding_length");
        cfg.num_layers        = meta.raw_kv.get_uint32("clip.vision.block_count");
        cfg.num_attn_heads    = meta.raw_kv.get_uint32("clip.vision.attention.head_count");
        cfg.intermediate_size = meta.raw_kv.get_uint32("clip.vision.feed_forward_length");
        cfg.layer_norm_eps    = meta.raw_kv.get_float ("clip.vision.attention.layer_norm_epsilon");
        cfg.projection_dim    = meta.raw_kv.get_uint32("clip.vision.projection_dim");
        cfg.n_merge           = 1;  // SigLIP merges via the 4×4 pool, not the conv

        // mm_tokens_per_image is derivable: (image_size / patch_size / pool_factor)^2.
        // For Gemma 3: (896 / 14 / 4)^2 = 16^2 = 256. We compute it rather than
        // read it (mmproj doesn't carry this key — that token-count is a
        // host-side property of the text-side chunk-list builder).
        const uint32_t patches_per_side = cfg.image_size / cfg.patch_size;
        if (patches_per_side * cfg.patch_size != cfg.image_size) {
            throw_slot("derive(patches_per_side)",
                       "image_size divisible by patch_size",
                       std::to_string(cfg.image_size) + " / " +
                       std::to_string(cfg.patch_size));
        }
        const uint32_t tokens_per_side = patches_per_side / cfg.pool_factor;
        if (tokens_per_side * cfg.pool_factor != patches_per_side) {
            throw_slot("derive(tokens_per_side)",
                       "patches_per_side divisible by pool_factor",
                       std::to_string(patches_per_side) + " / " +
                       std::to_string(cfg.pool_factor));
        }
        cfg.mm_tokens_per_image = tokens_per_side * tokens_per_side;

        // ── Required tensor presence + shape (cross-checks the metadata).
        //    mm.input_projection.weight ne = [projection_dim=2560, hidden=1152]
        //    (ggml-conventional [out_dim, in_dim]).
        require_tensor_shape_2d(meta, "mm.input_projection.weight",
                                cfg.projection_dim, cfg.hidden_size);
        require_tensor_shape_1d(meta, "mm.soft_emb_norm.weight", cfg.hidden_size);
        require_tensor_shape_1d(meta, "v.post_ln.weight", cfg.hidden_size);
        require_tensor_shape_1d(meta, "v.post_ln.bias",   cfg.hidden_size);
        // v.patch_embd.weight is a rank-4 conv weight here; verify presence only.
        if (meta.tensor_inventory.find("v.patch_embd.weight") ==
            meta.tensor_inventory.end()) {
            throw_slot("tensor:v.patch_embd.weight",
                       "present in mmproj inventory", "absent");
        }
        if (meta.tensor_inventory.find("v.position_embd.weight") ==
            meta.tensor_inventory.end()) {
            throw_slot("tensor:v.position_embd.weight",
                       "present in mmproj inventory", "absent");
        }
    } else {  // VisionProjectorType::Gemma4Uv
        // Blockless: block_count / head_count / feed_forward_length are all 0.
        cfg.num_channels      = 3;
        cfg.image_size        = meta.raw_kv.get_uint32("clip.vision.image_size");      // 224 (informational; dyn-size per image)
        cfg.patch_size        = meta.raw_kv.get_uint32("clip.vision.patch_size");      // 16 (raw; effective = patch·n_merge)
        cfg.hidden_size       = meta.raw_kv.get_uint32("clip.vision.embedding_length");// 3840
        cfg.projection_dim    = meta.raw_kv.get_uint32("clip.vision.projection_dim");  // 3840
        cfg.num_layers        = 0;
        cfg.num_attn_heads    = 0;
        cfg.intermediate_size = 0;
        // For gemma4uv the LayerNorms are pytorch-default eps 1e-5 (hardcoded in
        // the encoder); this kv (clip.vision.attention.layer_norm_epsilon = 1e-6)
        // is the encoder's FINAL RMSNorm eps (gemma4uv.cpp uses hparams.eps).
        cfg.layer_norm_eps    = meta.raw_kv.get_float ("clip.vision.attention.layer_norm_epsilon");
        // Token merge is folded onto the conv (effective patch = 16·3 = 48). The
        // gguf carries no clip.vision.projector.scale_factor key, so default 3.
        cfg.n_merge           = meta.raw_kv.get_uint32_opt("clip.vision.projector.scale_factor").value_or(3);
        // Token count is per-image (variable, 40..280) — NOT a constant. The
        // Gemma4UvEncoder derives it from the preprocessed bitmap dims.
        cfg.mm_tokens_per_image = 0;

        const uint32_t eff_patch_area =
            cfg.patch_size * cfg.n_merge * cfg.patch_size * cfg.n_merge * cfg.num_channels;  // 48·48·3 = 6912

        // mm.input_projection.weight ne = [hidden=3840, projection_dim=3840]
        // (build_mm does a DIRECT mul_mat, so ne[0]=in=hidden, ne[1]=out=proj).
        require_tensor_shape_2d(meta, "mm.input_projection.weight",
                                cfg.hidden_size, cfg.projection_dim);
        // im2col patch projection: weight [P·P·C, n_embd], bias [n_embd].
        require_tensor_shape_2d(meta, "v.patch_embd.weight",
                                eff_patch_area, cfg.hidden_size);
        require_tensor_shape_1d(meta, "v.patch_embd.bias", cfg.hidden_size);
        // Three pytorch LayerNorms (1-indexed in this gguf): .1 acts on the
        // im2col output (dim P·P·C), .2 and .3 act on n_embd.
        require_tensor_shape_1d(meta, "v.patch_norm.1.weight", eff_patch_area);
        require_tensor_shape_1d(meta, "v.patch_norm.1.bias",   eff_patch_area);
        require_tensor_shape_1d(meta, "v.patch_norm.2.weight", cfg.hidden_size);
        require_tensor_shape_1d(meta, "v.patch_norm.2.bias",   cfg.hidden_size);
        require_tensor_shape_1d(meta, "v.patch_norm.3.weight", cfg.hidden_size);
        require_tensor_shape_1d(meta, "v.patch_norm.3.bias",   cfg.hidden_size);
        // v.position_embd.weight is rank-3 [n_embd, pos_size, 2] (x/y tables);
        // verify presence only (the 2-D check would understate the rank).
        if (meta.tensor_inventory.find("v.position_embd.weight") ==
            meta.tensor_inventory.end()) {
            throw_slot("tensor:v.position_embd.weight",
                       "present in mmproj inventory", "absent");
        }
        // mm.a.* audio tensors are intentionally ignored (out of scope).
    }

    // ── Token markers are TEXT-MODEL properties (the mmproj is encoder-
    //    side; the tokenizer that uses <image_soft_token> lives on the
    //    text side). We leave config().{image_soft_token_id,
    //    boi_token_id, eoi_token_id} at their -1 defaults; the
    //    Phase 5 orchestrator wires the text-model values in.

    model.set_mmproj_path(path);
}

void VisionLoader::load_tensors(VisionModel& model, ggml_backend_t backend) {
    if (!gguf_ || !gguf_->is_loaded()) {
        throw_slot("VisionLoader.state",
                   "parse_metadata called and file still mapped",
                   gguf_ ? "file unmapped" : "parse_metadata not called");
    }
    if (backend == nullptr) {
        throw_slot("backend",
                   "non-null ggml_backend_t (shared with text engine)",
                   "nullptr");
    }

    // Re-extract the inventory so we know how many tensors we need slots
    // for. parse_metadata already validated the critical few; here we
    // load them all into a ggml_context + backend buffer.
    ModelMetadata meta;
    gguf_->extract_metadata(meta);
    const size_t n_tensors = meta.tensor_inventory.size();

    // Context for tensor METADATA only (no_alloc=true). Sized at
    // tensor_overhead * n_tensors + small slack. Backend buffer is
    // allocated separately by ggml_backend_alloc_ctx_tensors.
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * (n_tensors + 16);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;

    ggml_context* ctx = ggml_init(ip);
    if (ctx == nullptr) {
        throw_slot("ggml_init(vision-weight-ctx)",
                   "non-null", "ggml_init returned nullptr");
    }

    // Build ggml_tensor metadata for every inventory entry. This populates
    // the (name → tensor*) map but does NOT allocate storage yet.
    std::unordered_map<std::string, ggml_tensor*> tensors;
    gguf_->load_tensor_metadata(ctx, tensors);

    // Allocate the backend buffer to hold all of those tensors.
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_free(ctx);
        throw_slot("ggml_backend_alloc_ctx_tensors",
                   "non-null", "allocation failed");
    }

    // Upload tensor data from the mmap'd file into the backend buffer.
    for (const auto& [name, tensor] : tensors) {
        const void* src = gguf_->get_tensor_data(name);
        ggml_backend_tensor_set(tensor, src, 0, ggml_nbytes(tensor));
    }

    // Hand ownership to the model. VisionModel's destructor frees both.
    model.set_context(ctx);
    model.set_buffer(buf);
    model.tensors() = std::move(tensors);

    // The mmap'd file is no longer needed; tensor data is now in the
    // backend buffer. Drop the loader to release the mapping.
    gguf_->unload_model();
}

}  // namespace qinf::vision
