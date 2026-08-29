#include "vision_profile.h"

#include <stdexcept>

#include "gemma4uv_encoder.h"
#include "qwen3vl_encoder.h"
#include "siglip_encoder.h"

namespace qinf::vision {

using PT = VisionProjectorType;

std::string to_string(PT type) {
    switch (type) {
        case PT::Gemma3Siglip:  return "Gemma3Siglip";
        case PT::Gemma4Uv:      return "Gemma4Uv";
        case PT::Qwen3VlMerger: return "Qwen3VlMerger";
    }
    return "<unregistered VisionProjectorType>";
}

namespace {

// Linear scan of the vocabulary — same lookup both call sites did inline. Runs
// three times, once at setup, on a vocab of ~250 K entries; not hot.
int32_t find_token_id(const std::vector<std::string>& id_to_token,
                      const std::string&              token) {
    for (size_t i = 0; i < id_to_token.size(); ++i)
        if (id_to_token[i] == token) return static_cast<int32_t>(i);
    return -1;
}

// Resolve the three marker tokens or refuse. A vocab without them is a
// text-only checkpoint paired with a multimodal mmproj.
void resolve_markers(VisionProfile&                  profile,
                     const std::vector<std::string>& id_to_token,
                     const std::string&              boi,
                     const std::string&              soft,
                     const std::string&              eoi,
                     const std::string&              error_prefix) {
    profile.boi_id  = find_token_id(id_to_token, boi);
    profile.eoi_id  = find_token_id(id_to_token, eoi);
    profile.soft_id = find_token_id(id_to_token, soft);
    if (profile.boi_id < 0 || profile.eoi_id < 0 || profile.soft_id < 0)
        throw std::runtime_error(
            error_prefix + ": expected the model's tokenizer to define " +
            boi + "/" + soft + "/" + eoi +
            ", actual: a text-only (non-multimodal) vocab");
}

}  // namespace

VisionProfile make_vision_profile(const VisionModel&              vmodel,
                                  ggml_backend_t                   backend,
                                  const std::vector<std::string>&  id_to_token,
                                  const std::string&               error_prefix) {
    const auto& cfg = vmodel.config();
    VisionProfile profile;

    switch (cfg.projector_type) {
        case PT::Gemma3Siglip: {
            profile.projector_tag = "gemma3-siglip";
            resolve_markers(profile, id_to_token,
                            "<start_of_image>", "<image_soft_token>",
                            "<end_of_image>", error_prefix);
            // HF Gemma3Processor wraps the image block in "\n\n" on both sides.
            profile.marker_prefix  = "\n\n<start_of_image>\n\n";
            profile.wants_thinking = false;
            profile.preprocess =
                gemma3_preprocess(static_cast<int>(cfg.image_size));
            // Encoder LAST: it allocates a ggml context + scheduler, so every
            // cheap refusal above happens before we pay for one.
            profile.encoder = std::make_unique<SiglipEncoder>(
                vmodel, backend, cfg.projection_dim);
            return profile;
        }
        case PT::Gemma4Uv: {
            profile.projector_tag = "gemma4uv";
            // Markers per llama.cpp mtmd: <|image> … <image|>. The per-position
            // filler <|image|> is cosmetic — its embedding is overwritten by the
            // substitution; only the N reserved positions matter.
            resolve_markers(profile, id_to_token,
                            "<|image>", "<|image|>", "<image|>", error_prefix);
            // Inline framing, NO surrounding newlines, per llama.cpp mtmd. The
            // gemma3-style "\n\n…\n\n" makes Gemma 4 misread the image
            // ("abstract digital…" instead of the real content). Combined with
            // the thinking branch this matches llama's exact token stream.
            // See docs/server-image-multirequest-bug.md §5.
            profile.marker_prefix  = "<|image>";
            profile.wants_thinking = true;
            profile.preprocess = gemma4uv_preprocess(
                static_cast<int>(cfg.patch_size * cfg.n_merge));
            // Encoder LAST — see the Gemma3Siglip case.
            profile.encoder = std::make_unique<Gemma4UvEncoder>(
                vmodel, backend, cfg.projection_dim);
            return profile;
        }
        case PT::Qwen3VlMerger: {
            profile.projector_tag = "qwen3vl-merger";
            // Qwen 3.5-family markers, per the GGUF chat template.
            resolve_markers(profile, id_to_token,
                            "<|vision_start|>", "<|image_pad|>",
                            "<|vision_end|>", error_prefix);
            profile.marker_prefix  = "<|vision_start|>";
            profile.wants_thinking = false;
            profile.preprocess = qwen3vl_preprocess(
                static_cast<int>(cfg.patch_size * cfg.n_merge));
            // Encoder LAST — see the Gemma3Siglip case.
            profile.encoder = std::make_unique<Qwen3VlEncoder>(
                vmodel, backend, cfg.projection_dim);
            return profile;
        }
    }

    // Fail-loud floor for a projector the loader can parse but this build
    // cannot host. Refusing beats silently encoding with the wrong tower.
    throw std::runtime_error(
        error_prefix + ": expected a projector this build can host (" +
        to_string(PT::Gemma3Siglip) + ", " + to_string(PT::Gemma4Uv) + " or " +
        to_string(PT::Qwen3VlMerger) + "), actual: " +
        to_string(cfg.projector_type) + " from mmproj '" +
        vmodel.mmproj_path() + "'");
}

}  // namespace qinf::vision
