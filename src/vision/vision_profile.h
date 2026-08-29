#pragma once
// vision_profile.h — P0 of docs/plan-qwen35-vision-impl.md.
//
// ONE place that maps a loaded mmproj's VisionProjectorType onto everything
// family-specific about hosting it: which IVisionEncoder to build, the
// embedding-cache identity tag, the three marker token ids, the framing string
// that wraps the image block, whether the recipe needs its thinking branch, and
// the preprocessing recipe.
//
// Before this, that mapping was an if/else duplicated in cli/chat.cpp and
// server/server_vision.cpp whose trailing `else` silently MEANT Gemma4Uv. The
// mmproj loader does reject an unknown projector string before either site is
// reached (vision_loader.cpp:130), so this was latent, not live — but it goes
// live the moment a third VisionProjectorType exists, and both sites would then
// build a Gemma4UvEncoder for it and encode garbage rather than refuse. The
// dispatch below is exhaustive and fail-loud, which is the precondition for P1
// adding `qwen3vl_merger` safely.
//
// Placement: src/vision/, with the encoders it selects between. A profile is
// entirely projector knowledge, so it belongs to the encoder subsystem. This
// became possible when the preprocessing RECIPE moved out of image/image_loader.h
// into vision/image_preprocess.h — the profile carries a recipe, not a pipeline,
// and src/vision/ stays free of image IO exactly as before.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"

#include "i_vision_encoder.h"
#include "image_preprocess.h"
#include "vision_model.h"

namespace qinf::vision {

// Everything the text side needs to know about a projector, resolved once at
// setup. Move-only: it owns the encoder.
struct VisionProfile {
    // The encoder for this projector, already constructed against the mmproj.
    std::unique_ptr<IVisionEncoder> encoder;

    // Stable identity string for the persistent embedding-cache header. A
    // different tag ⇒ different header ⇒ cache miss, never a wrong hit.
    std::string projector_tag;

    // Marker token ids resolved against the host tokenizer's vocabulary.
    // Guaranteed >= 0 on return (the factory throws otherwise).
    int32_t boi_id  = -1;
    int32_t eoi_id  = -1;
    int32_t soft_id = -1;

    // Inserted before the user's text on the image turn; expand_image_markers
    // then turns the single begin-marker token into the soft-token span. The
    // exact framing (with or without surrounding newlines) is empirically
    // load-bearing per family — see the notes at each case in the .cpp.
    std::string marker_prefix;

    // Whether image input on this family requires the recipe's thinking branch.
    bool wants_thinking = false;

    // Resize + normalize recipe the encoder was trained against.
    ImagePreprocess preprocess;
};

// Build the profile for `vmodel`'s projector type.
//
// `id_to_token` is the host tokenizer's vocabulary, used to resolve the three
// marker tokens; a vocab that lacks them is a text-only vocab paired with a
// multimodal mmproj, which is refused.
//
// `error_prefix` is the caller's fail-loud context and parameter, e.g.
// "run_chat: parameter '--image'" or "ServerVision: parameter '--mmproj'". It
// is prepended verbatim so the message names the slot the user actually typed
// (CLAUDE.md fail-loud contract: slot, expected, actual).
//
// Throws std::runtime_error on an unhandled projector type or a vocab missing
// the marker tokens.
VisionProfile make_vision_profile(const VisionModel&              vmodel,
                                  ggml_backend_t                   backend,
                                  const std::vector<std::string>&  id_to_token,
                                  const std::string&               error_prefix);

// Human-readable name for a projector type, for fail-loud messages.
std::string to_string(VisionProjectorType type);

}  // namespace qinf::vision
