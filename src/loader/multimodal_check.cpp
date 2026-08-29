#include "multimodal_check.h"

#include "engine/model.h"
#include "gguf_loader.h"

#include <fstream>
#include <sstream>
#include <string_view>

namespace multimodal_check {

namespace {

// GGUF metadata keys that, when present, indicate the upstream HF model
// shipped a vision tower / image-token pipeline.  
constexpr std::string_view kMultimodalKeys[] = {
    "mm_tokens_per_image",      // Gemma 3 VLM, common HF field
    "image_token_id",           // generic HF
    "image_token_index",        // Gemma 3 / SigLIP variants
    "boi_token_index",          // begin-of-image marker (Gemma 3)
    "eoi_token_index",          // end-of-image marker (Gemma 3)
    "vision_config.hidden_size",
    "vision_config.image_size",
    "vision_config.model_type",
    "vision.hidden_size",
    "vision.image_size",
};

// HuggingFace `AutoModelForX` class names that imply a multimodal head.
// GGUF general.architecture is normally a short string (e.g. "gemma3")
// but third-party converters sometimes surface the full HF class name.
constexpr std::string_view kMultimodalArchSuffixes[] = {
    "ForConditionalGeneration",  // e.g. Gemma3ForConditionalGeneration
};

// Tensor name prefixes that mark a vision encoder or image projector.
// If the GGUF retains any vision-side tensors at all, one of these
// catches it before we wire it into a text-only forward pass.
constexpr std::string_view kMultimodalTensorPrefixes[] = {
    "mm.",                       // multimodal projector
    "v.",                        // llama.cpp llava convention
    "vision.",
    "vision_tower.",
    "multi_modal_projector.",
};

// Vocab token strings whose presence indicates the model was trained to
// inject image patches into the token stream.
constexpr std::string_view kMultimodalVocabTokens[] = {
    "<image_soft_token>",
};

bool ends_with(const std::string& s, std::string_view suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool starts_with(const std::string& s, std::string_view prefix) {
    return s.size() >= prefix.size() &&
           s.compare(0, prefix.size(), prefix) == 0;
}

}  // namespace

DetectionResult detect_multimodal_only(const ModelMetadata& meta,
                                        const std::string& gguf_path)
{
    DetectionResult r;

    // 1. Architecture-string suffix (e.g. "Gemma3ForConditionalGeneration").
    for (auto suffix : kMultimodalArchSuffixes) {
        if (ends_with(meta.architecture, suffix)) {
            r.is_multimodal  = true;
            r.trigger_field  = "general.architecture";
            r.expected_value = "text-only architecture (no '*ForConditionalGeneration' suffix)";
            r.actual_value   = meta.architecture;
            return r;
        }
    }

    // 2. Multimodal scalar keys in the raw KV bag.
    for (auto key : kMultimodalKeys) {
        if (meta.raw_kv.contains(std::string(key))) {
            r.is_multimodal  = true;
            r.trigger_field  = std::string(key);
            r.expected_value = "absent (text-only checkpoint)";
            r.actual_value   = "present";
            return r;
        }
    }

    // 3. Vision-side tensors in the tensor inventory.
    for (const auto& [name, _] : meta.tensor_inventory) {
        for (auto prefix : kMultimodalTensorPrefixes) {
            if (starts_with(name, prefix)) {
                r.is_multimodal  = true;
                r.trigger_field  = "tensor: " + name;
                r.expected_value = "no vision-tower / projector tensors";
                r.actual_value   = "found '" + name + "' (prefix '" + std::string(prefix) + "')";
                return r;
            }
        }
    }

    // 4. Image-patch placeholder in the vocabulary.
    for (size_t i = 0; i < meta.id_to_token.size(); ++i) {
        for (auto needle : kMultimodalVocabTokens) {
            if (meta.id_to_token[i] == needle) {
                r.is_multimodal  = true;
                r.trigger_field  = std::string("vocab[") + std::to_string(i) + "]";
                r.expected_value = "no image-patch placeholder token";
                r.actual_value   = "found '" + std::string(needle) + "'";
                return r;
            }
        }
    }

    // 5. Sibling HF config.json
    auto last_sep = gguf_path.find_last_of("/\\");
    std::string dir = (last_sep == std::string::npos)
                          ? std::string()
                          : gguf_path.substr(0, last_sep + 1);
    std::string sidecar = dir + "config.json";
    std::ifstream f(sidecar);
    if (f) {
        std::stringstream ss; ss << f.rdbuf();
        const std::string body = ss.str();
        if (body.find("\"vision_config\"") != std::string::npos) {
            r.is_multimodal  = true;
            r.trigger_field  = "sibling config.json: vision_config";
            r.expected_value = "absent (text-only models have no vision_config block)";
            r.actual_value   = "present in " + sidecar;
            return r;
        }
    }

    return r;
}

void enforce_text_only_or_throw(const ModelMetadata& meta,
                                 const std::string& gguf_path)
{
    auto r = detect_multimodal_only(meta, gguf_path);
    if (!r.is_multimodal) return;

    throw GGUFLoadError(
        "multimodal-only checkpoint refused at load: field '" +
        r.trigger_field + "': expected " + r.expected_value +
        ", got " + r.actual_value +
        ". See docs/plan-multimodal-eval.md");
}

}  // namespace multimodal_check
