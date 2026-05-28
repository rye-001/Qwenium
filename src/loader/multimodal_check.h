#pragma once
// multimodal_check.h — load-time refusal of multimodal-only checkpoints.
//
// Until multimodal support lands (see docs/plan-multimodal-eval.md),
// the engine must fail loud at load time rather than load a vision-stripped
// fine-tune and decode degenerate text.  This module exposes:
//
//   detect_multimodal_only(meta, path) — probe the metadata + a sibling HF
//       config.json (when present) for signals that this checkpoint was
//       trained with a vision tower; returns a DetectionResult naming the
//       trigger.
//   enforce_text_only_or_throw(meta, path) — thin wrapper that calls
//       detect_multimodal_only and throws GGUFLoadError with a CLAUDE.md-
//       shaped message if the result is multimodal.
//
// The two-function split exists so the future multimodal path can call
// detect_multimodal_only() directly and branch into a vision pipeline,
// without having to undo a throw.

#include <string>

struct ModelMetadata;

namespace multimodal_check {

struct DetectionResult {
    bool        is_multimodal = false;
    std::string trigger_field;    // metadata key, tensor name, or sidecar file
    std::string expected_value;   // what a text-only checkpoint looks like
    std::string actual_value;     // what was observed
};

// Probe `meta` and the sibling HF config.json (if one exists next to
// gguf_path) for multimodal-only signals.  Returns is_multimodal=false
// when no signal fires.  Never throws.
DetectionResult detect_multimodal_only(const ModelMetadata& meta,
                                        const std::string& gguf_path);

// Calls detect_multimodal_only and throws GGUFLoadError if a signal
// fired.  Message shape (CLAUDE.md fail-loud contract):
//
//   "multimodal-only checkpoint refused at load: field '<trigger>':
//    expected <expected>, got <actual>. See docs/plan-multimodal-eval.md"
void enforce_text_only_or_throw(const ModelMetadata& meta,
                                 const std::string& gguf_path);

}  // namespace multimodal_check
