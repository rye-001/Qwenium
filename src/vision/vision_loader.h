#pragma once
// vision_loader.h — read a Gemma 3 mmproj GGUF into a VisionModel.
//
// The mmproj is structurally a normal GGUF with vision-specific tensor
// naming (prefixes `v.`, `mm.`, etc.) and SigLIP/Gemma-3 metadata keys.
// VisionLoader reuses src/loader/GGUFLoader for the binary parse and
// extracts the vision-tower + projection tensors into a VisionModel.
//
// Mirrors the text-side Model::load_metadata / Model::load_tensors split:
//   1. parse_metadata(path, model) — opens the GGUF, populates config().
//   2. load_tensors(model, backend) — allocates a backend buffer, copies
//      tensor data, populates model.tensors().
//
// VisionLoader is a friend-of-VisionModel boundary today (loader writes,
// everyone else reads). If that boundary becomes load-bearing, formalize
// with `friend class VisionLoader` in vision_model.h.
//
// Phase 2.0 (this commit) — declaration only. The bodies in
// vision_loader.cpp throw a CLAUDE.md-shaped error so the build stays
// green and any premature call site fails loudly.

#include <memory>
#include <string>

#include "ggml-backend.h"

class GGUFLoader;

namespace qinf::vision {

class VisionModel;

class VisionLoader {
public:
    VisionLoader();
    ~VisionLoader();

    // Open the mmproj GGUF at `path`, read metadata + tensor inventory,
    // populate `model.config()` and `model.mmproj_path()`. Does NOT copy
    // tensor data yet — that happens in load_tensors so the backend choice
    // is explicit and visible at the call site.
    //
    // Internally uses GGUFLoader with validate_as_text_model=false to skip
    // the registered-architecture + tensor-inventory gates (the mmproj
    // declares general.architecture="clip" and ships v.*/mm.* tensors,
    // neither of which the text-model validators know about). The binary
    // GGUF parse + raw_kv extraction are identical.
    //
    // Fail-loud contract: throws GGUFLoadError on missing required keys,
    // missing required tensors, or shape mismatches (e.g. projection
    // matrix that disagrees with the declared projection_dim).
    void parse_metadata(const std::string& path, VisionModel& model);

    // Allocate `backend`-side storage in `model`, copy tensor data, and
    // populate `model.tensors()` with named ggml_tensor* handles. The
    // backend is passed in (not owned) so the engine can share its Metal
    // backend across text + vision — single device, multiple schedulers.
    //
    // Precondition: parse_metadata has been called on `model` with the
    // same path; this VisionLoader instance still holds the open file.
    void load_tensors(VisionModel& model, ggml_backend_t backend);

private:
    // Owns the open file across parse_metadata → load_tensors. Reset on
    // destruction. One file mapping per VisionLoader instance.
    std::unique_ptr<GGUFLoader> gguf_;
    std::string                 path_;
};

}  // namespace qinf::vision
