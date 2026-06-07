#include "vision_model.h"

namespace qinf::vision {

VisionModel::VisionModel() = default;

VisionModel::~VisionModel() {
    // P2.0 stub: no resources owned yet. P2.1 will allocate ctx_/buffer_
    // and the destructor will free them in dependency order:
    //   1. ggml_free(ctx_)         — releases tensor metadata
    //   2. ggml_backend_buffer_free(buffer_) — releases device memory
    // Both are no-ops on nullptr, so the body stays the same shape.
    if (buffer_ != nullptr) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
    if (ctx_ != nullptr) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
}

}  // namespace qinf::vision
