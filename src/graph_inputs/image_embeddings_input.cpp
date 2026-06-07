#include "image_embeddings_input.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <stdexcept>
#include <string>

void ImageEmbeddingsInput::set_input(const StepContext& step) {
    // require_tensor throws naming the slot if the tensor is absent or the
    // wrong dtype (CLAUDE.md fail-loud: slot, expected, actual).
    ggml_tensor* t = require_tensor(step, slot_, GGML_TYPE_F32);

    const size_t want = static_cast<size_t>(ggml_nelements(t));
    if (data_.size() != want)
        throw std::runtime_error(
            std::string("ImageEmbeddingsInput: slot '") + slot_ +
            "': expected " + std::to_string(want) +
            " floats (n_embd * n_image_tokens from the graph slot), got: " +
            std::to_string(data_.size()) +
            " (armed image-embedding payload size mismatch)");

    ggml_backend_tensor_set(t, data_.data(), 0, data_.size() * sizeof(float));
}
