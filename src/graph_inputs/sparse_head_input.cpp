#include "sparse_head_input.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <stdexcept>

void SparseHeadInput::set_input(const StepContext& step) {
    if (!step.sparse_ids || step.sparse_ids->empty())
        throw std::runtime_error(
            "SparseHeadInput: slot 'valid_indices': expected non-empty valid set, "
            "got: empty (this input must only be registered on the sparse path)");

    ggml_tensor* t = require_tensor(step, slot_, GGML_TYPE_I32);
    ggml_backend_tensor_set(t, step.sparse_ids->data(), 0,
                            step.sparse_ids->size() * sizeof(int32_t));
}
