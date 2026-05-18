#include "tokens_input.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <stdexcept>

void TokensInput::set_input(const StepContext& step) {
    if (!step.tokens)
        throw std::runtime_error(
            "TokensInput: slot 'tokens': expected token list, got: null "
            "StepContext::tokens");
    const size_t n_tokens = step.tokens->size();
    ggml_tensor* t = require_tensor(step, slot_, GGML_TYPE_I32);
    ggml_backend_tensor_set(t, step.tokens->data(), 0,
                            n_tokens * sizeof(int32_t));
}
