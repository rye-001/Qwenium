#include "graph_input.h"

#include "ggml.h"

#include <stdexcept>
#include <string>

ggml_tensor* GraphInput::require_tensor(const StepContext& step,
                                        const char* slot,
                                        int expected_type) {
    if (!step.gf) {
        throw std::runtime_error(
            std::string("GraphInput: slot '") + slot +
            "': expected a built graph, got: null StepContext::gf");
    }
    ggml_tensor* t = ggml_graph_get_tensor(step.gf, slot);
    if (!t) {
        throw std::runtime_error(
            std::string("GraphInput: slot '") + slot + "': expected type " +
            ggml_type_name(static_cast<ggml_type>(expected_type)) +
            ", got: absent (tensor not found in graph)");
    }
    if (t->type != expected_type) {
        throw std::runtime_error(
            std::string("GraphInput: slot '") + slot + "': expected type " +
            ggml_type_name(static_cast<ggml_type>(expected_type)) +
            ", got: " + ggml_type_name(t->type));
    }
    return t;
}
