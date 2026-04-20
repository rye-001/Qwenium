#pragma once
// weight_binding.h — declarative tensor-name → module-slot binding.
//
// Responsibility: resolve GGUF tensor names to typed weight slots and validate
//   shape and dtype at load time.  All four failure modes are fail-loud errors:
//   unknown name, missing slot, shape mismatch, dtype mismatch.
// Public surface: WeightSlot, TensorDescriptor, TensorNameMap, BindingResult.
// State owned: none — TensorNameMap is declarative; bind() is stateless.
// Invariants:
//   - Every tensor in the GGUF list must resolve to a registered slot.
//   - Every registered slot must be resolved by exactly one tensor.
//   - Shape and dtype must match exactly; no implicit casts.
// Reference: docs/modular-layer-architecture.md § Weight Binding.
// Unit test: tests/unit/test_weight_binding.cpp

#include "ggml.h"
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// Describes what a module slot expects from a GGUF tensor.
struct WeightSlot {
    std::string name;              // slot identifier, e.g. "q_proj"
    int64_t     expected_ne[4] = {0, 0, 0, 0};  // expected dimensions (ne[0]..ne[n_dims-1])
    int         n_dims = 0;
    ggml_type   dtype  = GGML_TYPE_F16;
};

// Minimal description of a tensor as read from a GGUF file.
// Data pointer is not stored here — binding is a validation pass only.
struct TensorDescriptor {
    std::string name;
    int64_t     ne[4] = {0, 0, 0, 0};
    int         n_dims = 0;
    ggml_type   dtype  = GGML_TYPE_F16;
};

// Maps GGUF tensor names to slot names, validates shape and dtype.
// BindingResult maps slot name → the descriptor that was bound to it.
using BindingResult = std::map<std::string, TensorDescriptor>;

class TensorNameMap {
public:
    // Declare a required weight slot.
    void add_slot(const std::string& slot_name, const WeightSlot& slot);

    // Map a GGUF tensor name to a slot name.
    // Multiple tensor names may map to the same slot (layer-indexed patterns).
    void register_name(const std::string& tensor_name, const std::string& slot_name);

    // Walk tensors, resolve each name, validate shape+dtype, return bound result.
    // Throws std::runtime_error on any of the four failure modes:
    //   1. tensor_name not registered → unknown-name error
    //   2. slot has no tensor → missing-slot error
    //   3. shape mismatch → slot-error with expected/actual
    //   4. dtype mismatch → slot-error with expected/actual
    BindingResult bind(const std::vector<TensorDescriptor>& tensors) const;

private:
    std::unordered_map<std::string, WeightSlot> slots_;
    std::unordered_map<std::string, std::string> name_to_slot_;
};
