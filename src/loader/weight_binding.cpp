#include "weight_binding.h"
#include "../qinf_error.h"

#include <sstream>
#include <stdexcept>
#include <string>

void TensorNameMap::add_slot(const std::string& slot_name, const WeightSlot& slot) {
    slots_[slot_name] = slot;
}

void TensorNameMap::register_name(const std::string& tensor_name,
                                   const std::string& slot_name) {
    name_to_slot_[tensor_name] = slot_name;
}

static std::string shape_str(const int64_t* ne, int n_dims) {
    std::ostringstream ss;
    ss << "[";
    for (int i = 0; i < n_dims; ++i) {
        if (i) ss << ", ";
        ss << ne[i];
    }
    ss << "]";
    return ss.str();
}

static const char* dtype_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_BF16: return "bf16";
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_Q4_K: return "q4_k";
        case GGML_TYPE_Q8_0: return "q8_0";
        default:             return ggml_type_name(t);
    }
}

BindingResult TensorNameMap::bind(const std::vector<TensorDescriptor>& tensors) const {
    BindingResult result;

    // Pass 1: resolve each tensor name → slot, validate shape and dtype.
    for (const auto& desc : tensors) {
        auto it = name_to_slot_.find(desc.name);
        if (it == name_to_slot_.end()) {
            // Failure mode 1: unknown tensor name.
            throw std::runtime_error(
                std::string("weight_binding: unknown tensor name \"") + desc.name + "\"");
        }
        const std::string& slot_name = it->second;
        const WeightSlot&  slot      = slots_.at(slot_name);

        // Failure mode 3: shape mismatch.
        if (desc.n_dims != slot.n_dims) {
            QINF_SLOT_ERROR(slot_name,
                "n_dims " + std::to_string(slot.n_dims),
                "n_dims " + std::to_string(desc.n_dims));
        }
        for (int i = 0; i < slot.n_dims; ++i) {
            if (desc.ne[i] != slot.expected_ne[i]) {
                QINF_SLOT_ERROR(slot_name,
                    "shape " + shape_str(slot.expected_ne, slot.n_dims) +
                    " dtype " + dtype_name(slot.dtype),
                    "shape " + shape_str(desc.ne, desc.n_dims) +
                    " dtype " + dtype_name(desc.dtype));
            }
        }

        // Failure mode 4: dtype mismatch.
        if (desc.dtype != slot.dtype) {
            QINF_SLOT_ERROR(slot_name,
                std::string("shape ") + shape_str(slot.expected_ne, slot.n_dims) +
                " dtype " + dtype_name(slot.dtype),
                std::string("shape ") + shape_str(desc.ne, desc.n_dims) +
                " dtype " + dtype_name(desc.dtype));
        }

        result[slot_name] = desc;
    }

    // Pass 2: verify every declared slot was resolved (failure mode 2).
    for (const auto& kv : slots_) {
        if (result.find(kv.first) == result.end()) {
            const WeightSlot& slot = kv.second;
            QINF_SLOT_ERROR(kv.first,
                std::string("shape ") + shape_str(slot.expected_ne, slot.n_dims) +
                " dtype " + dtype_name(slot.dtype),
                "no tensor found in GGUF");
        }
    }

    return result;
}
