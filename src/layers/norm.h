#pragma once

struct ggml_context;
struct ggml_tensor;

// Canonical RMS norm + weight scaling shared by all layer types.
// Names the intermediate tensor "cur_rms_normed.il" (or "cur_rms_normed" when
// il < 0) to match the ForwardPassBase::build_norm convention so that graph
// debug output and tensor-name-based lookups in set_inputs remain stable.
ggml_tensor* build_rms_norm(
    ggml_context* ctx,
    ggml_tensor*  cur,
    ggml_tensor*  weight,
    float         eps,
    int           il);
