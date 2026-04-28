#pragma once
// ffn.h — SwiGLU dense feed-forward network.
//
// Responsibility: fused up/gate/down FFN block with SwiGLU activation.
// Public surface: build_ffn_swiglu free function.
// State owned: none (ggml_context is passed in by the caller).
// Invariants: appends to the caller's ggml_cgraph; creates no ggml_context.
// Reference: ForwardPassBase::ffn_swiglu in src/models/forward_pass_base.cpp.
// Unit test: tests/unit/test_ffn.cpp

#include "layer.h"
#include "ggml.h"

struct ggml_context;
struct ggml_cgraph;

// SwiGLU FFN: down(swiglu(gate(cur), up(cur))).
// Sets tensor names ffn_up.N, ffn_gate.N, ffn_swiglu.N for debuggability.
// phase is accepted for interface uniformity with other layer modules;
// Dense FFN has one graph shape so the argument is ignored.
// Extracted from ForwardPassBase::ffn_swiglu — identical logic.
ggml_tensor* build_ffn_swiglu(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  cur,
    ggml_tensor*  gate,
    ggml_tensor*  up,
    ggml_tensor*  down,
    int           il,
    Phase         phase = Phase::Prefill);

// Gemma-style GeGLU FFN: down(geglu_tanh(gate(cur), up(cur))).
// Uses the tanh-approximation of GELU (`gelu_pytorch_tanh`), not the exact
// erf-based GELU. Wrong variant produces drift that compounds across layers.
ggml_tensor* build_ffn_geglu_tanh(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  cur,
    ggml_tensor*  gate,
    ggml_tensor*  up,
    ggml_tensor*  down,
    int           il,
    Phase         phase = Phase::Prefill);
