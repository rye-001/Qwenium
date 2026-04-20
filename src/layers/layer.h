#pragma once
// layer.h — common types shared by all layer modules.
//
// Responsibility: vocabulary that every layer module uses; kept minimal so
//   including this header does not drag in heavy dependencies.
// Contents: Phase enum (controls prefill vs decode graph topology).
// Reference: docs/modular-layer-architecture.md § Prefill vs Decode.

// Controls which graph topology a layer module emits.
// Passed explicitly so the memory allocator sees a single-topology graph per
// forward pass — mixing both paths in one graph defeats ggml_gallocr reuse.
enum class Phase {
    Prefill,  // batched prompt processing: full causal attention over all input tokens
    Decode,   // single-step autoregressive: one token per slot, attending to KV cache
};
