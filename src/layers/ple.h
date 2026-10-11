#pragma once
// ple.h — Per-Layer Embeddings (PLE), the Gemma 4 family's second signal path.
//
// Every layer gets its own small input vector per token, and after the layer's
// FFN residual that vector gates a low-rank update of the residual stream:
//
//   once per pass:  ple_in = rms_norm(model_proj · x_scaled / sqrt(n_embd),
//                                     proj_norm)            → [ple_dim, n_tokens, n_layers]
//   per layer il:   cur += post_norm(proj · (gelu(inp_gate · cur) ⊙ ple_in[il]))
//
// It is its own module, not a knob on the FFN or the embedding, because it is a
// different signal flow: one projection of the input feeds every layer, and each
// layer injects it at the END of the block.
//
// Public surface:
//   build_ple_inputs — the once-per-pass per-layer inputs
//   build_ple_inject — one layer's gated injection; returns the new residual
//
// State owned: none — weights are borrowed. Appends to the caller's graph;
// creates no ggml_context.
//
// History (2026-10-07). This file used to hold build_ple: a per-layer TOKEN
// table looked up, projected to hidden_dim and added at block ENTRY. It was
// designed in G4 against synthetic configs, never ran on a real model, and
// matches neither EmbeddingGemma-2 (this flow, the first real PLE model we
// host) nor llama.cpp's Gemma 4 E2B/E4B. It had no production caller and was
// replaced, not kept beside this. The E-models ALSO add a per-layer token
// table to ple_in (scaled by sqrt(ple_dim), then the sum scaled by 1/sqrt(2));
// that term is not here because no model we host carries it — add it with the
// model that gates it.
//
// Reference: llama.cpp src/models/gemma-embedding2.cpp (build_inp_per_layer
//   and the per-layer block), f498f864f.
// Unit test:  tests/unit/test_ple.cpp

#include <cstdint>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

// x_scaled   : [n_embd, n_tokens] — the token embeddings AFTER the sqrt(n_embd)
//              embed scale (the same tensor the first block reads).
// model_proj : [n_embd, ple_dim * n_layers] (per_layer_model_proj.weight)
// proj_norm  : [ple_dim]                    (per_layer_proj_norm.weight)
// Returns [ple_dim, n_tokens, n_layers], contiguous, so a layer's slice is a
// plain 2-D view. Fail-loud if model_proj's width is not ple_dim * n_layers.
ggml_tensor* build_ple_inputs(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  x_scaled,
    ggml_tensor*  model_proj,
    ggml_tensor*  proj_norm,
    int64_t       ple_dim,
    int64_t       n_layers,
    float         eps);

// cur        : [n_embd, n_tokens] — the residual after the layer's FFN add
// ple_inputs : build_ple_inputs' result
// inp_gate   : [n_embd, ple_dim]   (blk.il.inp_gate.weight)
// proj       : [ple_dim, n_embd]   (blk.il.proj.weight)
// post_norm  : [n_embd]            (blk.il.post_norm.weight)
// Returns cur + post_norm(proj · (gelu(inp_gate · cur) ⊙ ple_inputs[il])).
ggml_tensor* build_ple_inject(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  cur,
    ggml_tensor*  ple_inputs,
    ggml_tensor*  inp_gate,
    ggml_tensor*  proj,
    ggml_tensor*  post_norm,
    float         eps,
    int           il);
