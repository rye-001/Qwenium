#pragma once
// ple.h — Per-Layer Embedding (PLE) lookup + projection.
//
// PLE is a Gemma 4 feature: each transformer block owns its own
// low-dimensional embedding table.  At runtime, the per-layer entry for
// each input token is looked up, projected into hidden_dim, and added to
// the residual stream at the block-entry injection point (the second
// residual slot G4.2 added to build_transformer_layer).
//
// Public surface:
//   build_ple(ctx, gf, token_ids, ple_table, ple_proj, il)
//     → [hidden_dim, n_tokens] residual contribution.
//
// State owned: none — all weight tensors are borrowed references.
// Invariants: appends to the caller's ggml_cgraph; creates no ggml_context.
//
// Status in 26B-A4B: INACTIVE.  embedding_length_per_layer_input == 0 in the
// 26B-A4B GGUF, so the recipe never wires PLE through.  The interface ships
// in G4 and is exercised by a synthetic-config unit test (test_ple.cpp).
// The dense 31B follow-up will exercise it end-to-end.
//
// Reference: docs/plan-gemma-impl.md → Phase G4 → "Per-Layer Embeddings (PLE)"
// Unit test:  tests/unit/test_ple.cpp

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

// Build the PLE residual contribution for one transformer layer.
//
// Inputs:
//   token_ids : [n_tokens] I32 — same token-id vector used by the input
//               embedding lookup.  This module does not own it; the recipe
//               passes the same tensor it routed into ggml_get_rows for
//               the main embedding.
//   ple_table : [ple_dim, vocab_size] — the per-layer embedding table.
//               Each layer's table is independent; the recipe selects the
//               correct one before this call.
//   ple_proj  : [hidden_dim, ple_dim] — projection from the low-dim space
//               into the residual-stream width.
//   il        : physical layer index (used only for tensor naming so debug
//               graph dumps remain unambiguous).
//
// Returns: [hidden_dim, n_tokens] — pass directly as ple_residual to
// build_transformer_layer.  Caller is responsible for adding it to the
// residual stream (the block does the add internally at the injection
// point — do not pre-add here).
ggml_tensor* build_ple(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  token_ids,
    ggml_tensor*  ple_table,
    ggml_tensor*  ple_proj,
    int           il);
