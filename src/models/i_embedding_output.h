#pragma once

#include <cstdint>
#include <vector>

struct ggml_cgraph;

// ============================================================================
// IEmbeddingOutput — the capability a recipe offers to produce text
// embeddings (2026-10-07). Mirrors i_image_embeddable and i_mtp_draftable: a
// narrow, recipe-side interface that a model-agnostic consumer
// (engine/embed) drives without knowing the family.
//
// The split of work is deliberate:
//   - the RECIPE builds one graph over one whole sequence whose output tensor
//     "embd_tokens" holds one vector per token, already through whatever
//     per-token projection the model has (EmbeddingGemma-2's output.weight is
//     per-token and linear, so projecting before pooling equals projecting
//     after);
//   - engine/embed POOLS those vectors (mean or last token, which the recipe
//     reports from its GGUF) and L2-normalises — the same code for every
//     family, so two families cannot drift on the arithmetic a client compares.
//
// Hosted by gemma_embedding2 (an encoder: no KV cache, no decode). Proven on
// Gemma only. A Qwen embedder (Qwen3-Embedding) is NOT a drop-in: the qwen3
// recipe's graph is cache-bound and ends in an LM head, so it would need a
// second, cache-free causal graph; and embedding GGUFs share
// general.architecture "qwen3" with chat models, so the registry would have to
// decide the capability per FILE, not per architecture (architecture.md §12).
// ============================================================================
class IEmbeddingOutput {
public:
    enum class Pooling { Mean, Last };

    virtual ~IEmbeddingOutput() = default;

    // Build the graph over `tokens` — the complete, already-framed sequence
    // (BOS/EOS as engine/embed adds them), positions 0..n-1. The output tensor
    // "embd_tokens" is F32 [embedding_dim(), tokens.size()]. Inputs are filled
    // by ForwardPassBase::set_prefill_inputs(gf, tokens, 0). Fail-loud on an
    // empty sequence or one longer than the pass was sized for.
    virtual ggml_cgraph* build_embedding_graph(const std::vector<int32_t>& tokens) = 0;

    // Width of one embedding (EmbeddingGemma-2: 768).
    virtual uint32_t embedding_dim() const = 0;

    // How engine/embed reduces the per-token vectors to one.
    virtual Pooling pooling() const = 0;
};
