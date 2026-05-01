#pragma once
// gemma4.h — Gemma 4 (26B-A4B-It, text-only) recipe.
//
// Architecture summary (from docs/plan-gemma-impl.md → Phase G4):
//   - 30 blocks; mixed sliding/global attention per
//     gemma4.attention.sliding_window_pattern (per-layer array of 30).
//   - Sliding layers: head_dim_k = head_dim_v = 256, n_kv_heads = 8,
//     rope_dim = 256, rope_base = 1e4, separate attn_v.weight.
//   - Global  layers: head_dim_k = head_dim_v = 512, n_kv_heads = 2,
//     rope_dim = 512, rope_base = 1e6, V = K (no attn_v.weight in GGUF).
//   - n_head = 16; n_head * head_dim != d_model (= 2816) — the recipe must
//     not assume that identity.
//   - QK-norm per head (head_dim-wide), sandwich norm (post_attention_norm
//     and post_ffw_norm), Gemma-1-style (1+w) RMS-norm encoded into the
//     GGUF weights (so build_rms_norm — the standard x*w form — is correct).
//   - Embedding scale = sqrt(d_model), GeGLU-tanh activation everywhere
//     (dense FFN AND MoE experts).
//   - Tied embeddings (no output.weight; LM head reuses token_embd.weight).
//   - Final logit soft-cap = 30 (G2.4's build_softcap reused — G4.6 wires).
//   - Each block has TWO FFNs in series:
//       1. Dense GeGLU: ffn_norm → ffn_gate/up/down → post_ffw_norm → residual
//       2. MoE  GeGLU: pre_ffw_norm_2 → router (with ffn_gate_inp.scale
//          per-channel input scaling) → top-8 of 128 experts (no shared
//          expert) with fused gate+up tensor → down → layer_output_scale
//          (scalar) → residual.
//     post_ffw_norm_1 / post_ffw_norm_2 are Altup artifacts; Altup is out
//     of scope for G4 and they are skipped in the basic forward path.
//   - PLE (G4.3) and shared-KV (G4.4): INACTIVE in 26B-A4B
//     (embedding_length_per_layer_input == 0; shared_kv_layers == 0); the
//     interfaces ship in this PR but are dormant.  Validation lives in the
//     synthetic-config tests; the dense 31B follow-up promotes them to
//     canary gates.
//   - Per-layer KV storage: two simple_kv_cache instances (one for sliding
//     layers, one for global) because they have different (n_kv_heads,
//     head_dim) shapes.  This is the architectural counter to the assumed
//     "all KV slabs same shape" of pre-G4 recipes.
//
// Scope of this PR (G4.8):
//   - Prefill graph + matching set_inputs (per-layer KQ masks).
//   - Decoding (single-token / batched) is stubbed with a clear error,
//     mirroring G3's stub boundary.  The canary gate runs against prefill
//     logits.
//
// Status of related interfaces:
//   - PLE residual hook in build_transformer_layer (G4.2): the recipe does
//     not call build_transformer_layer (it composes the block manually
//     because of dual-FFN + V==K + dual-cache routing), so the hook is
//     unused here.  Validated in test_transformer_block_ple.cpp.
//   - KvSharingSpec (G4.4): the recipe constructs the global cache with
//     an empty sharing vector (shared_kv_layers == 0).  Validated in
//     test_kv_reference_mode.cpp.

#include "forward_pass_base.h"
#include "../state/kv_cache_simple.h"
#include "../loader/tokenizer_config.h"
#include <vector>

struct ModelMetadata;

// ── Gemma 4 architecture config ──────────────────────────────────────────────
//
// Populated by Gemma4Config::from_metadata.  Per-layer arrays
// (head_count_kv, sliding_window_pattern) are not exposed by GGUFKVBag's
// scalar-only interface — they are derived instead from the tensor
// inventory: a layer is global iff blk.N.attn_v.weight is absent (global
// layers reuse K as V).  That single check determines every per-layer
// attention shape because Gemma 4 has only two attention "kinds".
struct Gemma4Config {
    // Universal
    uint32_t n_layers          = 0;   // gemma4.block_count (= 30 for A4B)
    uint32_t n_head            = 0;   // gemma4.attention.head_count (= 16)
    uint32_t hidden_dim        = 0;   // gemma4.embedding_length (= 2816)
    uint32_t context_len       = 0;   // gemma4.context_length (= 262144)
    float    rms_norm_eps      = 0.f; // gemma4.attention.layer_norm_rms_epsilon

    // Per-layer-kind shape (set on every layer; selected by attn_kind below)
    uint32_t head_dim_global   = 0;   // gemma4.attention.key_length (= 512)
    uint32_t head_dim_swa      = 0;   // gemma4.attention.key_length_swa (= 256)
    uint32_t n_kv_heads_global = 0;   // derived from blk.N (global).attn_k.shape
    uint32_t n_kv_heads_swa    = 0;   // derived from blk.N (sliding).attn_k.shape
    uint32_t rope_dim_global   = 0;   // gemma4.rope.dimension_count (= 512)
    uint32_t rope_dim_swa      = 0;   // gemma4.rope.dimension_count_swa (= 256)
    float    rope_base_global  = 0.f; // gemma4.rope.freq_base (= 1e6)
    float    rope_base_swa     = 0.f; // gemma4.rope.freq_base_swa (= 1e4)
    uint32_t sliding_window    = 0;   // gemma4.attention.sliding_window (= 1024)

    // FFN
    uint32_t ffn_dim_dense     = 0;   // gemma4.feed_forward_length (= 2112)
    uint32_t ffn_dim_expert    = 0;   // gemma4.expert_feed_forward_length (= 704)
    uint32_t n_experts         = 0;   // gemma4.expert_count (= 128)
    uint32_t expert_top_k      = 0;   // gemma4.expert_used_count (= 8)

    // Output head
    float    final_softcap     = 0.f; // gemma4.final_logit_softcapping (= 30)

    // Dormant in A4B; populated from raw_kv but expected to be zero.
    uint32_t shared_kv_layers           = 0; // gemma4.attention.shared_kv_layers
    uint32_t embedding_length_per_layer = 0; // gemma4.embedding_length_per_layer_input

    // Per-layer attention kind: true → global (V==K, head_dim 512, 2 KV heads,
    // rope_base 1e6).  false → sliding (separate V, head_dim 256, 8 KV heads,
    // rope_base 1e4).  Derived from tensor inventory.
    std::vector<bool> is_global;          // size n_layers

    // Per-layer logical index inside its respective KV cache; -1 for layers
    // that don't belong to that cache.  These are precomputed once so the
    // hot path is a single vector lookup.
    std::vector<int>  swa_layer_idx;      // size n_layers; -1 for globals
    std::vector<int>  global_layer_idx;   // size n_layers; -1 for slidings

    // Total per-cache layer counts (cache constructors take these directly).
    uint32_t n_swa_layers      = 0;
    uint32_t n_global_layers   = 0;

    // Factory: reads required scalars from raw_kv, derives the per-layer
    // pattern from the tensor inventory, and validates the documented A4B
    // invariants (PLE / shared-KV inactive; n_kv_heads consistent).
    static Gemma4Config from_metadata(const ModelMetadata& meta);
};

// Validates the tensor inventory for the gemma4 architecture.
// Throws std::runtime_error naming the missing tensor on failure.
// G4.8 covers attention (with the V-conditional check), dual-FFN, sandwich
// norm, MoE expert tensors, layer_output_scale, and the per-layer
// pre_ffw_norm_2.  Altup-only tensors (post_ffw_norm_1 / post_ffw_norm_2)
// are NOT required — they're present in the GGUF but unused in the basic
// forward path; if a future PR enables Altup, the validator widens.
void validate_gemma4_inventory(const ModelMetadata& meta);

// TokenizerConfig for Gemma 4 (SpaceToUnderscore + byte fallback +
// <start_of_turn>/<end_of_turn> chat specials).  Same shape as Gemma 1/2/3
// at the runtime-config level.
TokenizerConfig gemma4_tokenizer_config();

class Gemma4ForwardPass : public ForwardPassBase {
public:
    Gemma4ForwardPass(const Model& model, const ModelMetadata* metadata,
                      uint32_t context_len, uint32_t max_batch_size = 1,
                      int kv_quant_bits = 0);
    ~Gemma4ForwardPass() override = default;

    ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens,
                                      int pos, uint32_t slot_idx = 0) override;

    ggml_cgraph* build_decoding_graph(const std::vector<int32_t>& tokens,
                                      const std::vector<uint32_t>& slots,
                                      const std::vector<int32_t>& positions) override;

    void set_inputs(ggml_cgraph* gf, const std::vector<int32_t>& tokens,
                    int pos) override;

    // DEBUG: instrumented prefill — same as base, but reads back named
    // intermediate tensors after compute and prints stats (G4 hallucination).
    std::vector<float> run_prefill(
        const std::vector<int32_t>& tokens,
        int pos, uint32_t slot_idx,
        ggml_backend_sched_t scheduler) override;

    void set_batched_inputs(ggml_cgraph* gf,
                            const std::vector<int32_t>& tokens,
                            const std::vector<uint32_t>& slots,
                            const std::vector<int32_t>& positions) override;

    // Cache routing: every per-slot position-management op fans out to BOTH
    // the sliding and global caches so the per-layer logical positions stay
    // synchronized.  This works because both caches share the same
    // n_batch_max and n_ctx_max — only the per-layer shapes differ.
    void advance_cache(uint32_t n_tokens, uint32_t slot_idx) override;
    void clear_slot(uint32_t slot_idx) override;
    void set_cache_pos(uint32_t pos, uint32_t slot_idx) override;
    uint32_t get_cache_pos(uint32_t slot_idx) const override;
    uint32_t get_physical_cache_pos(uint32_t slot_idx) const override;
    void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) override;

private:
    Gemma4Config                     config_;
    std::unique_ptr<simple_kv_cache> kv_cache_swa_;
    std::unique_ptr<simple_kv_cache> kv_cache_global_;

    // Per-block weight handles loaded once at construction so the hot path
    // is a vector lookup instead of a string-keyed ggml_get_tensor every
    // graph build.  Layout: index by physical layer index.
    struct BlockWeights {
        // Attention
        ggml_tensor* attn_norm        = nullptr;
        ggml_tensor* attn_q           = nullptr;
        ggml_tensor* attn_k           = nullptr;
        ggml_tensor* attn_v           = nullptr;  // null for global layers (V == K)
        ggml_tensor* attn_q_norm      = nullptr;
        ggml_tensor* attn_k_norm      = nullptr;
        ggml_tensor* attn_output      = nullptr;
        ggml_tensor* post_attn_norm   = nullptr;  // sandwich
        // Dense FFN
        ggml_tensor* ffn_norm         = nullptr;
        ggml_tensor* ffn_gate         = nullptr;
        ggml_tensor* ffn_up           = nullptr;
        ggml_tensor* ffn_down         = nullptr;
        ggml_tensor* post_ffn_norm_1  = nullptr;  // post_ffw_norm_1 — after dense FFN branch
        ggml_tensor* post_ffn_norm    = nullptr;  // post_ffw_norm — outer, after (dense + MoE)
        // MoE FFN
        ggml_tensor* pre_moe_norm     = nullptr;  // pre_ffw_norm_2
        ggml_tensor* post_ffn_norm_2  = nullptr;  // post_ffw_norm_2 — after MoE branch
        ggml_tensor* moe_router_scale = nullptr;  // ffn_gate_inp.scale [n_embd]
        ggml_tensor* moe_router       = nullptr;  // ffn_gate_inp.weight [n_embd, n_experts]
        ggml_tensor* moe_gate_up_exps = nullptr;  // [n_embd, 2*ffn_dim, n_experts] fused
        ggml_tensor* moe_down_exps    = nullptr;  // [ffn_dim, n_embd, n_experts]
        ggml_tensor* layer_out_scale  = nullptr;  // [1] scalar
    };
    std::vector<BlockWeights> block_w_;

    // Resolve a per-block tensor by suffix (mirror of Gemma3's helper).
    ggml_tensor* require_tensor(uint32_t il, const char* suffix) const;
    ggml_tensor* maybe_tensor  (uint32_t il, const char* suffix) const;

    // Build one Gemma 4 block into gf and return the updated residual.
    ggml_tensor* build_block(ggml_cgraph* gf, ggml_tensor* cur,
                              ggml_tensor* inp_pos, uint32_t il,
                              uint32_t slot_idx, uint32_t n_tokens);

    // Inline MoE GeGLU dispatch — separate from MoELayer because
    // MoELayer uses SwiGLU (silu) and Gemma 4 uses GeGLU-tanh.
    // expert_in:  the per-token feature passed to the experts (post pre_ffw_norm_2).
    // router_in:  the per-token feature passed to the router (rms_norm of attn_out
    //             scaled by 1/sqrt(n_embd) and per-channel ffn_gate_inp.scale).
    ggml_tensor* build_moe_geglu(ggml_cgraph* gf,
                                  ggml_tensor* expert_in,
                                  ggml_tensor* router_in,
                                  const BlockWeights& w, uint32_t il,
                                  uint32_t n_tokens);
};
