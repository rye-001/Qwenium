#pragma once

#include "forward_pass_base.h"
#include "i_image_embeddable.h"   // Seam B
#include "../layers/attention.h"  // MRopeSections
#include "../graph_inputs/mrope_positions_input.h"
#include "../graph_inputs/positions_input.h"
#include "../layers/deltanet.h"
#include "../state/deltanet_state.h"
#include "../state/kv_cache_simple.h"

// Validates the tensor inventory for qwen35 architecture.
// Throws std::runtime_error naming the missing tensor on failure.
void validate_qwen35_inventory(const ModelMetadata& meta);

// Typed config for the qwen35 recipe.  Holds only the fields that are
// family-specific to qwen35 (SSM / DeltaNet, partial-RoPE dimension, and
// the hybrid-attention interval that backs the layer-kind helpers).
// Universal fields (block_count, embedding_length, head counts, RoPE base,
// RMS eps) stay on ModelMetadata and are read directly by the recipe.
struct Qwen35Config {
    // SSM / DeltaNet block
    uint32_t ssm_conv_kernel;
    uint32_t ssm_state_size;
    uint32_t ssm_group_count;
    uint32_t ssm_time_step_rank;
    uint32_t ssm_inner_size;

    // MoE block — ZERO on a dense checkpoint. `is_moe()` is the seam between the
    // two Qwen 3.5-family hybrids: qwen35 (dense SwiGLU FFN) and qwen35moe /
    // Qwen 3.6 (routed experts). They are otherwise the same recipe — same
    // DeltaNet/attention interleave, same M-RoPE, same NextN head-out — so the
    // FFN choice is a PARAMETER here, exactly as Gemma 4 treats its own dense
    // vs MoE split, rather than a second recipe.
    uint32_t expert_count = 0;
    uint32_t expert_used_count = 0;
    uint32_t expert_feed_forward_length = 0;

    // Partial-RoPE dimension (0 -> fall back to full head dimension at use sites)
    uint32_t rope_dimension_count;

    // M-RoPE section widths from qwen35.rope.dimension_sections (P2 of
    // docs/plan-qwen35-vision-impl.md). Every Qwen 3.5-family GGUF declares
    // [11, 11, 10, 0]; absent ⇒ inactive ⇒ plain NEOX, which is what this
    // engine did before P2. Text-only output is identical either way — the
    // four components carry the same position until an image splits them.
    MRopeSections mrope_sections;

    // Hybrid-attention scheduling
    uint32_t full_attention_interval;

    // Depth of the trailing NextN / MTP head (0 = none).  Qwen3.5-9B ships 0;
    // Qwen3.8-27B ships 1, i.e. block_count 65 = 64 main + 1 NextN.  The head
    // is held out of the main decode stack — see n_main_layers_.
    uint32_t nextn_predict_layers;

    bool has_mtp_head() const { return nextn_predict_layers > 0; }

    // The dense/MoE seam. Derived from the checkpoint, never configured.
    bool is_moe() const { return expert_count > 0; }

    bool is_full_attention_layer(uint32_t il) const {
        if (full_attention_interval == 0) return true;
        return (il % full_attention_interval) == (full_attention_interval - 1);
    }
    bool is_ssm_layer(uint32_t il) const { return !is_full_attention_layer(il); }

    // Factory: copies family-specific fields from meta and validates the
    // family's invariants. Throws std::runtime_error on violation.
    //
    // Serves BOTH architectures. The GGUF key prefix IS meta.architecture
    // ("qwen35" or "qwen35moe"), so one reader covers both and every error names
    // the key the checkpoint actually should have carried. The expert keys are
    // read only for qwen35moe; rope.dimension_count is required there and
    // optional (falling back to the head dimension) for qwen35 — preserved
    // exactly as the two separate factories behaved.
    static Qwen35Config from_metadata(const ModelMetadata& meta);
};

/**
 * Forward pass for the Qwen3.5 hybrid DeltaNet/attention architecture.
 *
 * Hosts every model whose GGUF declares arch "qwen35" — the Qwen3.5 and
 * Qwen3.8 releases alike.  Layer counts are read from metadata, never
 * assumed: Qwen3.5-9B is 32 layers (24 DeltaNet + 8 attention), Qwen3.8-27B
 * is 64 (48 + 16) plus one NextN head block.  Full attention lands on every
 * `full_attention_interval`-th layer (3, 7, 11, ...).
 *
 * Key differences from Qwen2/3:
 *   - Joint Q+Gate projection in attention layers (strided view extraction)
 *   - Gate sigmoid gating after attention output
 *   - Partial RoPE (64 of 256 dims)
 *   - KV cache only for the attention layers, not every layer
 *   - Fixed-size DeltaNet recurrent state + conv state for the SSM layers
 *   - post_attention_norm instead of ffn_norm
 *   - Optional trailing NextN / MTP head block, excluded from the decode
 *     stack: the stack is n_main_layers_ deep, not meta_.block_count.
 */
class Qwen35ForwardPass : public ForwardPassBase,
                          public IImageEmbeddable {
public:
    // ── M-RoPE plumbing (P2 of docs/plan-qwen35-vision-impl.md) ──────────────
    // Position components per token: 4 when the GGUF declares
    // rope.dimension_sections, else 1. inp_pos is sized by this and filled by
    // the matching graph input, so the two can never disagree.
    int n_pos_per_token() const {
        return cfg_.mrope_sections.active ? MRopePositionsInput::kComponents : 1;
    }

    Qwen35ForwardPass(const Model& model, const ModelMetadata* metadata,
                      uint32_t context_len, uint32_t max_batch_size = 1,
                      ggml_type kv_type = GGML_TYPE_F32);
    ~Qwen35ForwardPass() override = default;

    // --- Graph building ---
    struct ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx = 0, bool want_logits = true) override;

    ggml_cgraph* build_decoding_graph(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions) override;

    // qwen35 honors want_logits=false with one head-guard site in
    // build_prefill_graph.
    bool feed_tokens_supported() const override { return true; }

    // Decode graph is persistent-capable: every step-varying quantity is a
    // graph-input value (P1, docs/plan-persistent-decode-graph.md).
    bool supports_persistent_decode() const override { return true; }

    // ── Seam B (docs/plan-qwen35-vision-impl.md) ─────────────────────────────
    // Identical in shape to Qwen36ForwardPass: both host the same projector
    // (qwen3vl_merger) and the same M-RoPE position construction, and differ
    // only in the layer stack the substituted residual stream then flows
    // through. Qwen 3.8-27B is the target that made this recipe multimodal.
    //
    // grid_w/grid_h are load-bearing here, as on qwen36: under M-RoPE an image
    // token's components are (t=pos, h=pos+row, w=pos+col, e=0), so the grid
    // width decides the row/column of every token. An image armed with
    // grid_w == 0 is refused at the splice rather than silently encoded as a
    // 1-D run.
    void set_image_embeddings(std::vector<float> embd,
                              int32_t span_start,
                              uint32_t n_tokens,
                              uint32_t grid_w = 0,
                              uint32_t grid_h = 0) override {
        image_embd_       = std::move(embd);
        image_span_start_ = span_start;
        image_n_tokens_   = n_tokens;
        image_grid_w_     = grid_w;
        image_grid_h_     = grid_h;
    }

    // M-RoPE puts an image on a 2-D position grid, so the span advances the
    // sequence position by max(nx, ny), not by its token count. Without the
    // sections declared this recipe is on scalar NEOX positions and the image
    // is a plain 1-D run, same as Gemma.
    bool image_span_is_2d() const override {
        return cfg_.mrope_sections.active;
    }

    // --- Cache management (delegates to KV and SSM caches) ---
    void advance_cache(uint32_t n_tokens, uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->advance(n_tokens, slot_idx);
    }

    void clear_slot(uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->clear_slot(slot_idx);
        if (dn_state_) dn_state_->clear_slot(slot_idx);
    }

    void set_cache_pos(uint32_t pos, uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->set_pos(pos, slot_idx);
    }

    uint32_t get_cache_pos(uint32_t slot_idx) const override {
        return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
    }

    void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) override {
        if (kv_cache_) kv_cache_->clone_slot(src_slot, dst_slot, n_tokens);
        if (dn_state_) dn_state_->clone_slot(src_slot, dst_slot);
    }

    // --- Accessors ---
    simple_kv_cache* get_kv_cache_ptr() { return kv_cache_.get(); }
    DeltaNetState*   get_dn_state_ptr() { return dn_state_.get(); }

    // L2 snapshot reach-through (both lanes: hybrid attention + DeltaNet).
    simple_kv_cache* snapshot_kv_cache() override { return kv_cache_.get(); }
    DeltaNetState*   snapshot_recurrent() override { return dn_state_.get(); }

    // Physical layer → cache index mappings
    int32_t get_kv_layer_index(uint32_t physical_layer) const {
        return (physical_layer < kv_layer_map_.size()) ? kv_layer_map_[physical_layer] : -1;
    }
    int32_t get_ssm_layer_index(uint32_t physical_layer) const {
        return (physical_layer < ssm_layer_map_.size()) ? ssm_layer_map_[physical_layer] : -1;
    }

private:
    // Armed image-token embeddings for the next prefill (empty => text-only).
    // Consumed (moved) by build_prefill_graph. See set_image_embeddings.
    std::vector<float> image_embd_;
    int32_t            image_span_start_ = -1;
    uint32_t           image_n_tokens_   = 0;
    uint32_t           image_grid_w_     = 0;
    uint32_t           image_grid_h_     = 0;

    Qwen35Config cfg_;  // family-specific config, derived from ModelMetadata at construction

    // Depth of the main decode stack: meta_.block_count minus the trailing
    // NextN / MTP head blocks.  Every graph-building loop and every layer map
    // is sized to this, NOT to block_count — the head is loaded but not run
    // as part of the residual stream.
    uint32_t n_main_layers_ = 0;

    std::unique_ptr<simple_kv_cache> kv_cache_;       // attention layers
    std::unique_ptr<DeltaNetState>   dn_state_;       // DeltaNet recurrent state (always present)

    // Physical layer index → cache layer index (-1 = not this cache type)
    // Sized to n_main_layers_; entry is the index into the corresponding
    // cache, or -1 when the physical layer is not of that kind.
    std::vector<int32_t> kv_layer_map_;    // physical layer → KV cache index
    std::vector<int32_t> ssm_layer_map_;   // physical layer → DeltaNet state index
};