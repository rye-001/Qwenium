#pragma once
// qwen36.h — Forward pass for the Qwen 3.6-35B-A3B hybrid architecture.
//
// Architecture: 40 layers, layer_idx % 4 == 3: softmax attention (10 layers),
//   else GatedDeltaNet (30 layers). Every layer uses a MoE FFN (256 experts,
//   top-8, 1 shared expert). GGUF architecture string: "qwen35moe".
//
// State owned:
//   simple_kv_cache  — 10 layers, standard F32 KV cache
//   DeltaNetState    — 30 layers, backend-backed recurrent + conv state
//
// Graph shape: one monolithic ggml_cgraph per prefill call (≈2400 nodes,
//   well within the 16 384-node budget).

#include "forward_pass_base.h"
#include "i_mtp_draftable.h"
#include "../state/kv_cache_simple.h"
#include "../state/deltanet_state.h"
#include "../layers/moe.h"

#include <cstdint>
#include <memory>
#include <vector>

// Validates the tensor inventory for qwen35moe architecture.
// Throws std::runtime_error naming the missing tensor on failure.
void validate_qwen36_inventory(const ModelMetadata& meta);

// Typed config for the qwen35moe recipe.  Holds only the fields that are
// family-specific to qwen35moe (SSM / DeltaNet, MoE, partial-RoPE dimension).
// Universal fields (block_count, embedding_length, head counts, RoPE base,
// RMS eps) stay on ModelMetadata and are read directly by the recipe.
//
// Construct once at recipe startup via from_metadata(); all subsequent
// graph-building code reads cfg_.* for these fields.
struct Qwen35MoEConfig {
    // SSM / DeltaNet block
    uint32_t ssm_conv_kernel;
    uint32_t ssm_state_size;
    uint32_t ssm_group_count;
    uint32_t ssm_time_step_rank;
    uint32_t ssm_inner_size;

    // MoE block
    uint32_t expert_count;
    uint32_t expert_used_count;
    uint32_t expert_feed_forward_length;

    // Hybrid-attention
    uint32_t rope_dimension_count;
    uint32_t full_attention_interval;

    // MTP / NextN head: number of trailing blocks that are the multi-token
    // prediction head rather than main decode layers (docs/plan-mtp-decode.md).
    // 0 for a standard GGUF. The head's tensors are bound (§4) but not executed
    // until Phase 3/4; here it only shortens the main decode stack.
    uint32_t nextn_predict_layers;

    // Capability query (mirrors i_image_embeddable's presence check): does this
    // model carry a trained MTP head?
    bool has_mtp_head() const { return nextn_predict_layers > 0; }

    // Layer-kind helpers — identical semantics to ModelMetadata::is_*_layer
    // for the qwen35moe architecture, but self-contained on the config so the
    // recipe does not read through ModelMetadata at graph-building time.
    bool is_full_attention_layer(uint32_t il) const {
        if (full_attention_interval == 0) return true;
        return (il % full_attention_interval) == (full_attention_interval - 1);
    }
    bool is_ssm_layer(uint32_t il) const { return !is_full_attention_layer(il); }

    // Factory: copies family-specific fields from meta and validates
    // qwen35moe-specific invariants.  Throws std::runtime_error on violation
    static Qwen35MoEConfig from_metadata(const ModelMetadata& meta);
};

class Qwen36ForwardPass : public ForwardPassBase, public IMtpDraftable {
public:
    Qwen36ForwardPass(const Model&     model,
                      const ModelMetadata*  metadata,
                      uint32_t              context_len,
                      uint32_t              max_batch_size = 1,
                      ggml_type             kv_type = GGML_TYPE_F32);
    ~Qwen36ForwardPass() override = default;

    // ── Graph building ───────────────────────────────────────────────────────
    ggml_cgraph* build_prefill_graph(
        const std::vector<int32_t>& tokens,
        int pos, uint32_t slot_idx = 0, bool want_logits = true) override;

    bool feed_tokens_supported() const override { return true; }

    // Decode graph is persistent-capable: every step-varying quantity is a
    // graph-input value (P1, docs/plan-persistent-decode-graph.md).
    bool supports_persistent_decode() const override { return true; }

    ggml_cgraph* build_decoding_graph(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>&  positions) override;

    // ── Cache management ─────────────────────────────────────────────────────
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

    void clone_slot(uint32_t src_slot, uint32_t dst_slot,
                    uint32_t n_tokens) override {
        if (kv_cache_) kv_cache_->clone_slot(src_slot, dst_slot, n_tokens);
        if (dn_state_) dn_state_->clone_slot(src_slot, dst_slot);
    }

    // State reach-through (mirrors qwen35): L2 snapshots and the speculative
    // rollback path (checkpoint before verify, restore on partial reject) both
    // need these. qwen36 lacked them, which silently disabled the hybrid
    // rollback — snapshot_recurrent() fell through to the nullptr default.
    simple_kv_cache* snapshot_kv_cache() override { return kv_cache_.get(); }
    DeltaNetState*   snapshot_recurrent() override { return dn_state_.get(); }

    // ── IMtpDraftable (docs/plan-mtp-decode.md §4, Phase 3) ─────────────────
    // The NextN head: one gated-attention + MoE block (blk.40) seeded from the
    // main model's pre-final-norm hidden. Private single-slot KV, reset per
    // call — drafting is stateless across decode steps (§4.6). Pass a
    // DEDICATED scheduler: the head graph is a new shape, and scheduler/galloc
    // reuse across graph shapes is the known corruption mode
    // (docs/server-image-multirequest-bug.md).
    bool mtp_supported() const override { return cfg_.has_mtp_head(); }

    std::vector<int32_t> mtp_draft(
        uint32_t                  slot,
        const std::vector<float>& hidden,
        int32_t                   last_token,
        int                       pos,
        uint32_t                  k,
        ggml_backend_sched_t      sched) override;

private:
    // One NextN draft step: embed(token) + hidden → eh_proj → block 40 →
    // shared_head_norm → logits ("mtp_logits") + chained hidden ("mtp_h_next").
    // n_past = entries already in the private head KV this draft attempt.
    ggml_cgraph* build_mtp_graph(uint32_t n_past);


    Qwen35MoEConfig cfg_;  // family-specific config, derived from ModelMetadata at construction

    std::unique_ptr<simple_kv_cache> kv_cache_;  // 10 attention layers
    std::unique_ptr<DeltaNetState>   dn_state_;   // 30 DeltaNet layers

    // NextN head's private KV — 1 layer, 1 slot, tiny context (draft window
    // only; reset per mtp_draft call). NOT a new engine state kind: an
    // ordinary append-KV object private to the drafting path (§4.4).
    std::unique_ptr<simple_kv_cache> mtp_kv_;

    // kv_layer_map_[il] = KV cache index (0‥9)  if attention layer, else -1.
    // dn_layer_map_[il] = DeltaNet index (0‥29) if DeltaNet layer,  else -1.
    std::vector<int32_t> kv_layer_map_;
    std::vector<int32_t> dn_layer_map_;

    // Main decode layers = block_count − nextn_predict_layers. The trailing
    // NextN block(s) are held out of every main-stack loop below.
    uint32_t n_main_layers_ = 0;

    // Attention hparams cached from metadata (used in both prefill + decode).
    int   n_embd_head_;  // 256
    int   n_rot_;        // 64  — partial RoPE dimension count
    int   n_head_;       // 16
    int   n_head_kv_;    // 2

    MoELayer::Hparams moe_hp_;
};
