#pragma once
// qwen36.h — Forward pass for the Qwen 3.6-35B-A3B hybrid architecture.
//
// Architecture: 40 layers, layer_idx % 4 == 3 → softmax attention (10 layers),
//   else GatedDeltaNet (30 layers). Every layer uses a MoE FFN (256 experts,
//   top-8, 1 shared expert). GGUF architecture string: "qwen35moe".
//
// State owned:
//   simple_kv_cache  — 10 layers, standard F32 KV cache
//   DeltaNetState    — 30 layers, backend-backed recurrent + conv state
//
// Graph shape: one monolithic ggml_cgraph per prefill call (≈2400 nodes,
//   well within the 16 384-node budget). Sub-graph batching (à la Qwen3.5 TQ)
//   is deferred to Phase 4 if the node budget is ever approached.

#include "forward_pass_base.h"
#include "../state/kv_cache_simple.h"
#include "../state/deltanet_state.h"
#include "../layers/moe.h"

#include <cstdint>
#include <memory>
#include <vector>

class Qwen36ForwardPass : public ForwardPassBase {
public:
    Qwen36ForwardPass(const Qwen3Model&     model,
                      const Qwen3Metadata*  metadata,
                      uint32_t              context_len,
                      uint32_t              max_batch_size = 1,
                      int                   kv_quant_bits  = 0);
    ~Qwen36ForwardPass() override = default;

    // ── Graph building ───────────────────────────────────────────────────────
    ggml_cgraph* build_prefill_graph(
        const std::vector<int32_t>& tokens,
        int pos, uint32_t slot_idx = 0) override;

    ggml_cgraph* build_decoding_graph(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>&  positions) override;

    // ── Input setting ────────────────────────────────────────────────────────
    void set_inputs(ggml_cgraph* gf,
                    const std::vector<int32_t>& tokens,
                    int pos) override;

    void set_batched_inputs(ggml_cgraph* gf,
                            const std::vector<int32_t>& tokens,
                            const std::vector<uint32_t>& slots,
                            const std::vector<int32_t>&  positions) override;

    // ── Cache management ─────────────────────────────────────────────────────
    void advance_cache(uint32_t n_tokens, uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->advance(n_tokens, slot_idx);
        // DeltaNet state is updated in-graph; no manual advance needed.
        snapkv_advance_seq_pos(slot_idx, n_tokens);
    }

    void clear_slot(uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->clear_slot(slot_idx);
        if (dn_state_) dn_state_->clear_slot(slot_idx);
        snapkv_clear_seq_pos(slot_idx);
    }

    void set_cache_pos(uint32_t pos, uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->set_pos(pos, slot_idx);
    }

    uint32_t get_cache_pos(uint32_t slot_idx) const override {
        uint32_t seq = snapkv_get_seq_pos(slot_idx);
        if (seq > 0) return seq;
        return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
    }

    uint32_t get_physical_cache_pos(uint32_t slot_idx) const override {
        return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
    }

    void clone_slot(uint32_t src_slot, uint32_t dst_slot,
                    uint32_t n_tokens) override {
        if (kv_cache_) kv_cache_->clone_slot(src_slot, dst_slot, n_tokens);
        if (dn_state_) dn_state_->clone_slot(src_slot, dst_slot);
    }

private:
    std::unique_ptr<simple_kv_cache> kv_cache_;  // 10 attention layers
    std::unique_ptr<DeltaNetState>   dn_state_;   // 30 DeltaNet layers

    // kv_layer_map_[il] = KV cache index (0‥9)  if attention layer, else -1.
    // dn_layer_map_[il] = DeltaNet index (0‥29) if DeltaNet layer,  else -1.
    std::vector<int32_t> kv_layer_map_;
    std::vector<int32_t> dn_layer_map_;

    // Attention hparams cached from metadata (used in both prefill + decode).
    int   n_embd_head_;  // 256
    int   n_rot_;        // 64  — partial RoPE dimension count
    int   n_head_;       // 16
    int   n_head_kv_;    // 2

    MoELayer::Hparams moe_hp_;
};
