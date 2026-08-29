#pragma once

#include "qwen35.h"   // Qwen35Config — shared by both family hybrids
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
#include "i_image_embeddable.h"
#include "../state/kv_cache_simple.h"
#include "../state/deltanet_state.h"
#include "../layers/moe.h"
#include "../layers/attention.h"  // MRopeSections
#include "../graph_inputs/mrope_positions_input.h"
#include "../graph_inputs/positions_input.h"

#include <cstdint>
#include <memory>
#include <vector>

// Validates the tensor inventory for qwen35moe architecture.
// Throws std::runtime_error naming the missing tensor on failure.
void validate_qwen36_inventory(const ModelMetadata& meta);

// Qwen 3.6's config IS Qwen 3.5's config. The two hybrids differ only in the
// FFN — dense SwiGLU vs routed experts — which Qwen35Config carries as
// expert_count/expert_used_count/expert_feed_forward_length and exposes as
// is_moe(). They were separate, near-identical structs with duplicated helpers
// and duplicated factories until 2026-08-29; the duplication had already begun
// drifting stylistically, which is how copies start drifting semantically.
using Qwen35MoEConfig = Qwen35Config;

class Qwen36ForwardPass : public ForwardPassBase,
                          public IMtpDraftable,
                          public IImageEmbeddable {
public:
    // ── M-RoPE plumbing (P2). Mirrors Qwen35ForwardPass exactly; inp_pos is
    // sized by n_pos_per_token() and filled by the matching graph input, so
    // the two cannot disagree.
    int n_pos_per_token() const {
        return cfg_.mrope_sections.active ? MRopePositionsInput::kComponents : 1;
    }
    void add_positions_input() {
        if (cfg_.mrope_sections.active)
            graph_inputs_.add(std::make_unique<MRopePositionsInput>());
        else
            graph_inputs_.add(std::make_unique<PositionsInput>());
    }

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

    // ── Seam B (P4 of docs/plan-qwen35-vision-impl.md) ───────────────────────
    // Arm precomputed image-token embeddings for the next prefill. The recipe
    // owns the residual stream and performs the substitution; it does not know
    // how the embeddings were produced.
    //
    // grid_w/grid_h are the image's soft-token grid. Unlike Gemma, this recipe
    // USES them: under M-RoPE each image token's position components are
    // (t=pos, h=pos+row, w=pos+col, e=0), so the row and column of every token
    // — hence the grid width — is load-bearing. A non-empty image armed with
    // grid_w == 0 is refused at the splice rather than silently falling back to
    // text positions, which would encode the image as a 1-D run.
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
    // Armed image-token embeddings for the next prefill (empty => text-only).
    // Consumed (moved) by build_prefill_graph. See set_image_embeddings.
    std::vector<float> image_embd_;
    int32_t            image_span_start_ = -1;
    uint32_t           image_n_tokens_   = 0;
    uint32_t           image_grid_w_     = 0;
    uint32_t           image_grid_h_     = 0;

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
