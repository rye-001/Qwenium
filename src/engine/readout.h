#pragma once
// readout.h — one forward step over a slot, with what a client reads back.
//
// Responsibility: run ONE prefill or ONE decode step over the loaded model and
//   return what the caller asked to read — attention taps, last-row logits, an
//   MoE routing trace — with every per-call override (layer cut, taps, prefill
//   attention, routing capture / replay, sparse decode ids) scoped to the call.
//   The overrides are restored to their previous values before return, also
//   when the step throws.
//
// WHY THIS EXISTS (docs/plan-readout-api.md, approved 2026-10-05). The lens
//   (../Qemmi-lens-server) hand-wrote the same sequence about 12 times —
//   arm → build → mark → alloc_readback_graph → inputs → compute → read →
//   advance → disarm — against ~30 ForwardPassBase methods and raw
//   ggml_backend_sched_* calls, and guarded each with its own restore struct.
//   The stakes of a missed disarm are silent: a leaked truncate_after_layer
//   makes the NEXT ordinary request build a partial stack and return plausible,
//   wrong logits. One owner of the sequence removes that class of bug and gives
//   the client one narrow seam (architecture.md §11 (a)).
//
// This is extraction, not optimization: readout::prefill / decode compute
//   bit-identically to the hand-written sequence (tests/unit/test_readout.cpp,
//   Qwen and Gemma recipes). Nothing here changes numerics.
//
// Unit test: tests/unit/test_readout.cpp

#include <cstdint>
#include <optional>
#include <vector>

#include "ggml-backend.h"
#include "../models/forward_pass_base.h"

namespace qinf::engine::readout {

struct Spec {
    // -1 = the full stack. >= 0 = stop the prefill's layer loop after this
    // physical layer (ForwardPassBase::set_truncate_after_layer). Prefill only:
    // a decode with a layer cut is refused (no recipe honours it on decode).
    int last_layer = -1;
    // Attention taps (ForwardPassBase::set_attention_taps). Empty = no taps.
    // `tap_heads` empty = every head of each tapped layer.
    std::vector<int> tap_layers;
    std::vector<int> tap_heads;
    // Read the step's output logits (Result::logits).
    bool logits = false;
    // MoE routing. `capture` receives the step's expert selections; `replay`
    // forces them. A recipe without MoE layers simply records nothing.
    RoutingTrace* capture = nullptr;
    const RoutingTrace* replay = nullptr;
    // Prefill attention implementation for this call. Unset = leave as is.
    // Taps require Materialized; a tap with Flash fails loud at mark time.
    std::optional<ForwardPassBase::AttnImpl> prefill_attn;
    // Decode only: restrict the output head to these token ids (grammar). Empty
    // = the full vocabulary.
    std::vector<int32_t> sparse_ids;
};

struct Result {
    std::vector<ForwardPassBase::AttentionTap> taps;   // in tap_layers order
    // Every row the graph's output head produced (a prefill builds the head on
    // the last token only, unless set_slice_prefill_head(false)), row-major;
    // `logits_width` floats per row, so the last row is the last
    // `logits_width` floats. Empty and 0 unless spec.logits.
    std::vector<float> logits;
    size_t logits_width = 0;
};

// Prefill `tokens` into `slot` at positions [pos, pos + n): writes the slot's
// KV (and recurrent state) and advances its cache position by n. Fails loud on
// empty tokens or a tap spec under Flash.
Result prefill(ForwardPassBase& fp, ggml_backend_sched_t sched, uint32_t slot, int pos,
               const std::vector<int32_t>& tokens, const Spec& spec);

// One decode row: `token` at rope position `position` in `slot`; advances the
// slot's cache position by 1. One row per call — no client needs a batched
// readout yet, and an untested shape is not offered.
Result decode(ForwardPassBase& fp, ggml_backend_sched_t sched, uint32_t slot, int32_t position,
              int32_t token, const Spec& spec);

}  // namespace qinf::engine::readout
