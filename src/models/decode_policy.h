#pragma once
// decode_policy.h — the run-time policy a caller sets on a forward pass.
//
// Responsibility: hold the handful of settings that change HOW a pass is built
//   rather than WHAT model it is — and hold them as one value, so "what mode is
//   this pass in" is a single object to read, pass, and reason about instead of
//   five loose members on a base class.
//
// Every one of these is an explicit caller choice with a byte-reproducible
//   default. None is a fallback, and nothing here is inferred at run time:
//     slice_prefill_head — prefill returns last-position logits only (default),
//       vs all positions (speculative verification needs every draft position).
//     output_hidden      — expose the last hidden state as a graph output
//       (MTP drafting, plan-mtp-decode.md §5 D3). Default off.
//     attention_taps     — layers whose attention rows are materialized for the
//       lens (plan-qemmi-lens.md P1/A1). Empty = marks no node.
//     kv_write_mode      — Cpy (default, baked-offset ggml_cpy) vs SetRows
//       (value-driven, position is a graph input).
//     decode_kv_bucket   — 0 (default, exact n_kv) vs B (round up to B).
//
// Invariant worth stating loudly: the DEFAULTS ARE THE BYTE-REPRODUCIBLE PATH.
//   The opt-in seams are byte-inert when disarmed — an empty tap set marks no
//   node, output_hidden adds no output — which is what lets the receipts claims
//   in architecture.md §11 hold for a default-configured pass. Two of these are
//   NOT byte-identical when armed: kv_write_mode=SetRows is byte-identical at
//   exact n_kv but the persistent path turns it on together with bucketing, and
//   bucketing re-blocks the attention reduction, so it is token-stable modulo
//   ties, not bit-identical. That is why --persistent-graph is opt-in.
//
// Extracted from ForwardPassBase (2026-08-29), following GraphArena: the base
//   HOLDS this rather than being it, continuing composition-over-inheritance
//   (architecture.md §12). The base keeps its accessors, delegating here, so no
//   caller changed.
//
// Unit test: tests/unit/test_decode_policy.cpp

#include <cstdint>
#include <vector>

struct DecodePolicy {
    // Differential seam for the decode KV write. Cpy → the legacy baked-offset
    // ggml_cpy write: today's decode, byte-reproducible. SetRows → the
    // value-driven ggml_set_rows write whose position is a graph input, which is
    // what makes a decode graph reusable across steps
    // (docs/plan-persistent-decode-graph.md §2.1). Byte-identical to Cpy at
    // exact n_kv (P1 gate). Only recipes that pass kv_write_indices into the
    // batched attention helpers honor it; others are Cpy-only regardless.
    enum class KvWriteMode { SetRows, Cpy };

    bool             slice_prefill_head = true;
    bool             output_hidden      = false;
    std::vector<int> attention_taps;
    KvWriteMode      kv_write_mode      = KvWriteMode::Cpy;

    // Bucket B ⇒ converted recipes size the decode graph's KV read width
    // (mask / gather / gathered views) at the next multiple of B instead of
    // exactly max_pos+1, so one graph shape — hence one allocation — stays valid
    // across a whole bucket of steps: the persistent-graph precondition
    // (plan-persistent-decode-graph.md §2.2). Padded columns are −inf-masked and
    // read zero-initialized cache rows. 0 = exact sizing.
    uint32_t decode_kv_bucket = 0;

    // Bucketed decode KV width: max_pos_plus_1 rounded up to the bucket, capped
    // at the cache's n_ctx_max. Bucket 0 ⇒ exact (max_pos_plus_1 unchanged).
    // The cap matters: without it a bucket near the end of the context would
    // size the graph past the cache and read rows that do not exist.
    uint32_t decode_kv_len(uint32_t max_pos_plus_1, uint32_t n_ctx_max) const {
        if (decode_kv_bucket == 0) return max_pos_plus_1;
        const uint64_t up =
            (static_cast<uint64_t>(max_pos_plus_1) + decode_kv_bucket - 1) /
            decode_kv_bucket * decode_kv_bucket;
        return up < n_ctx_max ? static_cast<uint32_t>(up) : n_ctx_max;
    }

    // True when the pass is in its default, byte-reproducible configuration.
    // The receipts claims in §11 are made about a pass in this state.
    bool is_default_byte_reproducible() const {
        return slice_prefill_head && !output_hidden && attention_taps.empty()
            && kv_write_mode == KvWriteMode::Cpy && decode_kv_bucket == 0;
    }
};
