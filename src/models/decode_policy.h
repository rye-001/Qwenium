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
//     attn_impl          — DECODE: Materialized (default, kq/soft_max/kqv
//       written out) vs Flash (one ggml_flash_attn_ext). Flash never
//       materializes kq_soft, so it is mutually exclusive with a tapped
//       decode — see below.
//     prefill_attn_impl  — the same choice, scoped to PREFILL. Separate
//       because the two phases are tapped by different callers: `extract`
//       taps decode and never taps prefill, so its prefill — the dominant
//       cost on a lens workload — can run flash while the tapped decode
//       stays materialized (docs/plan-lens-server-shape.md §4.2.1).
//     truncate_after_layer — PREFILL-only: stop the layer loop after this
//       physical layer instead of building every layer through the output
//       head. -1 (default) = full stack. Introduced for teacher-forced lens
//       verification (docs/plan-lens-server-shape.md §3.4) — see below.
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

    // How the attention core is built. Materialized (default) writes kq,
    // kq_soft and kqv as real tensors — the byte-reproducible path, and the
    // one the receipts identity depends on: kq_soft IS the lens tap's read
    // surface (architecture.md §1, §11). Flash replaces that whole chain with
    // one ggml_flash_attn_ext, which keeps the softmax in registers and never
    // writes it out. Faster, and NOT byte-identical (different accumulation
    // order), and it makes attention_taps impossible to honor — hence the
    // fail-loud pairing check in is_attn_impl_coherent().
    enum class AttnImpl { Materialized, Flash };

    bool             slice_prefill_head = true;
    bool             output_hidden      = false;
    std::vector<int> attention_taps;
    KvWriteMode      kv_write_mode      = KvWriteMode::Cpy;
    AttnImpl         attn_impl          = AttnImpl::Materialized;

    // PREFILL's attention implementation, independent of decode's. Default
    // Materialized = today's path on every recipe. --flash-attn sets BOTH (it
    // has always meant "flash everywhere"); the lens sets this one alone.
    //
    // Phase-scoping is sound because flash and materialized differ only in HOW
    // the attention output is reduced, not in what is written to the KV cache
    // — K and V are cached before either path runs. What it is NOT is
    // byte-inert: the attention output feeds the residual stream, so a flash
    // prefill changes every later layer's hidden state and therefore the K/V
    // those layers write. A pass with flash prefill and materialized decode is
    // a THIRD numerical configuration, not a mix of two known ones, which is
    // why it is scored by the drift gate (BANDDRIFT arm `flash`) rather than
    // assumed equivalent.
    AttnImpl         prefill_attn_impl  = AttnImpl::Materialized;

    // PREFILL-only. -1 (default) = build every layer, today's behaviour.
    // Otherwise the physical layer index (0-based, inclusive) to stop the
    // layer loop after. Causality is what makes this exact rather than an
    // approximation: an attention layer cannot depend on a layer above it, so
    // a prefill truncated after max(citation_layer, coverage_layer) produces
    // IDENTICAL tapped rows to the untruncated pass, cheaper because the
    // omitted layers' matmuls (and any MoE/DeltaNet dispatch in them) never
    // run. Decode graphs ignore this field entirely — verify (the only caller)
    // never decodes. See effective_layer_count() and
    // docs/plan-lens-server-shape.md §3.4.
    int truncate_after_layer = -1;

    // The layer count a prefill graph should actually build, given the full
    // stack depth `full`. truncate_after_layer < 0 ⇒ `full` unchanged (every
    // recipe's default path); otherwise the smaller of `full` and
    // truncate_after_layer + 1 (so truncate_after_layer == 0 means "build just
    // layer 0"). Every recipe's build_prefill_graph bounds BOTH its mask-
    // registration loop and its layer-body loop with this — registering a
    // typed input for a layer whose node was never built would fail loud at
    // set-input time (the tensor it looks up by name would not exist), so the
    // two loops must agree.
    uint32_t effective_layer_count(uint32_t full) const {
        if (truncate_after_layer < 0) return full;
        const uint32_t cut = static_cast<uint32_t>(truncate_after_layer) + 1;
        return cut < full ? cut : full;
    }

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
            && kv_write_mode == KvWriteMode::Cpy && decode_kv_bucket == 0
            && attn_impl == AttnImpl::Materialized
            && prefill_attn_impl == AttnImpl::Materialized
            && truncate_after_layer < 0;
    }

    // Flash attention and the lens tap cannot both be armed: Flash never
    // materializes kq_soft, so a tap on it would read a node that does not
    // exist. Callers check this and fail loud rather than silently dropping
    // one of the two — see the --flash-attn / --attention-lens pairing.
    //
    // DECODE only, and deliberately not mirrored for prefill. The one caller
    // that taps prefill (teacher-forced verification) sets prefill_attn_impl
    // itself for its tapped pass, and if that line were ever lost,
    // mark_attention_taps already fails loud on the missing kq_soft node — a
    // structural guard beats a predicate nothing calls.
    bool is_attn_impl_coherent() const {
        return attn_impl == AttnImpl::Materialized || attention_taps.empty();
    }
};
