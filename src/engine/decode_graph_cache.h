#pragma once
// decode_graph_cache.h — the persistent decode graph (opt-in, --persistent-graph).
//
// Responsibility: build + allocate the single-token decode graph ONCE per
//   {active slots, n_kv bucket} key and reuse it across steps — on a key hit
//   only refill the typed inputs and recompute, skipping the ~12 ms galloc
//   replan that decode_breakdown localized (docs/note-decode-overhead-probes.md).
//   Delivered 1.32× decode on Qwen 3.6 (20 → 27 tok/s, stable ×3); the
//   standalone probe's ceiling for the same move was 1.28×. This is the payoff
//   of P1 (value-driven set_rows KV write) + P2 (bucketed n_kv) in
//   docs/plan-persistent-decode-graph.md.
// Runs on a DEDICATED ggml_backend_sched so prefill / feed_tokens / MTP graph
//   work on the main scheduler can never invalidate the persistent allocation
//   (same galloc-isolation rule as the MTP head and image-prefill scheds;
//   docs/server-image-multirequest-bug.md).
// Correctness: decode is token-stable but NOT byte-identical vs the exact-n_kv
//   rebuild path (bucketing re-blocks the reduction — plan §0.1), which is why
//   the whole path is opt-in.
// NOT for: sparse-head (grammar) steps, or recipes with
//   supports_persistent_decode()==false — those keep the per-step rebuild.
// Unit test: tests/unit/test_decode_graph_cache.cpp

#include <cstdint>
#include <vector>

#include "ggml-backend.h"

class Model;
class ForwardPassBase;
struct ggml_cgraph;

class DecodeGraphCache {
public:
    // Bucket width for the persistent n_kv (plan §2.2). One allocation stays
    // valid for a whole bucket of positions; re-alloc only at bucket crossings.
    static constexpr uint32_t kBucket = 256;

    // Creates and owns a dedicated scheduler over the model's backends. The
    // caller must set the forward pass to the persistent write mode + bucket
    // before first step() (enable_persistent_decode() does this).
    DecodeGraphCache(Model& model, ForwardPassBase* fp);
    ~DecodeGraphCache();

    DecodeGraphCache(const DecodeGraphCache&) = delete;
    DecodeGraphCache& operator=(const DecodeGraphCache&) = delete;

    // One single-token-per-slot decode step. Rebuilds (reset+build+alloc) when
    // the {slots, bucket} key changes or the cache was invalidated; otherwise
    // refills inputs + recomputes on the retained graph. Returns the graph so
    // the caller reads logits via fp->get_output_logits(gf). Does NOT advance
    // the KV position — the caller owns advance_cache, exactly as the rebuild
    // path does.
    ggml_cgraph* step(const std::vector<int32_t>& tokens,
                      const std::vector<uint32_t>& slots,
                      const std::vector<int32_t>& positions);

    // Force the next step() to rebuild. MUST be called after anything that
    // resets the forward pass's ggml_context out from under the retained graph
    // (prefill, feed_tokens, a sparse/bridge decode step) — the retained
    // graph's tensors are then dangling. Cheap (a bool); the rebuild is the
    // normal ~13 ms miss path.
    void invalidate() { has_cached_ = false; }

    ggml_backend_sched_t scheduler() const { return sched_; }

    // Diagnostics for the perf/A-B harness.
    uint64_t rebuilds() const { return rebuilds_; }
    uint64_t reuses()   const { return reuses_; }

private:
    ForwardPassBase*     fp_;
    ggml_backend_sched_t sched_ = nullptr;
    bool                 owns_sched_ = false;

    // The retained graph and the key it was built for.
    ggml_cgraph*          gf_ = nullptr;
    bool                  has_cached_ = false;
    std::vector<uint32_t> key_slots_;
    uint32_t              key_bucket_nkv_ = 0;

    uint64_t rebuilds_ = 0;
    uint64_t reuses_   = 0;
};

// Put a forward pass into the persistent-decode configuration: value-driven
// set_rows KV write + bucketed n_kv. The inverse (exact/cpy) is the default.
// Fail-loud if the recipe is not persistent-capable — the caller must gate on
// supports_persistent_decode() first; this refuses rather than silently
// producing an unbucketed graph the cache would mis-key.
void enable_persistent_decode(ForwardPassBase* fp);
