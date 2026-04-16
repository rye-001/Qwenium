#pragma once
// snapkv-eviction.h — SnapKV post-prefill KV eviction logic.
//
// Pure C++, no ggml dependency. Takes per-head importance scores and produces
// a per-layer list of retained KV positions (Approach B: uniform eviction —
// union of top-B across heads).
//
// Usage:
//   1. After prefill, compute attention scores (Phase 1).
//   2. Call compute_eviction_mask() to get retained positions.
//   3. Call cache compact() for each layer to reorganize the cache.

#include <cstdint>
#include <vector>
#include <functional>

struct SnapKVResult {
    // Per-layer: sorted list of retained KV positions (union across heads).
    std::vector<std::vector<uint32_t>> retained_positions;  // [n_layers][variable]
    uint32_t original_length;  // pre-eviction sequence length
};

// Compute the eviction mask from per-head importance scores.
//
// scores: [n_layers][n_heads][n_positions] — importance per KV position per head.
// budget: max positions to keep per head (top-B by importance).
// obs_window: observation window size (last W positions always retained).
//
// Returns per-layer sorted retained position lists.
SnapKVResult compute_eviction_mask(
    const std::vector<std::vector<std::vector<float>>>& scores,
    uint32_t budget,
    uint32_t obs_window);

// Pool raw attention weights from kq_soft into per-position importance scores.
//
// attn_data: flat F32 array from the kq_soft tensor, shape [n_kv, n_tokens, n_heads].
//            Layout: data[h * n_tokens * n_kv + t * n_kv + j].
// obs_window: number of trailing query tokens to sum over.
//
// Returns: [n_heads][n_kv] importance scores (sum of attention from obs window queries).
std::vector<std::vector<float>> pool_attention_to_importance(
    const float* attn_data,
    uint32_t n_kv, uint32_t n_tokens, uint32_t n_heads,
    uint32_t obs_window);

// Forward declarations (avoid pulling in ggml/cache headers here)
struct ggml_tensor;
struct ggml_cgraph;
class simple_kv_cache;
class CompressedKVStore;

// Read kq_soft from a computed graph, pool to importance scores, compute
// eviction mask, and compact the cache(s). Call after prefill graph compute.
//
// scoring_layer: physical layer index whose kq_soft was marked as output.
// kv_cache: non-TQ cache to compact (may be nullptr if TQ path).
// tq_store: TQ compressed store to compact (may be nullptr if non-TQ path).
// invalidate_watermarks: callback to reset scratch watermarks after compaction.
void apply_snapkv_from_graph(
    ggml_cgraph* gf,
    uint32_t scoring_layer,
    uint32_t n_tokens,
    uint32_t n_layers,
    uint32_t snapkv_budget,
    uint32_t snapkv_window,
    uint32_t slot_idx,
    simple_kv_cache* kv_cache,
    CompressedKVStore* tq_store,
    std::function<void(uint32_t)> invalidate_watermarks);
