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
