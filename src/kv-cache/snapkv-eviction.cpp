#include "snapkv-eviction.h"
#include "simple-kv-cache.h"
#include "compressed_kv_store.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <cstdio>

std::vector<std::vector<float>> pool_attention_to_importance(
    const float* attn_data,
    uint32_t n_kv, uint32_t n_tokens, uint32_t n_heads,
    uint32_t obs_window)
{
    // Clamp window to available tokens
    const uint32_t W = std::min(obs_window, n_tokens);
    const uint32_t t_start = n_tokens - W;

    std::vector<std::vector<float>> importance(n_heads, std::vector<float>(n_kv, 0.0f));

    for (uint32_t h = 0; h < n_heads; ++h) {
        const float* head_base = attn_data + static_cast<size_t>(h) * n_tokens * n_kv;
        for (uint32_t t = t_start; t < n_tokens; ++t) {
            const float* row = head_base + static_cast<size_t>(t) * n_kv;
            for (uint32_t j = 0; j < n_kv; ++j) {
                importance[h][j] += row[j];
            }
        }
    }

    return importance;
}

SnapKVResult compute_eviction_mask(
    const std::vector<std::vector<std::vector<float>>>& scores,
    uint32_t budget,
    uint32_t obs_window)
{
    SnapKVResult result;
    if (scores.empty() || scores[0].empty() || scores[0][0].empty()) {
        result.original_length = 0;
        return result;
    }

    const uint32_t n_layers = scores.size();
    const uint32_t n_positions = scores[0][0].size();
    result.original_length = n_positions;
    result.retained_positions.resize(n_layers);

    // Window = last obs_window positions, always retained
    const uint32_t window_start = (n_positions > obs_window)
                                      ? (n_positions - obs_window)
                                      : 0;

    for (uint32_t l = 0; l < n_layers; ++l) {
        const uint32_t n_heads = scores[l].size();
        std::set<uint32_t> retained_set;

        // Always include the observation window
        for (uint32_t p = window_start; p < n_positions; ++p) {
            retained_set.insert(p);
        }

        // If budget >= n_positions, keep everything
        if (budget >= n_positions) {
            result.retained_positions[l].resize(n_positions);
            std::iota(result.retained_positions[l].begin(),
                      result.retained_positions[l].end(), 0u);
            continue;
        }

        // For each head: find top-B positions (excluding window, which is already kept)
        // We select from positions [0, window_start) only
        if (budget > 0 && window_start > 0) {
            // Reusable index buffer for partial sorting
            std::vector<uint32_t> indices(window_start);
            std::iota(indices.begin(), indices.end(), 0u);

            for (uint32_t h = 0; h < n_heads; ++h) {
                const auto& head_scores = scores[l][h];

                // Partial sort to find top-budget positions
                uint32_t k = std::min(budget, window_start);
                std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
                    [&head_scores](uint32_t a, uint32_t b) {
                        return head_scores[a] > head_scores[b];
                    });

                for (uint32_t i = 0; i < k; ++i) {
                    retained_set.insert(indices[i]);
                }

                // Restore index order for next head
                std::iota(indices.begin(), indices.end(), 0u);
            }
        }

        // Convert set to sorted vector
        result.retained_positions[l].assign(retained_set.begin(), retained_set.end());
    }

    return result;
}

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
    std::function<void(uint32_t)> invalidate_watermarks)
{
    char name[32];
    snprintf(name, sizeof(name), "kq_soft.%d", scoring_layer);
    ggml_tensor* kq_soft = ggml_graph_get_tensor(gf, name);
    if (!kq_soft) return;

    // kq_soft shape: [n_kv, n_tokens, n_heads, 1]
    const uint32_t n_kv    = kq_soft->ne[0];
    const uint32_t n_heads = kq_soft->ne[2];

    // Read attention weights to CPU
    std::vector<float> attn_data(static_cast<size_t>(n_kv) * n_tokens * n_heads);
    ggml_backend_tensor_get(kq_soft, attn_data.data(), 0,
        attn_data.size() * sizeof(float));

    // Pool to importance scores [n_heads][n_kv]
    auto head_scores = pool_attention_to_importance(
        attn_data.data(), n_kv, n_tokens, n_heads, snapkv_window);

    // Wrap into [1 layer][n_heads][n_kv] — all layers get the same mask
    std::vector<std::vector<std::vector<float>>> scores_all(1);
    scores_all[0] = std::move(head_scores);

    SnapKVResult eviction = compute_eviction_mask(scores_all, snapkv_budget, snapkv_window);
    if (eviction.retained_positions.empty()) return;
    const auto& retained = eviction.retained_positions[0];

    // Compact all layers using the same retained positions (uniform eviction)
    if (tq_store) {
        for (uint32_t l = 0; l < n_layers; ++l)
            tq_store->compact(l, slot_idx, retained);
        tq_store->set_pos(slot_idx, retained.size());
        if (invalidate_watermarks) invalidate_watermarks(slot_idx);
    } else if (kv_cache) {
        kv_cache->compact(slot_idx, retained);
    }

    printf("SnapKV: evicted %u → %zu KV positions (budget=%u, window=%u)\n",
           n_kv, retained.size(), snapkv_budget, snapkv_window);
}
