#include "snapkv-eviction.h"

#include <algorithm>
#include <numeric>
#include <set>

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
