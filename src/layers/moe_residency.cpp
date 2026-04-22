#include "moe_residency.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

MoEResidencyTracker::MoEResidencyTracker(int n_experts, int resident_k)
    : n_experts_(n_experts)
    , resident_k_(std::min(resident_k, n_experts))
    , freq_(static_cast<size_t>(n_experts), 0)
    , cold_count_(0)
{
    if (n_experts <= 0)
        throw std::runtime_error(
            "moe_residency: n_experts must be > 0; got " + std::to_string(n_experts));
    if (resident_k <= 0)
        throw std::runtime_error(
            "moe_residency: resident_k must be > 0; got " + std::to_string(resident_k));
}

void MoEResidencyTracker::record_dispatch(int expert_id) {
    if (expert_id < 0 || expert_id >= n_experts_) {
        throw std::runtime_error(
            "moe_residency: record_dispatch: expert_id=" + std::to_string(expert_id)
            + " out of range [0, " + std::to_string(n_experts_) + ")");
    }

    // Check residency BEFORE frequency update so the cold counter reflects
    // whether this expert was pinned at dispatch time.
    if (!is_resident(expert_id))
        ++cold_count_;

    freq_[static_cast<size_t>(expert_id)]++;
    refresh_resident_set();
}

bool MoEResidencyTracker::is_resident(int expert_id) const {
    for (int id : resident_set_) {
        if (id == expert_id) return true;
    }
    return false;
}

uint64_t MoEResidencyTracker::cold_dispatch_count() const {
    return cold_count_;
}

MoEMetrics MoEResidencyTracker::snapshot_metrics() const {
    MoEMetrics m;
    m.cold_dispatch_count = cold_count_;
    return m;
}

const std::vector<int>& MoEResidencyTracker::resident_set() const {
    return resident_set_;
}

const std::vector<uint64_t>& MoEResidencyTracker::frequencies() const {
    return freq_;
}

void MoEResidencyTracker::reset() {
    std::fill(freq_.begin(), freq_.end(), 0);
    resident_set_.clear();
    cold_count_ = 0;
}

void MoEResidencyTracker::refresh_resident_set() {
    std::vector<int> order(static_cast<size_t>(n_experts_));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + resident_k_, order.end(),
        [&](int a, int b) {
            return freq_[static_cast<size_t>(a)] > freq_[static_cast<size_t>(b)];
        });

    resident_set_.clear();
    for (int i = 0; i < resident_k_; ++i) {
        if (freq_[static_cast<size_t>(order[i])] > 0)
            resident_set_.push_back(order[i]);
    }
}
