#pragma once
// moe_residency.h — top-K expert residency bookkeeping for MoE layers.
//
// Responsibility: track per-expert dispatch frequencies, maintain a "resident"
//   set of the top-K most frequently dispatched experts, and count cold-path
//   dispatches (experts outside the resident set). Pure C++ bookkeeping — no
//   Metal residency APIs (those are Phase 4, PR 4.2).
// Public surface:
//   record_dispatch(expert_id) — record one expert dispatch; checks residency
//     BEFORE updating frequencies, increments cold counter if not resident,
//     then refreshes the resident set.
//   is_resident(expert_id)     — true if expert is currently in the top-K set.
//   cold_dispatch_count()      — total non-resident dispatches since construction
//     or last reset().
//   snapshot_metrics()         — copy current counters into a MoEMetrics struct.
//   resident_set()             — current resident expert ids (descending frequency).
//   frequencies()              — per-expert dispatch counts.
//   reset()                    — zero all state.
// State owned: frequency vector, resident set, cold counter — all on CPU.
// Invariants:
//   - Resident set is refreshed eagerly on every record_dispatch call.
//   - Cold counter increments when the dispatched expert was NOT in the resident
//     set AT THE MOMENT OF DISPATCH (before the frequency update).
//   - resident_k clamped to n_experts: all experts become resident when k >= n.
//   - record_dispatch with expert_id outside [0, n_experts) throws std::runtime_error
//     naming expert_id and the valid range.
// Reference: docs/plan-modular-layer-arch-impl.md § PR 3.M.4.
// Unit test: tests/unit/test_moe_residency.cpp

#include "../telemetry/metrics.h"

#include <cstdint>
#include <vector>

class MoEResidencyTracker {
public:
    // n_experts: total expert count; must match MoELayer::Hparams::n_experts.
    // resident_k: size of the pinned resident set (clamped to n_experts internally).
    MoEResidencyTracker(int n_experts, int resident_k);

    // Record one expert dispatch. Cold counter increments if expert was not resident.
    void record_dispatch(int expert_id);

    // True if expert_id is in the current resident set.
    bool is_resident(int expert_id) const;

    // Total cold dispatches since construction or last reset().
    uint64_t cold_dispatch_count() const;

    // Snapshot current counters.
    MoEMetrics snapshot_metrics() const;

    // Current resident expert ids, ordered by descending frequency.
    const std::vector<int>& resident_set() const;

    // Per-expert dispatch counts indexed by expert_id.
    const std::vector<uint64_t>& frequencies() const;

    // Zero all frequencies, clear resident set, reset cold counter.
    void reset();

private:
    int                   n_experts_;
    int                   resident_k_;
    std::vector<uint64_t> freq_;
    std::vector<int>      resident_set_;
    uint64_t              cold_count_;

    void refresh_resident_set();
};
