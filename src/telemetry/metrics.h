#pragma once
// metrics.h — plain-C++ observability surface for the inference engine.
//
// Responsibility: aggregate metric counters into snapshottable plain-C++ structs.
//   No transport or framework dependency — server layer hooks exporters as needed.
// Contents: MoEMetrics (first citizen, Phase 3 — introduced with MoE residency).
// Per-layer timing, KV cache occupancy, and residency hit rate are added here
//   as they become relevant in later phases.
// Reference: docs/plan-modular-layer-arch-impl.md § Observability bus.

#include <cstdint>

// Snapshot of MoE dispatch telemetry for one model instance.
// Populated by MoEResidencyTracker::snapshot_metrics().
struct MoEMetrics {
    uint64_t cold_dispatch_count = 0;  // expert dispatches that bypassed the resident set
};
