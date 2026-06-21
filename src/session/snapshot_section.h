#pragma once
// snapshot_section.h — the contract each stateful module implements to
// serialize *its own* bytes into a session snapshot.
//
// A SnapshotSection borrows a reference to live module state; it never owns a
// copy. The SessionManifest (session_manifest.h) holds an ordered list of these
// and frames each one in the stream. Per-section serialization logic lives
// co-located in the owning module (e.g. SamplerStateSection under src/sampling/),
// never here — this header is only the interface.
//
// StateLane is what structurally prevents the append-only KV lane and the
// overwrite-only recurrent lane from ever being unified behind one update model
// (docs/plan-session-snapshot.md, CLAUDE.md ggml constraints): the manifest
// iterates lanes, it never merges them.

#include <cstdint>

namespace qinf::session {

class SnapshotWriter;
class SnapshotReader;

// Stable per-section tag. Each section type owns a distinct constant so the
// manifest can validate the on-disk id against the registered section.
using SectionId = uint32_t;

enum class StateLane {
    Control,             // scalar/discrete: tokens, sampler RNG, grammar cursor
    AppendKV,            // append-only: KV cache K/V tensors
    OverwriteRecurrent,  // overwrite: SSM / DeltaNet recurrent state
};

struct SnapshotSection {
    virtual SectionId id() const = 0;
    virtual StateLane lane() const = 0;
    virtual void write(SnapshotWriter&) const = 0;
    virtual void read(SnapshotReader&) = 0;
    virtual ~SnapshotSection() = default;
};

}  // namespace qinf::session
