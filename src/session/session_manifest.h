#pragma once
// session_manifest.h — the snapshot coordinator: an ordered list of borrowed
// SnapshotSection pointers plus the compat header. It is a *manifest of
// references to existing state*, not a god-object that owns copies — never move
// state into this class (docs/plan-session-snapshot.md non-goals).
//
// capture() writes header + each section, framed so restore() can validate and
// bound it. restore() validates the header against the receiver's expected one,
// then dispatches each framed section to its registered reader. Both fail loud.
//
// On-stream layout:
//   [CompatHeader]
//   [u32 section_count]
//   repeated section_count times:
//     [u32 SectionId][u64 payload_len][payload bytes]

#include <vector>

#include "compat_header.h"
#include "snapshot_section.h"

namespace qinf::session {

class SnapshotWriter;
class SnapshotReader;

class SessionManifest {
public:
    // Registers a section. The manifest borrows the pointer; the section must
    // outlive every capture()/restore() call. Order is significant: capture and
    // restore walk the list in registration order.
    void add(SnapshotSection* section);

    // Serializes header + all registered sections into w.
    void capture(SnapshotWriter& w, const CompatHeader& header) const;

    // Reads and validates the header against `expected` (fail-loud), then
    // dispatches each framed section to the matching registered section.
    // Returns the header that was read.
    CompatHeader restore(SnapshotReader& r, const CompatHeader& expected);

    size_t section_count() const { return sections_.size(); }

private:
    std::vector<SnapshotSection*> sections_;  // borrowed, never owned
};

}  // namespace qinf::session
