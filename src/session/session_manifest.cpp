#include "session_manifest.h"

#include <stdexcept>
#include <string>

#include "snapshot_io.h"

namespace qinf::session {

void SessionManifest::add(SnapshotSection* section) {
    if (section == nullptr) {
        throw std::runtime_error(
            "session_manifest: add() expected a non-null section, got nullptr");
    }
    sections_.push_back(section);
}

void SessionManifest::capture(SnapshotWriter& w, const CompatHeader& header) const {
    header.write(w);
    w.put_u32(static_cast<uint32_t>(sections_.size()));
    for (const SnapshotSection* s : sections_) {
        w.put_u32(s->id());
        // Frame the payload with its byte length so restore can bound it and
        // detect writer/reader drift. Serialize into a scratch writer first to
        // learn the length, then splice it in length-prefixed.
        SnapshotWriter payload;
        s->write(payload);
        w.put_bytes(payload.buffer().data(), payload.buffer().size());
    }
}

CompatHeader SessionManifest::restore(SnapshotReader& r, const CompatHeader& expected) {
    CompatHeader header = CompatHeader::read(r);
    header.check(expected);

    uint32_t count = r.get_u32();
    if (count != sections_.size()) {
        throw std::runtime_error(
            std::string("session_manifest: section_count expected ") +
            std::to_string(sections_.size()) + ", got " + std::to_string(count));
    }

    for (SnapshotSection* s : sections_) {
        uint32_t id = r.get_u32();
        if (id != s->id()) {
            throw std::runtime_error(
                std::string("session_manifest: section id expected ") +
                std::to_string(s->id()) + ", got " + std::to_string(id));
        }
        std::vector<uint8_t> payload;
        r.get_bytes(payload);
        SnapshotReader sub(payload);
        s->read(sub);
        if (sub.remaining() != 0) {
            throw std::runtime_error(
                std::string("session_manifest: section ") + std::to_string(id) +
                " payload bytes consumed expected " +
                std::to_string(payload.size()) + ", got " +
                std::to_string(payload.size() - sub.remaining()));
        }
    }
    return header;
}

}  // namespace qinf::session
