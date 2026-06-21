#pragma once
// token_sequence_section.h — L1 SnapshotSection for the per-sequence running
// token vector (Control lane).
//
// The token id sequence is per-sequence Control state, co-located here alongside
// the other per-sequence lanes (AppendKV = kv_cache_simple, OverwriteRecurrent =
// recurrent/deltanet). It borrows a reference to the live vector the consumer
// owns (server Slot::context_tokens, or the CLI runner's token vector) plus a
// generation-start marker (prompt length) — it owns no copy.
//
// last_tokens (repetition-penalty history) is a window over this same vector, so
// it needs no separate section. On L1 restore this span is replayed through
// feed_tokens to rebuild KV + recurrent; the section itself stores only the ids.

#include <cstdint>
#include <vector>

#include "section_ids.h"
#include "snapshot_section.h"

namespace qinf::state {

class TokenSequenceSection : public qinf::session::SnapshotSection {
public:
    // tokens: the full running context (prompt + generated).
    // gen_start: index of the first *generated* token (== prompt length).
    TokenSequenceSection(std::vector<int32_t>& tokens, uint32_t& gen_start)
        : tokens_(tokens), gen_start_(gen_start) {}

    qinf::session::SectionId id() const override {
        return qinf::session::kTokenSequenceSectionId;
    }
    qinf::session::StateLane lane() const override {
        return qinf::session::StateLane::Control;
    }
    void write(qinf::session::SnapshotWriter& w) const override;
    void read(qinf::session::SnapshotReader& r) override;

private:
    std::vector<int32_t>& tokens_;
    uint32_t&             gen_start_;
};

}  // namespace qinf::state
