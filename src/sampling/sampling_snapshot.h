#pragma once
// sampling_snapshot.h — L1 SnapshotSections for sampling-owned control state.
//
// Two Control-lane sections, co-located with the modules whose state they
// serialize (Sampler RNG, GrammarVocab cursor). Each borrows a reference to the
// live object — it owns no copy. The owning consumer registers these with a
// SessionManifest alongside a TokenSequenceSection (src/state/).
//
// GrammarStateSection only restores the *cursor*; the GrammarVocab it borrows
// must already have been re-parsed from the same GBNF (see GrammarVocab::
// read_cursor). The section deliberately does not serialize the grammar
// definition — that is part of the session's request config, not its state.

#include "grammar_vocab.h"
#include "sampling.h"
#include "section_ids.h"
#include "snapshot_section.h"

namespace qwenium {

// Serializes a sampler's PRNG state (no-op for deterministic samplers).
class SamplerStateSection : public qinf::session::SnapshotSection {
public:
    explicit SamplerStateSection(Sampler& sampler) : sampler_(sampler) {}

    qinf::session::SectionId id() const override {
        return qinf::session::kSamplerStateSectionId;
    }
    qinf::session::StateLane lane() const override {
        return qinf::session::StateLane::Control;
    }
    void write(qinf::session::SnapshotWriter& w) const override;
    void read(qinf::session::SnapshotReader& r) override;

private:
    Sampler& sampler_;
};

// Serializes a grammar's cursor (impl_->stacks). The borrowed GrammarVocab must
// be re-parsed from the same GBNF before read().
class GrammarStateSection : public qinf::session::SnapshotSection {
public:
    explicit GrammarStateSection(GrammarVocab& grammar) : grammar_(grammar) {}

    qinf::session::SectionId id() const override {
        return qinf::session::kGrammarCursorSectionId;
    }
    qinf::session::StateLane lane() const override {
        return qinf::session::StateLane::Control;
    }
    void write(qinf::session::SnapshotWriter& w) const override {
        grammar_.write_cursor(w);
    }
    void read(qinf::session::SnapshotReader& r) override {
        grammar_.read_cursor(r);
    }

private:
    GrammarVocab& grammar_;
};

}  // namespace qwenium
