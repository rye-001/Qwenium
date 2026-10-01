#include "slot_snapshot.h"

#include <stdexcept>

#include <functional>
#include <memory>
#include <string>

#include "engine/model.h"                // ModelMetadata
#include "models/forward_pass_base.h"   // ForwardPassBase (+ simple_kv_cache via include)
#include "state/deltanet_state.h"       // DeltaNetState, DeltaNetStateSection
#include "state/kv_cache_simple.h"      // simple_kv_cache, KvCacheSection
#include "session/section_ids.h"
#include "session/session_manifest.h"
#include "session/snapshot_io.h"

namespace qinf::snapshot {

uint64_t combined_path_tag(const std::vector<simple_kv_cache*>& caches) {
    uint64_t tag = 0;
    bool first = true;
    for (const simple_kv_cache* c : caches) {
        const uint64_t pt = c->path_tag();
        tag = first ? pt : (tag * 1099511628211ull) ^ pt;
        first = false;
    }
    return tag;
}

qinf::session::CompatHeader make_snapshot_header(
    const ModelMetadata& m, const std::vector<simple_kv_cache*>& caches) {
    qinf::session::CompatHeader h;
    h.arch_id = static_cast<uint32_t>(std::hash<std::string>{}(m.architecture));
    h.weights_hash = m.weights_hash;
    h.block_count = m.block_count;
    h.embedding_length = m.embedding_length;
    h.attention_head_count = m.attention_head_count;
    h.attention_head_count_kv = m.attention_head_count_kv;
    h.attention_key_length = m.attention_key_length;
    h.vocab_size = m.vocab_size;
    h.build_path_tag = combined_path_tag(caches);
    return h;
}

namespace {
// The blob's sections in the fixed capture/restore order: one AppendKV section
// per KV cache (Gemma4 → 2, others → 1) in snapshot_kv_caches() order, plus the
// authoritative OverwriteRecurrent state for `slot` when the recipe has one. The
// two KvCacheSections share the section id and are matched POSITIONALLY by the
// manifest (capture and restore use this same order). The owning containers must
// outlive the manifest use.
void add_sections(qinf::session::SessionManifest& man,
                  std::vector<std::unique_ptr<KvCacheSection>>& kv_secs,
                  std::unique_ptr<DeltaNetStateSection>& dn_sec,
                  ForwardPassBase& fp, uint32_t slot) {
    for (simple_kv_cache* kv : fp.snapshot_kv_caches()) {
        kv_secs.push_back(std::make_unique<KvCacheSection>(*kv, slot));
        man.add(kv_secs.back().get());
    }
    if (fp.snapshot_recurrent()) {
        dn_sec = std::make_unique<DeltaNetStateSection>(*fp.snapshot_recurrent(), slot);
        man.add(dn_sec.get());
    }
}

// The slot's rope coordinate: ForwardPassBase's rows-vs-positions record
// (delta, rows_after). It lives here rather than beside the record because the
// record's owner (models/) does not link session/; it reaches the record only
// through ForwardPassBase's public rope_record / set_rope_record. Written only
// when the slot has diverged (an M-RoPE image span), and always as the LAST
// section, so text blobs and Gemma image blobs keep their exact bytes.
// docs/plan-image-verdict.md §4.
class RopeCoordinateSection : public qinf::session::SnapshotSection {
public:
    RopeCoordinateSection(ForwardPassBase& fp, uint32_t slot) : fp_(fp), slot_(slot) {}
    qinf::session::SectionId id() const override { return qinf::session::kRopeCoordinateSectionId; }
    qinf::session::StateLane lane() const override { return qinf::session::StateLane::Control; }
    void write(qinf::session::SnapshotWriter& w) const override {
        int32_t delta = 0, rows_after = 0;
        if (!fp_.rope_record(slot_, delta, rows_after))
            throw std::runtime_error("RopeCoordinateSection: slot '" + std::to_string(slot_) +
                                     "' expected a rope divergence to write, got none");
        w.put_u32(static_cast<uint32_t>(delta));
        w.put_u32(static_cast<uint32_t>(rows_after));
    }
    // Runs after the KV / recurrent sections (it is last), so the slot's rows
    // are already restored and set_rope_record can validate against them.
    void read(qinf::session::SnapshotReader& r) override {
        const int32_t delta = static_cast<int32_t>(r.get_u32());
        const int32_t rows_after = static_cast<int32_t>(r.get_u32());
        fp_.set_rope_record(slot_, delta, rows_after);
    }

private:
    ForwardPassBase& fp_;
    uint32_t slot_;
};

// How many sections the blob says it holds — read before restore registers its
// sections, so restore can tell a blob with the RPOS section from one without.
uint32_t blob_section_count(const std::vector<uint8_t>& blob) {
    qinf::session::SnapshotReader peek(blob);
    (void)qinf::session::CompatHeader::read(peek);
    return peek.get_u32();
}
}  // namespace

std::vector<uint8_t> capture_slot(ForwardPassBase& fp, uint32_t slot,
                                  const qinf::session::CompatHeader& header) {
    // The KV sections record a ROW COUNT. That is the whole position story while
    // rows and rope positions are the same number; an M-RoPE image span writes
    // nx*ny rows but advances the position by only max(nx, ny), so such a slot
    // also needs its rope coordinate — the RPOS section, appended last and only
    // then. (Until 2026-10-01 such a slot was refused here: VL sessions were
    // non-snapshottable, docs/plan-qwen35-vision-impl.md §4 decision 3.)
    qinf::session::SessionManifest man;
    std::vector<std::unique_ptr<KvCacheSection>> kv_secs;
    std::unique_ptr<DeltaNetStateSection> dn_sec;
    add_sections(man, kv_secs, dn_sec, fp, slot);
    std::unique_ptr<RopeCoordinateSection> rope_sec;
    if (fp.has_rope_divergence(slot)) {
        rope_sec = std::make_unique<RopeCoordinateSection>(fp, slot);
        man.add(rope_sec.get());
    }
    qinf::session::SnapshotWriter w;
    man.capture(w, header);
    return w.buffer();
}

void restore_slot(ForwardPassBase& fp, uint32_t slot,
                  const std::vector<uint8_t>& blob,
                  const qinf::session::CompatHeader& expected) {
    // Drop any record left from what this slot held BEFORE the restore — it
    // belongs to unrelated history. A blob with an RPOS section re-installs the
    // captured one; a blob without it restores a slot with rows == positions.
    fp.reset_rope_pos(slot);
    qinf::session::SessionManifest man;
    std::vector<std::unique_ptr<KvCacheSection>> kv_secs;
    std::unique_ptr<DeltaNetStateSection> dn_sec;
    add_sections(man, kv_secs, dn_sec, fp, slot);
    std::unique_ptr<RopeCoordinateSection> rope_sec;
    if (blob_section_count(blob) == man.section_count() + 1) {
        rope_sec = std::make_unique<RopeCoordinateSection>(fp, slot);
        man.add(rope_sec.get());
    }
    // Any other count (and a wrong id in the extra slot) is refused fail-loud
    // by the manifest, naming expected and actual.
    qinf::session::SnapshotReader r(blob);
    man.restore(r, expected);
}

}  // namespace qinf::snapshot
