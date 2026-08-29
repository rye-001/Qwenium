#include "slot_snapshot.h"

#include <stdexcept>

#include <functional>
#include <memory>
#include <string>

#include "engine/model.h"                // ModelMetadata
#include "models/forward_pass_base.h"   // ForwardPassBase (+ simple_kv_cache via include)
#include "state/deltanet_state.h"       // DeltaNetState, DeltaNetStateSection
#include "state/kv_cache_simple.h"      // simple_kv_cache, KvCacheSection
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
}  // namespace

std::vector<uint8_t> capture_slot(ForwardPassBase& fp, uint32_t slot,
                                  const qinf::session::CompatHeader& header) {
    // A snapshot records a ROW COUNT and no rope coordinate. That was lossless
    // while the two were the same number, but an M-RoPE image span writes nx*ny
    // rows while advancing the position by only max(nx, ny), so a slot that has
    // hosted an image cannot be round-tripped: restoring it would resume at the
    // row count and every later token would rotate at the wrong position.
    // Refuse rather than persist a blob that decodes wrong
    // (docs/plan-qwen35-vision-impl.md §4 decision 3 — VL sessions are declared
    // non-snapshottable in v1; carrying the coordinate needs a header bump).
    if (fp.has_rope_divergence(slot))
        throw std::runtime_error(
            "capture_slot: slot '" + std::to_string(slot) +
            "': expected KV rows == rope positions (a snapshottable slot), got: "
            "a slot containing an image span, where they diverge. VL sessions "
            "are not snapshottable in v1 — disable the prefix/snapshot cache for "
            "image turns.");

    qinf::session::SessionManifest man;
    std::vector<std::unique_ptr<KvCacheSection>> kv_secs;
    std::unique_ptr<DeltaNetStateSection> dn_sec;
    add_sections(man, kv_secs, dn_sec, fp, slot);
    qinf::session::SnapshotWriter w;
    man.capture(w, header);
    return w.buffer();
}

void restore_slot(ForwardPassBase& fp, uint32_t slot,
                  const std::vector<uint8_t>& blob,
                  const qinf::session::CompatHeader& expected) {
    qinf::session::SessionManifest man;
    std::vector<std::unique_ptr<KvCacheSection>> kv_secs;
    std::unique_ptr<DeltaNetStateSection> dn_sec;
    add_sections(man, kv_secs, dn_sec, fp, slot);
    qinf::session::SnapshotReader r(blob);
    man.restore(r, expected);
    // The blob carries no rope coordinate, and capture_slot guarantees it never
    // held a diverged span — so the restored slot has rows == positions. Drop
    // any record left over from what this slot held BEFORE the restore, which
    // would otherwise be applied to unrelated history.
    fp.reset_rope_pos(slot);
}

}  // namespace qinf::snapshot
