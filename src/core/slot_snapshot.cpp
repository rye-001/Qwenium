#include "slot_snapshot.h"

#include <functional>
#include <memory>
#include <string>

#include "model.h"                      // ModelMetadata
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
}

}  // namespace qinf::snapshot
