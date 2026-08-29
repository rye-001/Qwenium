#pragma once
// slot_snapshot.h — model-aware L2 snapshot host helpers, shared by the CLI and
// the HTTP server (docs/plan-session-snapshot.md). Captures / restores ONE
// forward-pass slot's KV (+ recurrent) to/from a SessionManifest blob, and
// builds the node-identity CompatHeader that gates a blob fail-loud on a
// model / quant / kernel-path mismatch.
//
// This is the model-aware layer ABOVE PrefixLibrary (which owns only the
// key->file mapping + the header gate). The KEY choice (system-prompt hash vs
// image-extended) and the HIT/MISS prefill action stay in the host; these
// helpers own the manifest/section plumbing so it lives in ONE place instead of
// being copied across chat.cpp, http_server.cpp, and the roundtrip harnesses.
//
// Compiled INTO consumers (cli / server / tests), NOT the `core` static lib: it
// depends on ForwardPassBase (qinf-models), which sits above core. Same
// arrangement as multimodal_prefill.cpp / decode_step.cpp.

#include <cstdint>
#include <vector>

#include "session/compat_header.h"

class ForwardPassBase;
class simple_kv_cache;
struct ModelMetadata;

namespace qinf::snapshot {

// build_path_tag over ALL of a recipe's KV caches (Gemma4 has two: global +
// sliding). A fold that returns the single path_tag unchanged for one cache, so
// single-cache recipes (Qwen35, Gemma3) are byte-identical to a single-cache
// header.
uint64_t combined_path_tag(const std::vector<simple_kv_cache*>& caches);

// Node-identity header: arch + shape + weights_hash + the live KV kernel-path
// tag(s). A blob from a different model / quant / backend mismatches and is
// refused fail-loud. `caches` = fp.snapshot_kv_caches().
qinf::session::CompatHeader make_snapshot_header(
    const ModelMetadata& meta, const std::vector<simple_kv_cache*>& caches);

// Capture slot `slot`'s KV (one AppendKV section per cache, in
// snapshot_kv_caches() order) plus the authoritative OverwriteRecurrent state
// (when the recipe has one), framed under `header`. Returns the manifest blob.
std::vector<uint8_t> capture_slot(ForwardPassBase& fp, uint32_t slot,
                                  const qinf::session::CompatHeader& header);

// Restore slot `slot`'s KV (+ recurrent) from `blob`, validating its embedded
// header against `expected` (throws fail-loud naming the first mismatched
// field). Sections are built in the SAME order as capture_slot and matched
// positionally by the manifest. After this the slot cursor is at the captured
// span length; the caller resumes/prefills from there with NO feed_tokens /
// prefill of the captured span.
void restore_slot(ForwardPassBase& fp, uint32_t slot,
                  const std::vector<uint8_t>& blob,
                  const qinf::session::CompatHeader& expected);

}  // namespace qinf::snapshot
