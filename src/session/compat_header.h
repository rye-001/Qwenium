#pragma once
// compat_header.h — the version/compatibility preamble of every session
// snapshot. A snapshot is only restorable when the receiver's model weights,
// Qwenium version, quant, and kernel path match the producer's; the header
// records exactly those discriminants so a mismatch is refused fail-loud
// instead of silently re-prefilled (docs/plan-session-snapshot.md).
//
// Phase 0 carries these as plain fields; populating them from a live Model
// (src/engine/model.h ModelMetadata) is deferred to the service phases. The
// struct intentionally has no dependency on the model so it stays unit-testable
// in isolation.

#include <cstdint>

namespace qinf::session {

class SnapshotWriter;
class SnapshotReader;

// "QSNP" little-endian. Bumps to format_version are breaking by default.
constexpr uint32_t kSnapshotMagic = 0x504E5351u;  // 'Q''S''N''P'
constexpr uint32_t kSnapshotFormatVersion = 1u;

struct CompatHeader {
    uint32_t magic = kSnapshotMagic;
    uint32_t format_version = kSnapshotFormatVersion;

    // Producer build/version stamp; receiver must match exactly.
    uint64_t qwenium_version = 0;
    // Hash of the model weights (GGUF). 0 = unset (Phase 0 placeholder).
    uint64_t weights_hash = 0;
    // ggml file_type / quant scheme enum.
    uint32_t quant_scheme = 0;
    // Hashed architecture string (ModelMetadata::architecture).
    uint32_t arch_id = 0;

    // Shape discriminants (ModelMetadata).
    uint32_t block_count = 0;
    uint32_t embedding_length = 0;
    uint32_t attention_head_count = 0;
    uint32_t attention_head_count_kv = 0;
    uint32_t attention_key_length = 0;
    uint32_t vocab_size = 0;

    // Prefill batch-shape / kernel path. Load-bearing for L2 (a KV blob is only
    // byte-safe to resume under an identical build path); inert for L1.
    uint64_t build_path_tag = 0;

    void write(SnapshotWriter&) const;
    static CompatHeader read(SnapshotReader&);

    // Fail-loud field-by-field compatibility check against the expected header
    // (the receiver's live model). Throws on the first mismatch, naming the
    // field, the expected value, and the actual value, in that order.
    void check(const CompatHeader& expected) const;
};

}  // namespace qinf::session
