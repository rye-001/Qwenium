#include "compat_header.h"

#include <stdexcept>
#include <string>

#include "snapshot_io.h"

namespace qinf::session {

void CompatHeader::write(SnapshotWriter& w) const {
    w.put_u32(magic);
    w.put_u32(format_version);
    w.put_u64(qwenium_version);
    w.put_u64(weights_hash);
    w.put_u32(quant_scheme);
    w.put_u32(arch_id);
    w.put_u32(block_count);
    w.put_u32(embedding_length);
    w.put_u32(attention_head_count);
    w.put_u32(attention_head_count_kv);
    w.put_u32(attention_key_length);
    w.put_u32(vocab_size);
    w.put_u64(build_path_tag);
}

CompatHeader CompatHeader::read(SnapshotReader& r) {
    CompatHeader h;
    h.magic = r.get_u32();
    h.format_version = r.get_u32();
    h.qwenium_version = r.get_u64();
    h.weights_hash = r.get_u64();
    h.quant_scheme = r.get_u32();
    h.arch_id = r.get_u32();
    h.block_count = r.get_u32();
    h.embedding_length = r.get_u32();
    h.attention_head_count = r.get_u32();
    h.attention_head_count_kv = r.get_u32();
    h.attention_key_length = r.get_u32();
    h.vocab_size = r.get_u32();
    h.build_path_tag = r.get_u64();
    return h;
}

namespace {

void require_field(bool ok, const char* field, uint64_t expected, uint64_t actual) {
    if (!ok) {
        throw std::runtime_error(
            std::string("compat_header: field \"") + field + "\" expected " +
            std::to_string(expected) + ", got " + std::to_string(actual));
    }
}

}  // namespace

void CompatHeader::check(const CompatHeader& expected) const {
    require_field(magic == expected.magic, "magic", expected.magic, magic);
    require_field(format_version == expected.format_version, "format_version",
                  expected.format_version, format_version);
    require_field(qwenium_version == expected.qwenium_version, "qwenium_version",
                  expected.qwenium_version, qwenium_version);
    require_field(weights_hash == expected.weights_hash, "weights_hash",
                  expected.weights_hash, weights_hash);
    require_field(quant_scheme == expected.quant_scheme, "quant_scheme",
                  expected.quant_scheme, quant_scheme);
    require_field(arch_id == expected.arch_id, "arch_id", expected.arch_id, arch_id);
    require_field(block_count == expected.block_count, "block_count",
                  expected.block_count, block_count);
    require_field(embedding_length == expected.embedding_length, "embedding_length",
                  expected.embedding_length, embedding_length);
    require_field(attention_head_count == expected.attention_head_count,
                  "attention_head_count", expected.attention_head_count,
                  attention_head_count);
    require_field(attention_head_count_kv == expected.attention_head_count_kv,
                  "attention_head_count_kv", expected.attention_head_count_kv,
                  attention_head_count_kv);
    require_field(attention_key_length == expected.attention_key_length,
                  "attention_key_length", expected.attention_key_length,
                  attention_key_length);
    require_field(vocab_size == expected.vocab_size, "vocab_size",
                  expected.vocab_size, vocab_size);
    require_field(build_path_tag == expected.build_path_tag, "build_path_tag",
                  expected.build_path_tag, build_path_tag);
}

}  // namespace qinf::session
