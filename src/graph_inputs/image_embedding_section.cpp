#include "image_embedding_section.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "session/snapshot_io.h"

namespace qinf::vision {

void ImageEmbeddingSection::write(qinf::session::SnapshotWriter& w) const {
    w.put_u64(e_.content_id);
    w.put_u32(e_.n_embd);
    w.put_u32(e_.n_tokens);
    w.put_bytes(e_.data.data(), e_.data.size() * sizeof(float));
}

void ImageEmbeddingSection::read(qinf::session::SnapshotReader& r) {
    e_.content_id = r.get_u64();
    e_.n_embd = r.get_u32();
    e_.n_tokens = r.get_u32();
    std::vector<uint8_t> bytes;
    r.get_bytes(bytes);
    const size_t expected =
        static_cast<size_t>(e_.n_embd) * e_.n_tokens * sizeof(float);
    if (bytes.size() != expected) {
        throw std::runtime_error(
            "image_embedding: field \"data bytes\" expected " +
            std::to_string(expected) + ", got " + std::to_string(bytes.size()));
    }
    e_.data.resize(static_cast<size_t>(e_.n_embd) * e_.n_tokens);
    std::memcpy(e_.data.data(), bytes.data(), expected);
}

}  // namespace qinf::vision
