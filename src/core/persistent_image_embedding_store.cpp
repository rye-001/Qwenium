#include "persistent_image_embedding_store.h"

#include <cstdio>
#include <filesystem>
#include <functional>
#include <stdexcept>

#include "session/snapshot_io.h"

using qinf::session::CompatHeader;
using qinf::session::SnapshotReader;
using qinf::session::SnapshotWriter;
using qinf::vision::ImageEmbedding;
using qinf::vision::ImageEmbeddingSection;

CompatHeader make_vision_header(const std::string& projector_tag,
                                uint32_t projection_dim,
                                const std::string& backend_tag) {
    CompatHeader h;
    h.arch_id = static_cast<uint32_t>(std::hash<std::string>{}(projector_tag));
    h.embedding_length = projection_dim;
    h.build_path_tag =
        static_cast<uint64_t>(std::hash<std::string>{}("vision|" + backend_tag));
    return h;
}

PersistentImageEmbeddingStore::PersistentImageEmbeddingStore(
    std::string dir, CompatHeader header)
    : dir_(std::move(dir)), header_(header) {
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);  // best-effort; store() fails loud if unwritable
}

std::string PersistentImageEmbeddingStore::path_for(uint64_t content_id) const {
    return dir_ + "/img-" + std::to_string(content_id) + ".embd";
}

void PersistentImageEmbeddingStore::store(const ImageEmbedding& e) const {
    if (e.content_id == 0) return;  // non-cacheable sentinel

    SnapshotWriter w;
    header_.write(w);
    ImageEmbeddingSection(const_cast<ImageEmbedding&>(e)).write(w);

    // Atomic publish: write a temp file then rename, so a concurrent reader
    // never sees a torn blob.
    const std::string final_path = path_for(e.content_id);
    const std::string tmp_path = final_path + ".tmp";
    qinf::session::save_to_file(tmp_path, w.buffer());
    std::error_code ec;
    std::filesystem::rename(tmp_path, final_path, ec);
    if (ec) {
        std::remove(tmp_path.c_str());
        throw std::runtime_error(
            "persistent_image_embedding_store: rename to \"" + final_path +
            "\" expected to succeed, got error: " + ec.message());
    }
}

bool PersistentImageEmbeddingStore::try_load(uint64_t content_id,
                                             ImageEmbedding& out) const {
    if (content_id == 0) return false;
    const std::string path = path_for(content_id);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) return false;

    // Any failure past here (corrupt bytes, header/version mismatch, content_id
    // mismatch) is a MISS, not an error: the caller re-encodes the correct
    // embedding. A mismatched blob is ignored, never used.
    try {
        std::vector<uint8_t> bytes = qinf::session::load_from_file(path);
        SnapshotReader r(bytes);
        CompatHeader stored = CompatHeader::read(r);
        stored.check(header_);  // throws on version / encoder-identity mismatch
        ImageEmbedding e;
        ImageEmbeddingSection(e).read(r);
        if (e.content_id != content_id) return false;  // collision / corruption guard
        out = std::move(e);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

ImageEmbedding PersistentImageEmbeddingStore::get_or_encode(
    uint64_t content_id,
    const std::function<ImageEmbedding()>& encode) const {
    ImageEmbedding hit;
    if (try_load(content_id, hit)) return hit;
    ImageEmbedding fresh = encode();
    store(fresh);
    return fresh;
}
