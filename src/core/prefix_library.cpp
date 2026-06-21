#include "prefix_library.h"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "session/snapshot_io.h"

using qinf::session::CompatHeader;
using qinf::session::SnapshotReader;

namespace fs = std::filesystem;

uint64_t PrefixLibrary::key_for(const std::vector<int32_t>& prefix_tokens) {
    if (prefix_tokens.empty()) return 0;  // non-cacheable sentinel
    // FNV-1a over the token bytes, little-endian, so the key is host-independent.
    uint64_t h = 1469598103934665603ull;
    for (int32_t t : prefix_tokens) {
        uint32_t u = static_cast<uint32_t>(t);
        for (int b = 0; b < 4; ++b) {
            h ^= static_cast<uint8_t>(u >> (8 * b));
            h *= 1099511628211ull;
        }
    }
    // Map a hash that happens to be 0 off the non-cacheable sentinel.
    return h == 0 ? 1 : h;
}

uint64_t PrefixLibrary::key_for(const std::vector<int32_t>& preceding_tokens,
                                uint64_t image_content_id) {
    if (image_content_id == 0) return 0;  // unset bitmap = non-cacheable
    // FNV-1a over the preceding-context token bytes (little-endian), then the
    // 8 content_id bytes, so the key is host-independent and discriminates both
    // the text before the image AND which image it is.
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint8_t byte) {
        h ^= byte;
        h *= 1099511628211ull;
    };
    for (int32_t t : preceding_tokens) {
        uint32_t u = static_cast<uint32_t>(t);
        for (int b = 0; b < 4; ++b) mix(static_cast<uint8_t>(u >> (8 * b)));
    }
    for (int b = 0; b < 8; ++b)
        mix(static_cast<uint8_t>(image_content_id >> (8 * b)));
    // Map a hash that happens to be 0 off the non-cacheable sentinel.
    return h == 0 ? 1 : h;
}

PrefixLibrary::PrefixLibrary(std::string dir, CompatHeader expected,
                             size_t max_entries)
    : dir_(std::move(dir)), expected_(expected), max_entries_(max_entries) {
    std::error_code ec;
    fs::create_directories(dir_, ec);  // best-effort; store() fails loud if unwritable
}

std::string PrefixLibrary::path_for(uint64_t key) const {
    return dir_ + "/prefix-" + std::to_string(key) + ".snap";
}

bool PrefixLibrary::try_load(uint64_t key, std::vector<uint8_t>& out) const {
    if (key == 0) return false;
    const std::string path = path_for(key);
    std::error_code ec;
    if (!fs::exists(path, ec)) return false;  // absent: first-use miss

    // Present: read the embedded header and gate it FAIL-LOUD. Unlike the
    // opportunistic image store, a mismatch here is a hard error — a prefix blob
    // from a different model / quant / kernel path must never be resumed, and
    // never silently re-prefilled (Phase 4 non-negotiable).
    std::vector<uint8_t> bytes = qinf::session::load_from_file(path);
    SnapshotReader r(bytes);
    CompatHeader stored = CompatHeader::read(r);
    stored.check(expected_);  // throws naming the first mismatched field
    out = std::move(bytes);
    return true;
}

void PrefixLibrary::evict_if_full() const {
    if (max_entries_ == 0) return;
    std::error_code ec;
    std::vector<fs::directory_entry> entries;
    for (const auto& e : fs::directory_iterator(dir_, ec)) {
        if (e.is_regular_file(ec) && e.path().extension() == ".snap")
            entries.push_back(e);
    }
    // Evict oldest-by-mtime until there is room for one more.
    while (entries.size() >= max_entries_) {
        auto oldest = entries.begin();
        for (auto it = entries.begin(); it != entries.end(); ++it)
            if (it->last_write_time(ec) < oldest->last_write_time(ec)) oldest = it;
        fs::remove(oldest->path(), ec);
        entries.erase(oldest);
    }
}

void PrefixLibrary::store(uint64_t key, const std::vector<uint8_t>& blob) const {
    if (key == 0) return;  // non-cacheable sentinel
    evict_if_full();

    const std::string final_path = path_for(key);
    const std::string tmp_path = final_path + ".tmp";
    qinf::session::save_to_file(tmp_path, blob);
    std::error_code ec;
    fs::rename(tmp_path, final_path, ec);
    if (ec) {
        std::remove(tmp_path.c_str());
        throw std::runtime_error(
            "prefix_library: rename to \"" + final_path +
            "\" expected to succeed, got error: " + ec.message());
    }
}
