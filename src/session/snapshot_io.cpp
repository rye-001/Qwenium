#include "snapshot_io.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace qinf::session {

// --- File persistence ------------------------------------------------------

void save_to_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error(
            "snapshot_file: path \"" + path +
            "\" expected to be writable, got open failure");
    }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        throw std::runtime_error(
            "snapshot_file: path \"" + path + "\" expected to write " +
            std::to_string(bytes.size()) + " bytes, got a write failure");
    }
}

std::vector<uint8_t> load_from_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error(
            "snapshot_file: path \"" + path +
            "\" expected to be a readable snapshot, got open failure");
    }
    const std::streamoff size = in.tellg();
    if (size < 0) {
        throw std::runtime_error(
            "snapshot_file: path \"" + path +
            "\" expected a known size, got tellg failure");
    }
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (size > 0) {
        in.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(size));
        if (in.gcount() != size) {
            throw std::runtime_error(
                "snapshot_file: path \"" + path + "\" expected to read " +
                std::to_string(size) + " bytes, got " +
                std::to_string(in.gcount()));
        }
    }
    return bytes;
}

// --- SnapshotWriter (little-endian) ---------------------------------------

void SnapshotWriter::put_u32(uint32_t v) {
    buffer_.push_back(static_cast<uint8_t>(v & 0xFF));
    buffer_.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    buffer_.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    buffer_.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

void SnapshotWriter::put_u64(uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        buffer_.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
    }
}

void SnapshotWriter::put_f32(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    put_u32(bits);
}

void SnapshotWriter::put_bytes(const void* data, size_t len) {
    put_u64(static_cast<uint64_t>(len));
    const uint8_t* p = static_cast<const uint8_t*>(data);
    buffer_.insert(buffer_.end(), p, p + len);
}

void SnapshotWriter::put_string(const std::string& s) {
    put_bytes(s.data(), s.size());
}

// --- SnapshotReader (little-endian, fail-loud) ----------------------------

void SnapshotReader::require(size_t n, const char* field) const {
    if (remaining() < n) {
        throw std::runtime_error(
            std::string("snapshot_reader: field \"") + field +
            "\" expected " + std::to_string(n) + " bytes, got " +
            std::to_string(remaining()));
    }
}

uint32_t SnapshotReader::get_u32() {
    require(4, "u32");
    uint32_t v = static_cast<uint32_t>(data_[pos_]) |
                 (static_cast<uint32_t>(data_[pos_ + 1]) << 8) |
                 (static_cast<uint32_t>(data_[pos_ + 2]) << 16) |
                 (static_cast<uint32_t>(data_[pos_ + 3]) << 24);
    pos_ += 4;
    return v;
}

uint64_t SnapshotReader::get_u64() {
    require(8, "u64");
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(data_[pos_ + i]) << (8 * i);
    }
    pos_ += 8;
    return v;
}

float SnapshotReader::get_f32() {
    uint32_t bits = get_u32();
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

void SnapshotReader::get_bytes(std::vector<uint8_t>& out) {
    uint64_t len = get_u64();
    require(static_cast<size_t>(len), "bytes");
    out.assign(data_ + pos_, data_ + pos_ + len);
    pos_ += static_cast<size_t>(len);
}

std::string SnapshotReader::get_string() {
    uint64_t len = get_u64();
    require(static_cast<size_t>(len), "string");
    std::string s(reinterpret_cast<const char*>(data_ + pos_),
                  static_cast<size_t>(len));
    pos_ += static_cast<size_t>(len);
    return s;
}

}  // namespace qinf::session
