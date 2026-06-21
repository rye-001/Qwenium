#pragma once
// snapshot_io.h — fixed-endian, length-prefixed binary IO for session snapshots.
//
// SnapshotWriter appends to an in-memory byte buffer; SnapshotReader consumes
// one. All multi-byte primitives are serialized little-endian explicitly, so a
// blob written on one host reads identically on another regardless of native
// byte order ("fixed-endian" per docs/plan-session-snapshot.md).
//
// Fail-loud contract (docs/plan-session-snapshot.md, qinf_error.h): a short
// read or a tag mismatch throws, naming the section/field, the expected value,
// and the actual value, in that order. There is no best-effort recovery.
//
// This module has no dependency on ggml or the model — it is pure byte plumbing
// so it can be unit-tested in isolation. Co-located test:
// tests/unit/test_session_manifest.cpp.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qinf::session {

class SnapshotWriter {
public:
    void put_u32(uint32_t v);
    void put_u64(uint64_t v);
    void put_f32(float v);

    // Length-prefixed raw bytes: [u64 len][bytes].
    void put_bytes(const void* data, size_t len);
    // Length-prefixed UTF-8 string: [u64 len][bytes], no trailing NUL.
    void put_string(const std::string& s);

    const std::vector<uint8_t>& buffer() const { return buffer_; }
    size_t size() const { return buffer_.size(); }

private:
    std::vector<uint8_t> buffer_;
};

// --- File persistence (Phase 2) -------------------------------------------
// Persist a serialized snapshot buffer to disk and load it back. Byte-identical
// round-trip: the batch shape is invariant (a pure memcpy of the buffer), so
// this is the byte gate, not token-stable. Fail-loud on any IO error (cannot
// open, short write/read), naming the path, in the qinf_error.h style.
void save_to_file(const std::string& path, const std::vector<uint8_t>& bytes);
std::vector<uint8_t> load_from_file(const std::string& path);

class SnapshotReader {
public:
    // Borrows the buffer; it must outlive the reader.
    SnapshotReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}
    explicit SnapshotReader(const std::vector<uint8_t>& buf)
        : data_(buf.data()), size_(buf.size()) {}

    uint32_t get_u32();
    uint64_t get_u64();
    float    get_f32();

    // Reads a length-prefixed byte blob into out.
    void get_bytes(std::vector<uint8_t>& out);
    std::string get_string();

    size_t position() const { return pos_; }
    size_t remaining() const { return size_ - pos_; }

private:
    // Fail-loud bounds check; names the reader, the bytes expected, and the
    // bytes actually remaining, in that order.
    void require(size_t n, const char* field) const;

    const uint8_t* data_;
    size_t         size_;
    size_t         pos_ = 0;
};

}  // namespace qinf::session
