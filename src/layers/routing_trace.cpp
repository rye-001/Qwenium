#include "routing_trace.h"

#include <array>
#include <cstring>
#include <limits>

namespace {

constexpr char kB64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string b64_encode(const std::vector<uint8_t>& in) {
    std::string out;
    out.reserve(((in.size() + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const uint32_t v = (uint32_t(in[i]) << 16) | (uint32_t(in[i + 1]) << 8) | in[i + 2];
        out += kB64[(v >> 18) & 63]; out += kB64[(v >> 12) & 63];
        out += kB64[(v >> 6) & 63];  out += kB64[v & 63];
    }
    if (i < in.size()) {
        uint32_t v = uint32_t(in[i]) << 16;
        const bool two = (i + 1 < in.size());
        if (two) v |= uint32_t(in[i + 1]) << 8;
        out += kB64[(v >> 18) & 63];
        out += kB64[(v >> 12) & 63];
        out += two ? kB64[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

std::vector<uint8_t> b64_decode(const std::string& in) {
    std::array<int8_t, 256> rev{};
    rev.fill(-1);
    for (int i = 0; i < 64; ++i) rev[static_cast<uint8_t>(kB64[i])] = static_cast<int8_t>(i);
    std::vector<uint8_t> out;
    out.reserve(in.size() / 4 * 3);
    uint32_t acc = 0;
    int bits = 0;
    for (char c : in) {
        if (c == '=' || c == '\n' || c == '\r') continue;
        const int8_t d = rev[static_cast<uint8_t>(c)];
        if (d < 0)
            throw std::runtime_error(
                std::string("RoutingTrace::from_blob: expected base64, actual character '") +
                c + "'");
        acc = (acc << 6) | static_cast<uint32_t>(d);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((acc >> bits) & 0xFF));
        }
    }
    return out;
}

void put_u16(std::vector<uint8_t>& b, uint16_t v) { b.push_back(v & 0xFF); b.push_back(v >> 8); }
void put_u32(std::vector<uint8_t>& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF); }
void put_u64(std::vector<uint8_t>& b, uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back((v >> (8 * i)) & 0xFF); }

uint16_t get_u16(const std::vector<uint8_t>& b, size_t& o) { uint16_t v = b[o] | (uint16_t(b[o+1]) << 8); o += 2; return v; }
uint32_t get_u32(const std::vector<uint8_t>& b, size_t& o) { uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= uint32_t(b[o+i]) << (8*i); o += 4; return v; }
uint64_t get_u64(const std::vector<uint8_t>& b, size_t& o) { uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= uint64_t(b[o+i]) << (8*i); o += 8; return v; }

// "QRT1" — magic + version. A blob from a future encoding must be refused,
// not misread: every field below is fixed-width, so a silent misparse would
// produce plausible-looking expert ids rather than an error.
constexpr uint32_t kMagic = 0x31545251u;

}  // namespace

std::string RoutingTrace::to_blob(int max_layer) const {
    std::vector<uint8_t> b;
    std::vector<const std::pair<const int, std::vector<int32_t>>*> keep;
    for (const auto& kv : by_layer_)
        if (max_layer < 0 || kv.first <= max_layer) keep.push_back(&kv);

    put_u32(b, kMagic);
    put_u16(b, static_cast<uint16_t>(top_k_));
    put_u16(b, static_cast<uint16_t>(keep.size()));
    // tokens_digest() not tokens_digest_: the member is a lazily-filled cache
    // and is 0 until something asks. Serializing the raw member shipped a blob
    // bound to nothing, which is the one field replay refuses on.
    put_u64(b, tokens_digest());
    for (const auto* kv : keep) {
        const size_t n = top_k_ ? kv->second.size() / static_cast<size_t>(top_k_) : 0;
        put_u16(b, static_cast<uint16_t>(kv->first));
        put_u32(b, static_cast<uint32_t>(n));
        for (int32_t id : kv->second) {
            if (id < 0 || id > std::numeric_limits<int16_t>::max())
                throw std::runtime_error(
                    "RoutingTrace::to_blob: expected expert ids in [0, 32767], actual " +
                    std::to_string(id) + " — either a gap was never written or this model "
                    "has more experts than the wire format holds");
            put_u16(b, static_cast<uint16_t>(id));
        }
    }
    return b64_encode(b);
}

RoutingTrace RoutingTrace::from_blob(const std::string& b64) {
    const std::vector<uint8_t> b = b64_decode(b64);
    size_t o = 0;
    auto need = [&](size_t n, const char* what) {
        if (o + n > b.size())
            throw std::runtime_error(
                std::string("RoutingTrace::from_blob: expected ") + what + " at byte " +
                std::to_string(o) + ", actual: blob ends at " + std::to_string(b.size()));
    };
    need(16, "a 16-byte header");
    const uint32_t magic = get_u32(b, o);
    if (magic != kMagic)
        throw std::runtime_error(
            "RoutingTrace::from_blob: expected magic 'QRT1', actual 0x" +
            std::to_string(magic) + " — not a routing trace, or a newer encoding");
    RoutingTrace t;
    t.top_k_ = static_cast<int>(get_u16(b, o));
    const int n_layers = static_cast<int>(get_u16(b, o));
    t.tokens_digest_ = get_u64(b, o);
    if (t.top_k_ <= 0)
        throw std::runtime_error(
            "RoutingTrace::from_blob: expected top_k > 0, actual " + std::to_string(t.top_k_));
    for (int l = 0; l < n_layers; ++l) {
        need(6, "a layer header");
        const int il = static_cast<int>(get_u16(b, o));
        const size_t n = get_u32(b, o);
        need(n * static_cast<size_t>(t.top_k_) * 2, "that layer's selections");
        std::vector<int32_t>& rows = t.by_layer_[il];
        rows.resize(n * static_cast<size_t>(t.top_k_));
        for (size_t i = 0; i < rows.size(); ++i) rows[i] = static_cast<int32_t>(get_u16(b, o));
    }
    return t;
}
