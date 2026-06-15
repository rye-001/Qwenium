#include "image_data_uri.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>

namespace qwenium {

namespace {

// The MIME types we accept = the formats the host-side decoder (vendored
// stb_image, see src/cli/image_loader.cpp) can actually decode. WebP is
// deliberately excluded: stb_image does not support it, so accepting it here
// would only defer the failure to an opaque stb error later. Fail-loud now.
bool is_supported_image_mime(const std::string& mime) {
    static const std::array<const char*, 6> kSupported = {
        "image/png", "image/jpeg", "image/jpg",
        "image/gif", "image/bmp",  "image/x-portable-anymap"};
    return std::any_of(kSupported.begin(), kSupported.end(),
                       [&](const char* m) { return mime == m; });
}

inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Standard base64 alphabet → 6-bit value; 0xFF = not a base64 symbol.
int8_t b64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return static_cast<int8_t>(c - 'A');
    if (c >= 'a' && c <= 'z') return static_cast<int8_t>(c - 'a' + 26);
    if (c >= '0' && c <= '9') return static_cast<int8_t>(c - '0' + 52);
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

// Strict base64 decode. Skips ASCII whitespace (data URIs are sometimes wrapped),
// rejects any other non-alphabet byte, and validates padding/length. Throws a
// fail-loud error naming the parameter on malformed input.
std::vector<uint8_t> base64_decode(const std::string& in, size_t max_out_bytes) {
    std::vector<uint8_t> out;
    out.reserve(in.size() / 4 * 3 + 3);

    uint32_t buf = 0;
    int bits = 0;
    int pad = 0;
    size_t symbols = 0;

    for (unsigned char c : in) {
        if (c == '=') { ++pad; continue; }
        if (std::isspace(c)) continue;
        if (pad > 0)
            throw std::runtime_error(
                "image_url.url: parameter 'base64 payload': expected no symbols "
                "after '=' padding, got: a base64 symbol following padding");
        int8_t v = b64_value(c);
        if (v < 0) {
            static const char* h = "0123456789abcdef";
            const std::string hex = {h[(c >> 4) & 0xF], h[c & 0xF]};
            throw std::runtime_error(
                "image_url.url: parameter 'base64 payload': expected a valid "
                "base64 character, got: byte 0x" + hex);
        }
        buf = (buf << 6) | static_cast<uint32_t>(v);
        bits += 6;
        ++symbols;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buf >> bits) & 0xFF));
            if (out.size() > max_out_bytes)
                throw std::runtime_error(
                    "image_url.url: parameter 'decoded image size': expected <= " +
                    std::to_string(max_out_bytes) + " bytes, got: more than that "
                    "(payload exceeds the image byte ceiling)");
        }
    }

    // A well-formed base64 group is 4 symbols; padding makes up the remainder.
    if ((symbols + static_cast<size_t>(pad)) % 4 != 0 || pad > 2)
        throw std::runtime_error(
            "image_url.url: parameter 'base64 payload': expected a length that is "
            "a multiple of 4 (with valid '=' padding), got: " +
            std::to_string(symbols) + " symbols + " + std::to_string(pad) +
            " padding chars");

    return out;
}

}  // namespace

DecodedImage decode_image_data_uri(const std::string& uri, size_t max_decoded_bytes) {
    // Shape: data:[<mime>][;base64],<payload>
    constexpr const char* kPrefix = "data:";
    if (uri.rfind(kPrefix, 0) != 0)
        throw std::runtime_error(
            "image_url.url: parameter 'url': expected a base64 'data:' URI "
            "(remote image URLs are not fetched), got: " +
            (uri.size() > 32 ? uri.substr(0, 32) + "…" : uri));

    const size_t comma = uri.find(',');
    if (comma == std::string::npos)
        throw std::runtime_error(
            "image_url.url: parameter 'url': expected a 'data:' URI with a ','"
            " separating the header from the payload, got: no comma found");

    // Header is between "data:" and the comma, e.g. "image/png;base64".
    const std::string header = uri.substr(5, comma - 5);
    const size_t semi = header.find(';');
    const std::string mime = to_lower(
        semi == std::string::npos ? header : header.substr(0, semi));
    const std::string params =
        semi == std::string::npos ? "" : to_lower(header.substr(semi + 1));

    if (params.find("base64") == std::string::npos)
        throw std::runtime_error(
            "image_url.url: parameter 'encoding': expected ';base64' (only "
            "base64-encoded data URIs are accepted), got: '" + header + "'");

    if (!is_supported_image_mime(mime))
        throw std::runtime_error(
            "image_url.url: parameter 'mime type': expected one of image/png, "
            "image/jpeg, image/jpg, image/gif, image/bmp, image/x-portable-anymap, "
            "got: '" + (mime.empty() ? std::string("<none>") : mime) + "'");

    DecodedImage out;
    out.mime  = mime;
    out.bytes = base64_decode(uri.substr(comma + 1), max_decoded_bytes);
    if (out.bytes.empty())
        throw std::runtime_error(
            "image_url.url: parameter 'base64 payload': expected non-empty image "
            "bytes, got: an empty payload");
    return out;
}

bool content_has_image(const nlohmann::json& content) {
    if (!content.is_array()) return false;
    for (const auto& part : content) {
        if (part.is_object() && part.value("type", "") == "image_url")
            return true;
    }
    return false;
}

std::vector<DecodedImage> extract_images_from_content(
    const nlohmann::json& content, size_t max_decoded_bytes) {
    std::vector<DecodedImage> images;
    if (!content.is_array()) return images;

    for (const auto& part : content) {
        if (!part.is_object() || part.value("type", "") != "image_url") continue;

        // OpenAI allows image_url as an object {"url": "..."} or a bare string.
        std::string url;
        const auto& iu = part.contains("image_url") ? part["image_url"]
                                                     : nlohmann::json();
        if (iu.is_string())       url = iu.get<std::string>();
        else if (iu.is_object())  url = iu.value("url", "");

        if (url.empty())
            throw std::runtime_error(
                "messages[].content[].image_url: parameter 'url': expected a "
                "non-empty data URI, got: missing or empty");

        images.push_back(decode_image_data_uri(url, max_decoded_bytes));
    }
    return images;
}

}  // namespace qwenium
