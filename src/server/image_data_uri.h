#pragma once
// image_data_uri.h — parse an OpenAI `image_url` data-URI into raw image bytes.
//
// The chat-completions endpoint accepts image content parts in the OpenAI shape:
//
//   "content": [{"type":"text","text":"..."},
//               {"type":"image_url","image_url":{"url":"data:image/png;base64,<...>"}}]
//
// This module turns a single such `image_url.url` data-URI into the decoded
// image bytes (which the host-side image pipeline then decodes/resizes/normalizes
// via stb), and walks a whole `content` array to collect every image part.
//
// It is the request-parsing layer ONLY — it does not touch the vision encoder or
// the model. Kept as its own unit (not buried in http_server.cpp's main()-bearing
// TU) so the base64 decode + the fail-loud guards are directly unit-testable.
//
// Fail-loud contract (CLAUDE.md): every rejection throws std::runtime_error
// naming the parameter, the expected value, and the actual value, in that order.
// Callers in the HTTP route translate the throw into an HTTP 400. Guards cover:
//   - not a `data:` URI (e.g. a remote http(s) URL — unsupported, never fetched);
//   - missing `;base64` (only base64 payloads are accepted);
//   - an unsupported / non-image MIME type;
//   - malformed base64;
//   - an oversized decoded payload (byte ceiling).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace qinf {

// Default ceiling on the DECODED image size (bytes). A 4 K order-management
// screenshot is well under this; a runaway payload is rejected fail-loud rather
// than handed to stb to OOM on. Override per call if needed.
constexpr size_t kDefaultMaxImageBytes = 20u * 1024 * 1024;  // 20 MiB

struct DecodedImage {
    std::string          mime;   // e.g. "image/png" (validated, lower-cased)
    std::vector<uint8_t> bytes;  // raw decoded image file bytes (PNG/JPEG/…)
};

// Decode one `data:<mime>;base64,<payload>` URI. Throws (named param
// "image_url.url") on any malformed/unsupported/oversize input.
DecodedImage decode_image_data_uri(const std::string& uri,
                                   size_t max_decoded_bytes = kDefaultMaxImageBytes);

// True if `content` (a chat message's content field) carries at least one
// image part. A string or null content has no images.
bool content_has_image(const nlohmann::json& content);

// Walk a message's `content` array and decode every `image_url` part, in order.
// A string/null content yields an empty vector. Throws (named param) on the
// first malformed image part. `max_decoded_bytes` bounds EACH image.
std::vector<DecodedImage> extract_images_from_content(
    const nlohmann::json& content,
    size_t max_decoded_bytes = kDefaultMaxImageBytes);

}  // namespace qinf
