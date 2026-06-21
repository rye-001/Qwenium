// test_image_data_uri.cpp — the chat-endpoint image-input parsing gate.
//
// Smoke test for accepting an OpenAI image content part on /v1/chat/completions:
// it posts one image part (the exact JSON shape the route sees) and asserts the
// data-URI/base64 decode + the fail-loud guards (malformed base64, unsupported
// mime, non-data URL, missing ;base64, oversize). This is the deterministic gate
// for src/server/image_data_uri.{h,cpp}; the full image→soft-token→coherent
// generation path is covered by the gemma4uv coherence smoke (see the server
// README / docs), which needs a model + mmproj and the user's Metal device.
//
// No model file, no encoder, no network — pure request-parsing, runs in ms.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/server/image_data_uri.h"

namespace {

// Minimal standard base64 encoder (test-only) so we can build data URIs from
// bytes we control and assert an exact round-trip through the decoder.
std::string b64_encode(const std::vector<uint8_t>& in) {
    static const char* tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        uint32_t n = (in[i] << 16) | (in[i + 1] << 8) | in[i + 2];
        out.push_back(tbl[(n >> 18) & 63]);
        out.push_back(tbl[(n >> 12) & 63]);
        out.push_back(tbl[(n >> 6) & 63]);
        out.push_back(tbl[n & 63]);
    }
    if (i + 1 == in.size()) {
        uint32_t n = in[i] << 16;
        out.push_back(tbl[(n >> 18) & 63]);
        out.push_back(tbl[(n >> 12) & 63]);
        out += "==";
    } else if (i + 2 == in.size()) {
        uint32_t n = (in[i] << 16) | (in[i + 1] << 8);
        out.push_back(tbl[(n >> 18) & 63]);
        out.push_back(tbl[(n >> 12) & 63]);
        out.push_back(tbl[(n >> 6) & 63]);
        out.push_back('=');
    }
    return out;
}

// The 8-byte PNG signature + a few payload bytes — the kind of bytes a real
// image_url carries. We only check round-trip (the decoder does not parse PNG).
const std::vector<uint8_t> kPngHeadBytes = {
    0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 0x00, 0x01, 0x02, 0x03, 0x04};

std::string png_data_uri() {
    return "data:image/png;base64," + b64_encode(kPngHeadBytes);
}

}  // namespace

using qwenium::decode_image_data_uri;
using qwenium::extract_images_from_content;
using qwenium::content_has_image;

// ── Happy path: a base64 PNG data URI decodes to exactly its bytes ────────────
TEST(ImageDataUri, DecodesPngDataUriRoundTrip) {
    auto img = decode_image_data_uri(png_data_uri());
    EXPECT_EQ(img.mime, "image/png");
    EXPECT_EQ(img.bytes, kPngHeadBytes);
}

TEST(ImageDataUri, MimeIsLowerCasedAndJpegAccepted) {
    auto img = decode_image_data_uri("data:IMAGE/JPEG;base64," + b64_encode({1, 2, 3}));
    EXPECT_EQ(img.mime, "image/jpeg");
    EXPECT_EQ(img.bytes, (std::vector<uint8_t>{1, 2, 3}));
}

// ── "Posting one image part": the exact OpenAI content-array shape ────────────
TEST(ImageDataUri, ExtractsSingleImagePartFromContentArray) {
    nlohmann::json content = nlohmann::json::array({
        {{"type", "text"}, {"text", "what is in this image?"}},
        {{"type", "image_url"}, {"image_url", {{"url", png_data_uri()}}}},
    });

    EXPECT_TRUE(content_has_image(content));
    auto imgs = extract_images_from_content(content);
    ASSERT_EQ(imgs.size(), 1u);
    EXPECT_EQ(imgs[0].mime, "image/png");
    EXPECT_EQ(imgs[0].bytes, kPngHeadBytes);
}

TEST(ImageDataUri, AcceptsBareStringImageUrl) {
    // OpenAI also allows image_url to be a bare string rather than {"url": ...}.
    nlohmann::json content = nlohmann::json::array({
        {{"type", "image_url"}, {"image_url", png_data_uri()}},
    });
    auto imgs = extract_images_from_content(content);
    ASSERT_EQ(imgs.size(), 1u);
    EXPECT_EQ(imgs[0].bytes, kPngHeadBytes);
}

TEST(ImageDataUri, TextOnlyContentHasNoImages) {
    EXPECT_FALSE(content_has_image("just a string"));
    nlohmann::json text_parts = nlohmann::json::array({
        {{"type", "text"}, {"text", "hello"}},
    });
    EXPECT_FALSE(content_has_image(text_parts));
    EXPECT_TRUE(extract_images_from_content(text_parts).empty());
}

// ── Fail-loud guards (each maps to an HTTP 400 in the route) ──────────────────
TEST(ImageDataUri, RejectsRemoteHttpUrl) {
    // Remote URLs are never fetched — only base64 data URIs are accepted.
    try {
        decode_image_data_uri("https://example.com/cat.png");
        FAIL() << "expected throw on remote URL";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("data:"), std::string::npos);
    }
}

TEST(ImageDataUri, RejectsMissingBase64Marker) {
    try {
        decode_image_data_uri("data:image/png,not-base64-here");
        FAIL() << "expected throw on missing ;base64";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("base64"), std::string::npos);
    }
}

TEST(ImageDataUri, RejectsUnsupportedMime) {
    try {
        decode_image_data_uri("data:image/webp;base64," + b64_encode({1, 2, 3}));
        FAIL() << "expected throw on unsupported mime";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("mime type"), std::string::npos);
        EXPECT_NE(std::string(e.what()).find("image/webp"), std::string::npos);
    }
}

TEST(ImageDataUri, RejectsMalformedBase64) {
    // '*' is not a base64 alphabet character.
    try {
        decode_image_data_uri("data:image/png;base64,QUJD*body");
        FAIL() << "expected throw on malformed base64";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("base64"), std::string::npos);
    }
}

TEST(ImageDataUri, RejectsOversizePayload) {
    // 300 bytes of payload against a 16-byte ceiling.
    std::vector<uint8_t> big(300, 0xAB);
    try {
        decode_image_data_uri("data:image/png;base64," + b64_encode(big),
                              /*max_decoded_bytes=*/16);
        FAIL() << "expected throw on oversize payload";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("decoded image size"),
                  std::string::npos);
    }
}

TEST(ImageDataUri, RejectsEmptyPayload) {
    try {
        decode_image_data_uri("data:image/png;base64,");
        FAIL() << "expected throw on empty payload";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("empty"), std::string::npos);
    }
}
