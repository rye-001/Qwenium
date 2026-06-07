// test_image_prompt.cpp — Phase 6: Gemma 3 image-marker expansion (pure logic).
// Structural gate (no model, no bitmap): the marker becomes start + N soft +
// end, and the reported span points exactly at the soft-token run.

#include <gtest/gtest.h>

#include <vector>

#include "../../src/cli/image_prompt.h"

using qinf::cli::expand_image_markers;

namespace {
constexpr int32_t  START = 1000, SOFT = 2000, END = 3000;
constexpr uint32_t N = 256;
}  // namespace

TEST(ImagePrompt, ExpandsMarkerIntoSoftTokenBlock) {
    // user prefix tokens, the marker, then trailing text tokens.
    std::vector<int32_t> in = {5, 6, START, 7, 8, 9};
    auto out = expand_image_markers(in, START, SOFT, END, N);

    // Length grew by exactly N soft tokens + 1 end-of-image.
    ASSERT_EQ(out.tokens.size(), in.size() + N + 1);

    // Marker survived at index 2; soft span begins right after it.
    EXPECT_EQ(out.tokens[2], START);
    EXPECT_EQ(out.span_start, 3);
    EXPECT_EQ(out.span_len, N);

    // The whole span is soft tokens...
    for (uint32_t k = 0; k < N; ++k)
        EXPECT_EQ(out.tokens[out.span_start + k], SOFT) << "k=" << k;
    // ...immediately closed by end-of-image...
    EXPECT_EQ(out.tokens[out.span_start + N], END);
    // ...and the original tail follows unmodified.
    EXPECT_EQ(out.tokens[out.span_start + N + 1], 7);
    EXPECT_EQ(out.tokens.back(), 9);
}

TEST(ImagePrompt, MarkerAtStartOfStream) {
    std::vector<int32_t> in = {START, 42};
    auto out = expand_image_markers(in, START, SOFT, END, 4);
    EXPECT_EQ(out.span_start, 1);
    EXPECT_EQ(out.span_len, 4u);
    // [START, S,S,S,S, END, 42]
    ASSERT_EQ(out.tokens.size(), 7u);
    EXPECT_EQ(out.tokens[5], END);
    EXPECT_EQ(out.tokens[6], 42);
}

TEST(ImagePrompt, RejectsNoMarker) {
    std::vector<int32_t> in = {1, 2, 3};
    EXPECT_THROW(expand_image_markers(in, START, SOFT, END, N), std::runtime_error);
}

TEST(ImagePrompt, RejectsMultipleMarkers) {
    std::vector<int32_t> in = {START, 1, START};
    EXPECT_THROW(expand_image_markers(in, START, SOFT, END, N), std::runtime_error);
}
