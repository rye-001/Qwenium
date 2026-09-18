// test_routing_trace.cpp — the recorded expert selection: round-trip, the
// chunked/decode position model, and the fail-loud edges that make a replayed
// receipt trustworthy rather than merely quiet.

#include <gtest/gtest.h>

#include "../../src/layers/routing_trace.h"

TEST(RoutingTrace, RoundTripsOneLayer) {
    RoutingTrace t;
    const int32_t a[4]{7, 3, 12, 0};
    const int32_t b[4]{1, 1, 5, 9};
    t.write(2, /*position=*/0, a, 4);
    t.write(2, /*position=*/1, b, 4);

    EXPECT_EQ(t.top_k(), 4);
    EXPECT_EQ(t.n_layers(), 1u);
    EXPECT_EQ(t.n_positions(2), 2u);
    EXPECT_FALSE(t.empty());
    for (int k = 0; k < 4; ++k) EXPECT_EQ(t.at(2, 0)[k], a[k]);
    for (int k = 0; k < 4; ++k) EXPECT_EQ(t.at(2, 1)[k], b[k]);
}

// Prefill may arrive in chunks and decode one row at a time, so positions are
// absolute and need not be written in order.
TEST(RoutingTrace, PositionsAreAbsoluteAndMayArriveOutOfOrder) {
    RoutingTrace t;
    const int32_t late[2]{4, 5};
    const int32_t early[2]{8, 9};
    t.write(0, 5, late, 2);
    t.write(0, 1, early, 2);

    EXPECT_EQ(t.n_positions(0), 6u);
    EXPECT_EQ(t.at(0, 5)[0], 4);
    EXPECT_EQ(t.at(0, 1)[0], 8);
}

// A gap is not a selection of expert 0. Reading one must fail rather than
// invent routing, which is the whole point of the type.
TEST(RoutingTrace, ReadingAGapFailsLoudRatherThanReturningZero) {
    RoutingTrace t;
    const int32_t sel[2]{3, 4};
    t.write(0, 2, sel, 2);
    EXPECT_NO_THROW(t.at(0, 2));
    EXPECT_THROW(t.at(0, 0), std::runtime_error);   // never written
    EXPECT_THROW(t.at(0, 9), std::runtime_error);   // past the end
    EXPECT_THROW(t.at(7, 2), std::runtime_error);   // no such layer
}

TEST(RoutingTrace, TopKIsFixedByTheFirstWrite) {
    RoutingTrace t;
    const int32_t two[2]{1, 2};
    const int32_t three[3]{1, 2, 3};
    t.write(0, 0, two, 2);
    EXPECT_THROW(t.write(0, 1, three, 3), std::runtime_error);
    EXPECT_THROW(t.write(1, 0, three, 3), std::runtime_error);
    EXPECT_THROW(t.write(0, 1, two, 0), std::runtime_error);
}

TEST(RoutingTrace, ClearResetsTopKSoATraceCanBeReused) {
    RoutingTrace t;
    const int32_t two[2]{1, 2};
    const int32_t three[3]{1, 2, 3};
    t.write(0, 0, two, 2);
    t.clear();
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.top_k(), 0);
    EXPECT_NO_THROW(t.write(0, 0, three, 3));
    EXPECT_EQ(t.top_k(), 3);
}

// Expert 0 is a real selection and must survive the round trip — the sentinel
// for "never written" is deliberately negative for exactly this reason.
TEST(RoutingTrace, ExpertZeroIsARealSelection) {
    RoutingTrace t;
    const int32_t sel[2]{0, 0};
    t.write(3, 0, sel, 2);
    EXPECT_NO_THROW(t.at(3, 0));
    EXPECT_EQ(t.at(3, 0)[0], 0);
}

// ── Digests ─────────────────────────────────────────────────────────────

namespace {
RoutingTrace two_layer(int32_t tweak_last = -1) {
    RoutingTrace t;
    const int32_t a[3]{4, 9, 2};
    const int32_t b[3]{1, 7, 5};
    const int32_t c[3]{3, 3, tweak_last >= 0 ? tweak_last : 8};
    t.write(0, 0, a, 3);
    t.write(0, 1, b, 3);
    t.write(6, 0, c, 3);
    return t;
}
}  // namespace

TEST(RoutingTraceDigest, IdenticalTracesAgree) {
    EXPECT_EQ(two_layer().digest(), two_layer().digest());
    EXPECT_EQ(two_layer().per_layer(), two_layer().per_layer());
}

TEST(RoutingTraceDigest, OneChangedExpertChangesTheDigest) {
    EXPECT_NE(two_layer().digest(), two_layer(/*tweak_last=*/11).digest());
}

// The point of per-layer digests: localise divergence without carrying the
// selections. A change in layer 6 must leave layer 0's digest alone.
TEST(RoutingTraceDigest, PerLayerLocalisesDivergence) {
    const auto base = two_layer().per_layer();
    const auto moved = two_layer(/*tweak_last=*/11).per_layer();
    ASSERT_EQ(base.size(), 2u);
    EXPECT_EQ(base.at(0), moved.at(0));   // untouched layer
    EXPECT_NE(base.at(6), moved.at(6));   // the one that changed
}

// Order of writes must not matter — only content. Prefill chunks and decode
// steps arrive in different orders across configurations.
TEST(RoutingTraceDigest, WriteOrderDoesNotChangeTheDigest) {
    RoutingTrace fwd, rev;
    const int32_t p0[2]{5, 6};
    const int32_t p1[2]{7, 8};
    fwd.write(0, 0, p0, 2); fwd.write(0, 1, p1, 2);
    rev.write(0, 1, p1, 2); rev.write(0, 0, p0, 2);
    EXPECT_EQ(fwd.digest(), rev.digest());
}

// A digest must distinguish "layer 0 selected X" from "layer 1 selected X",
// or two traces that route differently could collide.
TEST(RoutingTraceDigest, LayerIndexIsPartOfTheDigest) {
    RoutingTrace a, b;
    const int32_t sel[2]{3, 4};
    a.write(0, 0, sel, 2);
    b.write(1, 0, sel, 2);
    EXPECT_NE(a.digest(), b.digest());
}

TEST(RoutingTraceDigest, RefusesALayerItNeverRecorded) {
    EXPECT_THROW(two_layer().digest(99), std::runtime_error);
}

// ── Blob round-trip and the token binding ───────────────────────────────

namespace {
RoutingTrace blob_fixture() {
    RoutingTrace t;
    for (int il : {0, 3, 11, 12, 39})
        for (int pos = 0; pos < 4; ++pos) {
            int32_t sel[4];
            for (int k = 0; k < 4; ++k) sel[k] = il * 100 + pos * 10 + k;
            t.write(il, (size_t)pos, sel, 4);
        }
    t.bind_tokens({7, 8, 9, 10});
    return t;
}
}  // namespace

TEST(RoutingTraceBlob, RoundTripsSelectionsTopKAndTokenBinding) {
    const RoutingTrace a = blob_fixture();
    const RoutingTrace b = RoutingTrace::from_blob(a.to_blob());
    EXPECT_EQ(b.top_k(), a.top_k());
    EXPECT_EQ(b.n_layers(), a.n_layers());
    EXPECT_EQ(b.tokens_digest(), a.tokens_digest());
    EXPECT_EQ(b.digest(), a.digest()) << "a round-tripped trace must fingerprint the same";
    for (int il : {0, 3, 11, 12, 39})
        for (int pos = 0; pos < 4; ++pos)
            for (int k = 0; k < 4; ++k)
                EXPECT_EQ(b.at(il, pos)[k], a.at(il, pos)[k]) << "layer " << il << " pos " << pos;
}

// max_layer is a statement of reach, not a size hack: a truncated verify pass
// builds only the bottom of the stack and can never consult a deeper layer.
TEST(RoutingTraceBlob, MaxLayerKeepsOnlyWhatAConsumerCanReach) {
    const RoutingTrace a = blob_fixture();
    const RoutingTrace b = RoutingTrace::from_blob(a.to_blob(/*max_layer=*/11));
    EXPECT_EQ(b.n_layers(), 3u);                    // 0, 3, 11 — not 12 or 39
    EXPECT_NO_THROW(b.at(11, 0));
    EXPECT_THROW(b.at(12, 0), std::runtime_error);
    EXPECT_THROW(b.at(39, 0), std::runtime_error);
    EXPECT_LT(a.to_blob(11).size(), a.to_blob().size());
}

// THE safety property. An edited extraction tokenizes differently, so replaying
// onto it would pin one token's experts onto another. The binding is what lets
// a consumer notice.
TEST(RoutingTraceBlob, DifferentTokensProduceADifferentBinding) {
    RoutingTrace a, b, c;
    a.bind_tokens({1, 2, 3});
    b.bind_tokens({1, 2, 4});     // one token edited
    c.bind_tokens({1, 2, 3});     // identical
    EXPECT_NE(a.tokens_digest(), b.tokens_digest());
    EXPECT_EQ(a.tokens_digest(), c.tokens_digest());
    RoutingTrace d;
    d.bind_tokens({1, 2, 3, 3});  // length differs
    EXPECT_NE(a.tokens_digest(), d.tokens_digest());
}

TEST(RoutingTraceBlob, RefusesAForeignOrTruncatedBlob) {
    EXPECT_THROW(RoutingTrace::from_blob("bm90LWEtdHJhY2U="), std::runtime_error);  // "not-a-trace"
    EXPECT_THROW(RoutingTrace::from_blob(""), std::runtime_error);                  // empty
    EXPECT_THROW(RoutingTrace::from_blob("!!!!"), std::runtime_error);              // not base64
    const std::string good = blob_fixture().to_blob();
    EXPECT_THROW(RoutingTrace::from_blob(good.substr(0, good.size() / 2)),
                 std::runtime_error) << "a half blob must fail, not decode plausible ids";
}

// The wire format holds ids in int16. A model with more experts than that must
// fail loudly at encode rather than wrap into a valid-looking expert.
TEST(RoutingTraceBlob, RefusesAnExpertIdTheWireFormatCannotHold) {
    RoutingTrace t;
    const int32_t sel[2]{1, 40000};
    t.write(0, 0, sel, 2);
    EXPECT_THROW(t.to_blob(), std::runtime_error);
}
