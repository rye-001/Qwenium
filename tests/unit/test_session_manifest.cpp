// test_session_manifest.cpp — Phase 0 gate (docs/plan-session-snapshot.md).
//
// Exercises the session-snapshot seams in isolation (no model, no ggml):
//   - CompatHeader round-trips byte-for-byte.
//   - Every header-field mismatch fails loud with the field named.
//   - An empty manifest captures + restores a header only.
//   - Section framing round-trips, and the consumed-bytes fidelity check + the
//     short-read bounds check both fail loud.

#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "compat_header.h"
#include "session_manifest.h"
#include "snapshot_io.h"
#include "snapshot_section.h"

using namespace qinf::session;

namespace {

CompatHeader make_populated_header() {
    CompatHeader h;
    h.qwenium_version = 0xABCDEF0123456789ull;
    h.weights_hash = 0x1122334455667788ull;
    h.quant_scheme = 7;
    h.arch_id = 0xDEADBEEF;
    h.block_count = 36;
    h.embedding_length = 4096;
    h.attention_head_count = 32;
    h.attention_head_count_kv = 8;
    h.attention_key_length = 128;
    h.vocab_size = 151936;
    h.build_path_tag = 0xFEEDFACECAFEBEEFull;
    return h;
}

bool message_contains(const std::runtime_error& e, const std::string& needle) {
    return std::string(e.what()).find(needle) != std::string::npos;
}

// A trivial Control section that writes/reads a fixed payload, used to exercise
// the manifest's framing and fidelity checks without any model state.
struct FakeSection : SnapshotSection {
    uint32_t value = 0;
    bool under_read = false;  // when set on read, leaves bytes unconsumed

    SectionId id() const override { return 0x51010101u; }
    StateLane lane() const override { return StateLane::Control; }
    void write(SnapshotWriter& w) const override {
        w.put_u32(value);
        w.put_u32(value ^ 0xFFFFFFFFu);
    }
    void read(SnapshotReader& r) override {
        value = r.get_u32();
        if (!under_read) {
            (void)r.get_u32();  // consume the rest
        }
    }
};

}  // namespace

TEST(CompatHeader, RoundTripsByteForByte) {
    CompatHeader h = make_populated_header();
    SnapshotWriter w;
    h.write(w);

    SnapshotReader r(w.buffer());
    CompatHeader back = CompatHeader::read(r);

    EXPECT_EQ(r.remaining(), 0u);
    EXPECT_EQ(back.magic, h.magic);
    EXPECT_EQ(back.format_version, h.format_version);
    EXPECT_EQ(back.qwenium_version, h.qwenium_version);
    EXPECT_EQ(back.weights_hash, h.weights_hash);
    EXPECT_EQ(back.quant_scheme, h.quant_scheme);
    EXPECT_EQ(back.arch_id, h.arch_id);
    EXPECT_EQ(back.block_count, h.block_count);
    EXPECT_EQ(back.embedding_length, h.embedding_length);
    EXPECT_EQ(back.attention_head_count, h.attention_head_count);
    EXPECT_EQ(back.attention_head_count_kv, h.attention_head_count_kv);
    EXPECT_EQ(back.attention_key_length, h.attention_key_length);
    EXPECT_EQ(back.vocab_size, h.vocab_size);
    EXPECT_EQ(back.build_path_tag, h.build_path_tag);
}

TEST(CompatHeader, MatchingHeaderChecksClean) {
    CompatHeader a = make_populated_header();
    CompatHeader b = make_populated_header();
    EXPECT_NO_THROW(a.check(b));
}

// Every discriminant must fail loud naming its field. The pairs are
// {field-name-in-message, mutator}.
TEST(CompatHeader, EveryFieldMismatchFailsLoud) {
    const std::vector<std::pair<std::string, std::function<void(CompatHeader&)>>> cases = {
        {"magic", [](CompatHeader& h) { h.magic ^= 1u; }},
        {"format_version", [](CompatHeader& h) { h.format_version += 1u; }},
        {"qwenium_version", [](CompatHeader& h) { h.qwenium_version ^= 1u; }},
        {"weights_hash", [](CompatHeader& h) { h.weights_hash ^= 1u; }},
        {"quant_scheme", [](CompatHeader& h) { h.quant_scheme += 1u; }},
        {"arch_id", [](CompatHeader& h) { h.arch_id += 1u; }},
        {"block_count", [](CompatHeader& h) { h.block_count += 1u; }},
        {"embedding_length", [](CompatHeader& h) { h.embedding_length += 1u; }},
        {"attention_head_count", [](CompatHeader& h) { h.attention_head_count += 1u; }},
        {"attention_head_count_kv", [](CompatHeader& h) { h.attention_head_count_kv += 1u; }},
        {"attention_key_length", [](CompatHeader& h) { h.attention_key_length += 1u; }},
        {"vocab_size", [](CompatHeader& h) { h.vocab_size += 1u; }},
        {"build_path_tag", [](CompatHeader& h) { h.build_path_tag ^= 1u; }},
    };

    for (const auto& [field, mutate] : cases) {
        CompatHeader expected = make_populated_header();
        CompatHeader producer = make_populated_header();
        mutate(producer);
        try {
            producer.check(expected);
            ADD_FAILURE() << "expected throw for field " << field;
        } catch (const std::runtime_error& e) {
            EXPECT_TRUE(message_contains(e, field))
                << "field " << field << " message: " << e.what();
        }
    }
}

TEST(SessionManifest, EmptyManifestRoundTripsHeaderOnly) {
    CompatHeader h = make_populated_header();
    SessionManifest producer;
    SnapshotWriter w;
    producer.capture(w, h);

    SessionManifest receiver;
    SnapshotReader r(w.buffer());
    CompatHeader back = receiver.restore(r, h);

    EXPECT_EQ(back.weights_hash, h.weights_hash);
    EXPECT_EQ(r.remaining(), 0u);
    EXPECT_EQ(receiver.section_count(), 0u);
}

TEST(SessionManifest, RejectsAddNullSection) {
    SessionManifest m;
    EXPECT_THROW(m.add(nullptr), std::runtime_error);
}

TEST(SessionManifest, IncompatibleHeaderRefusedFailLoud) {
    CompatHeader producer_hdr = make_populated_header();
    SessionManifest producer;
    SnapshotWriter w;
    producer.capture(w, producer_hdr);

    CompatHeader receiver_hdr = make_populated_header();
    receiver_hdr.weights_hash ^= 1u;  // different model weights

    SessionManifest receiver;
    SnapshotReader r(w.buffer());
    try {
        receiver.restore(r, receiver_hdr);
        ADD_FAILURE() << "expected throw on weights_hash mismatch";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_contains(e, "weights_hash")) << e.what();
    }
}

TEST(SessionManifest, SectionFramingRoundTrips) {
    CompatHeader h = make_populated_header();

    FakeSection out;
    out.value = 0x0BADC0DE;
    SessionManifest producer;
    producer.add(&out);
    SnapshotWriter w;
    producer.capture(w, h);

    FakeSection in;
    SessionManifest receiver;
    receiver.add(&in);
    SnapshotReader r(w.buffer());
    receiver.restore(r, h);

    EXPECT_EQ(in.value, out.value);
    EXPECT_EQ(r.remaining(), 0u);
}

// A section that under-consumes its framed payload must fail loud — this is the
// writer/reader drift guard.
TEST(SessionManifest, SectionConsumedBytesMismatchFailsLoud) {
    CompatHeader h = make_populated_header();

    FakeSection out;
    out.value = 1;
    SessionManifest producer;
    producer.add(&out);
    SnapshotWriter w;
    producer.capture(w, h);

    FakeSection in;
    in.under_read = true;  // leaves 4 bytes unconsumed
    SessionManifest receiver;
    receiver.add(&in);
    SnapshotReader r(w.buffer());
    EXPECT_THROW(receiver.restore(r, h), std::runtime_error);
}

TEST(SnapshotReader, ShortReadFailsLoud) {
    SnapshotWriter w;
    w.put_u64(42);
    // Truncate to fewer bytes than a u64.
    std::vector<uint8_t> truncated(w.buffer().begin(), w.buffer().begin() + 3);
    SnapshotReader r(truncated);
    EXPECT_THROW(r.get_u64(), std::runtime_error);
}

TEST(SnapshotReader, BytesLengthOverrunFailsLoud) {
    SnapshotWriter w;
    w.put_u64(1000);  // claims 1000 bytes follow
    SnapshotReader r(w.buffer());
    std::vector<uint8_t> out;
    EXPECT_THROW(r.get_bytes(out), std::runtime_error);
}
