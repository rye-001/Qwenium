// test_l1_sections.cpp — Phase 1a control-state falsifier (model-free).
//
// L1 stores only the control state feed_tokens cannot rebuild: the token span,
// the sampler PRNG, and the grammar cursor. This test proves each section is a
// faithful, bit-exact substitute by reproducing downstream behavior after a
// capture → fresh-instance → restore round-trip:
//   - SamplerStateSection: a restored sampler draws an IDENTICAL token sequence.
//   - GrammarStateSection: a restored grammar yields an IDENTICAL valid set.
//   - TokenSequenceSection: the token vector + gen_start round-trip exactly.
//   - SessionManifest: all three together, framed under a CompatHeader.
// Plus the fail-loud guards (sampler-kind mismatch, grammar restore w/o re-parse).
//
// This is the bitwise-on-CPU control-lane gate. The model-based feed_tokens
// continuation falsifier (KV/recurrent rebuild) is the separate Phase-1b step.

#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "compat_header.h"
#include "grammar_vocab.h"
#include "sampling.h"
#include "sampling_snapshot.h"
#include "section_ids.h"
#include "session_manifest.h"
#include "snapshot_io.h"
#include "token_sequence_section.h"

using namespace qinf;
using qinf::session::CompatHeader;
using qinf::session::SessionManifest;
using qinf::session::SnapshotReader;
using qinf::session::SnapshotWriter;
using qinf::state::TokenSequenceSection;

namespace {

// Stochastic sampler over an 8-way fixed logit vector; top_k/top_p wide and
// repetition penalty disabled so the only nondeterminism is the PRNG draw.
TemperatureSampler make_stochastic_sampler() {
    return TemperatureSampler(/*temperature=*/1.0f, /*repetition_penalty=*/1.0f,
                              /*repetition_lookback=*/64, /*top_k=*/40,
                              /*top_p=*/1.0f);
}

std::vector<int> draw(Sampler& s, int n) {
    const std::vector<float> base = {0.5f, 1.0f, 0.2f, 2.0f,
                                     0.1f, 1.5f, 0.3f, 0.8f};
    std::vector<int> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        std::vector<float> logits = base;  // sample() mutates in place
        out.push_back(s.sample(logits, /*last_tokens=*/{}, /*token_strs=*/{}));
    }
    return out;
}

const std::vector<std::string>& grammar_vocab() {
    static const std::vector<std::string> v = {"a", "b", "c", "ab", "ac", "x"};
    return v;
}

const char* kGbnf = "root ::= \"ab\" | \"ac\"\n";

}  // namespace

TEST(L1SamplerSection, RestoredSamplerDrawsIdenticalSequence) {
    TemperatureSampler a = make_stochastic_sampler();
    draw(a, 5);  // advance the engine to a non-initial state

    SnapshotWriter w;
    SamplerStateSection(a).write(w);

    std::vector<int> seq_a = draw(a, 16);

    TemperatureSampler b = make_stochastic_sampler();  // different seed
    SnapshotReader r(w.buffer());
    SamplerStateSection(b).read(r);
    std::vector<int> seq_b = draw(b, 16);

    EXPECT_EQ(seq_a, seq_b);
    EXPECT_EQ(r.remaining(), 0u);
}

TEST(L1SamplerSection, KindMismatchFailsLoud) {
    GreedySampler greedy;  // no RNG state
    SnapshotWriter w;
    SamplerStateSection(greedy).write(w);

    TemperatureSampler temp = make_stochastic_sampler();
    SnapshotReader r(w.buffer());
    EXPECT_THROW(SamplerStateSection(temp).read(r), std::runtime_error);
}

TEST(L1GrammarSection, RestoredCursorYieldsIdenticalValidSet) {
    const auto& vocab = grammar_vocab();
    auto g = GrammarVocab::parse_impl(kGbnf);
    ASSERT_NE(g, nullptr);
    g->accept_token(0, vocab);  // consume "a" → cursor mid-literal in both alts

    std::vector<int32_t> valid_live = g->get_valid_tokens(vocab);
    ASSERT_FALSE(valid_live.empty());

    SnapshotWriter w;
    GrammarStateSection(*g).write(w);

    auto g2 = GrammarVocab::parse_impl(kGbnf);  // re-parsed, fresh cursor
    ASSERT_NE(g2, nullptr);
    SnapshotReader r(w.buffer());
    GrammarStateSection(*g2).read(r);

    EXPECT_EQ(g->get_valid_tokens(vocab), valid_live);  // unchanged by capture
    EXPECT_EQ(g2->get_valid_tokens(vocab), valid_live);  // restored matches
    EXPECT_EQ(r.remaining(), 0u);

    // Advancing both by the same token keeps them in lock-step.
    g->accept_token(1, vocab);   // "b"
    g2->accept_token(1, vocab);
    EXPECT_EQ(g2->get_valid_tokens(vocab), g->get_valid_tokens(vocab));
    EXPECT_EQ(g2->is_accepting_state(), g->is_accepting_state());
}

TEST(L1GrammarSection, RestoreWithoutReparseFailsLoud) {
    const auto& vocab = grammar_vocab();
    auto g = GrammarVocab::parse_impl(kGbnf);
    ASSERT_NE(g, nullptr);
    g->accept_token(0, vocab);
    SnapshotWriter w;
    GrammarStateSection(*g).write(w);

    GrammarVocab unparsed;  // no rules — cannot map coordinates
    SnapshotReader r(w.buffer());
    EXPECT_THROW(GrammarStateSection(unparsed).read(r), std::runtime_error);
}

TEST(L1TokenSection, RoundTripsExactly) {
    std::vector<int32_t> tokens = {101, 202, 303, 404, 505};
    uint32_t gen_start = 3;
    SnapshotWriter w;
    TokenSequenceSection(tokens, gen_start).write(w);

    std::vector<int32_t> restored;
    uint32_t restored_gen = 0;
    SnapshotReader r(w.buffer());
    TokenSequenceSection(restored, restored_gen).read(r);

    EXPECT_EQ(restored, tokens);
    EXPECT_EQ(restored_gen, gen_start);
    EXPECT_EQ(r.remaining(), 0u);
}

TEST(L1Disk, SaveLoadFileByteRoundTrip) {
    // Phase 2 disk fidelity: file persistence is a pure memcpy (invariant batch
    // shape), so the gate is byte-identical, not token-stable.
    std::vector<int32_t> tokens = {11, 22, 33, 44};
    uint32_t gen_start = 1;
    SessionManifest producer;
    TokenSequenceSection ts(tokens, gen_start);
    producer.add(&ts);
    CompatHeader header;
    header.weights_hash = 0x9911882277ull;
    SnapshotWriter w;
    producer.capture(w, header);

    const std::string path = std::string(testing::TempDir()) + "/l1_disk_rt.bin";
    qinf::session::save_to_file(path, w.buffer());
    std::vector<uint8_t> loaded = qinf::session::load_from_file(path);
    std::remove(path.c_str());

    EXPECT_EQ(loaded, w.buffer());  // byte-identical

    // And the loaded bytes restore to the same state as the in-memory buffer.
    std::vector<int32_t> r_tokens;
    uint32_t r_gen = 0;
    SessionManifest receiver;
    TokenSequenceSection r_ts(r_tokens, r_gen);
    receiver.add(&r_ts);
    SnapshotReader r(loaded);
    receiver.restore(r, header);
    EXPECT_EQ(r_tokens, tokens);
    EXPECT_EQ(r_gen, gen_start);
}

TEST(L1Disk, LoadMissingFileFailsLoud) {
    EXPECT_THROW(qinf::session::load_from_file("/nonexistent/l1_no_such.bin"),
                 std::runtime_error);
}

TEST(L1Manifest, AllThreeSectionsRoundTripUnderHeader) {
    const auto& vocab = grammar_vocab();

    // --- Producer session ---
    std::vector<int32_t> tokens = {7, 8, 9, 10, 11, 12};
    uint32_t gen_start = 4;
    TemperatureSampler sampler = make_stochastic_sampler();
    draw(sampler, 3);
    auto grammar = GrammarVocab::parse_impl(kGbnf);
    ASSERT_NE(grammar, nullptr);
    grammar->accept_token(0, vocab);

    CompatHeader header;
    header.weights_hash = 0xABCDABCDABCDABCDull;
    header.arch_id = 42;

    SessionManifest producer;
    TokenSequenceSection ts(tokens, gen_start);
    SamplerStateSection ss(sampler);
    GrammarStateSection gs(*grammar);
    producer.add(&ts);
    producer.add(&ss);
    producer.add(&gs);

    SnapshotWriter w;
    producer.capture(w, header);

    // Reference downstream behavior, captured AFTER the snapshot.
    std::vector<int> ref_draws = draw(sampler, 12);
    std::vector<int32_t> ref_valid = grammar->get_valid_tokens(vocab);

    // --- Receiver session (fresh instances) ---
    std::vector<int32_t> r_tokens;
    uint32_t r_gen = 0;
    TemperatureSampler r_sampler = make_stochastic_sampler();
    auto r_grammar = GrammarVocab::parse_impl(kGbnf);
    ASSERT_NE(r_grammar, nullptr);

    SessionManifest receiver;
    TokenSequenceSection r_ts(r_tokens, r_gen);
    SamplerStateSection r_ss(r_sampler);
    GrammarStateSection r_gs(*r_grammar);
    receiver.add(&r_ts);
    receiver.add(&r_ss);
    receiver.add(&r_gs);

    SnapshotReader r(w.buffer());
    receiver.restore(r, header);

    EXPECT_EQ(r_tokens, tokens);
    EXPECT_EQ(r_gen, gen_start);
    EXPECT_EQ(draw(r_sampler, 12), ref_draws);
    EXPECT_EQ(r_grammar->get_valid_tokens(vocab), ref_valid);
    EXPECT_EQ(r.remaining(), 0u);
}
