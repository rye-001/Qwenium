// Unit tests for src/sampling/sampling.cpp.
//
// The load-bearing property here is that greedy means argmax. GreedySampler
// carried a 1.2 repetition-penalty default until 2026-08, so every
// temperature-0 generation was quietly steered away from the model's actual
// argmax -- which read from the outside as a forward-pass divergence from
// llama.cpp and HF that did not exist. These tests pin the contract so it
// cannot regress silently again.

#include <gtest/gtest.h>

#include <vector>
#include <string>

#include "../../src/sampling/sampling.h"
#include "../../src/sampling/grammar_vocab.h"

using qinf::GreedySampler;

namespace {

// logits where token 3 is the clear argmax, and token 3 is ALSO in recent
// history -- exactly the shape a repetition penalty perturbs.
std::vector<float> peaked_logits() {
    return {1.0f, 2.0f, 3.0f, 10.0f, 4.0f};
}

}  // namespace

TEST(GreedySamplerTest, DefaultIsTrueArgmaxEvenForARepeatedToken) {
    GreedySampler s;                       // default ctor == what every call site uses
    auto logits = peaked_logits();
    const std::vector<int32_t> history{3, 3, 3};   // token 3 repeated
    EXPECT_EQ(s.sample(logits, history), 3);
}

TEST(GreedySamplerTest, DefaultDoesNotModifyLogits) {
    GreedySampler s;
    auto logits = peaked_logits();
    const auto before = logits;
    const std::vector<int32_t> history{0, 1, 2, 3, 4};
    s.sample(logits, history);
    EXPECT_EQ(logits, before) << "greedy must not perturb the caller's logits";
}

TEST(GreedySamplerTest, ExplicitPenaltyStillAvailableAndDemotesRepeats) {
    // Penalized argmax is not gone -- it just has to be asked for.
    // 10.0 / 2.0 = 5.0, which drops token 3 below token 4 (4.0)? No: 5.0 > 4.0,
    // so use a penalty large enough to actually flip the choice.
    GreedySampler s(/*repetition_penalty=*/4.0f);
    auto logits = peaked_logits();          // argmax = 3 (10.0), runner-up = 4 (4.0)
    const std::vector<int32_t> history{3};
    EXPECT_EQ(s.sample(logits, history), 4) << "10.0/4.0 = 2.5 < 4.0, so 4 wins";
}

TEST(GreedySamplerTest, PenaltyOneIsIdenticalToDefault) {
    GreedySampler def;
    GreedySampler one(1.0f);
    auto a = peaked_logits();
    auto b = peaked_logits();
    const std::vector<int32_t> history{3, 3};
    EXPECT_EQ(def.sample(a, history), one.sample(b, history));
    EXPECT_EQ(a, b);
}

TEST(GreedySamplerTest, EmptyHistoryIsArgmax) {
    GreedySampler s;
    auto logits = peaked_logits();
    EXPECT_EQ(s.sample(logits, {}), 3);
}

TEST(GreedySamplerTest, SparsePathDefaultIsAlsoTrueArgmax) {
    // sample_sparse returns the token ID, not the index into sparse_logits.
    GreedySampler s;
    std::vector<float>   sparse{1.0f, 9.0f, 2.0f};
    std::vector<int32_t> valid{10, 20, 30};
    const std::vector<int32_t> history{20, 20};   // the sparse argmax, repeated
    EXPECT_EQ(s.sample_sparse(sparse, valid, history), 20);
}

TEST(GreedySamplerTest, EmptyLogitsFailLoud) {
    GreedySampler s;
    std::vector<float> empty;
    EXPECT_THROW(s.sample(empty, {}), std::runtime_error);
}

// Banned tokens (tool_choice "none" bans a tool format's call opener,
// docs/plan-tool-calling.md): never returned, on the dense and the sparse path,
// greedy or sampled.
TEST(BannedTokensTest, NeverReturnedDenseOrSparse) {
    GreedySampler g;
    g.set_banned_tokens({3});
    auto logits = peaked_logits();
    EXPECT_EQ(g.sample(logits, {}), 4);   // the runner-up, not the banned argmax
    std::vector<float> sparse = {10.0f, 4.0f};
    EXPECT_EQ(g.sample_sparse(sparse, {3, 4}, {}), 4);

    qinf::TemperatureSampler t(/*temperature=*/1.0f, /*repetition_penalty=*/1.0f, 64, /*top_k=*/5, /*top_p=*/1.0f);
    t.seed(7);
    t.set_banned_tokens({3});
    for (int i = 0; i < 200; ++i) {
        std::vector<float> l = {1.0f, 1.0f, 1.0f, 50.0f, 1.0f};
        ASSERT_NE(t.sample(l, {}, {}), 3);
        std::vector<float> sp = {50.0f, 1.0f};
        ASSERT_NE(t.sample_sparse(sp, {3, 4}, {}), 3);
    }
}

TEST(BannedTokensTest, NoBanLeavesSamplingUnchanged) {
    GreedySampler s;
    auto logits = peaked_logits();
    const auto before = logits;
    EXPECT_EQ(s.sample(logits, {}), 3);
    EXPECT_EQ(logits, before);
}

// The reasoning gate (tool_choice "required", docs/plan-tool-calling.md P4):
// the grammar does not constrain the reasoning block between the opener and
// the closer, and neither marker is fed to it.
namespace {
//                                    0    1    2      3       4
const std::vector<std::string> kVocab = {"a", "b", "<t>", "</t>", "x"};
std::vector<float> favour(int id) {
    std::vector<float> l(kVocab.size(), 1.0f);
    l[id] = 10.0f;
    return l;
}
}  // namespace

TEST(GrammarGateTest, ReasoningRunsFreeThenTheGrammarHolds) {
    auto g = qinf::GrammarVocab::parse_impl("root ::= \"ab\"");
    GreedySampler s;
    s.set_grammar(g.get());
    s.build_token_trie(kVocab);
    s.set_grammar_gate(/*opener=*/2, /*closer=*/3, /*start_inside=*/false);

    auto l = favour(4);   // "x" is not grammar-valid; the opener is the allowed runner-up
    l[2] = 9.0f;
    int t = s.sample(l, {}, kVocab);
    EXPECT_EQ(t, 2);
    s.accept_token(t);
    EXPECT_FALSE(s.grammar_active());
    l = favour(4);         // inside the reasoning anything goes
    t = s.sample(l, {}, kVocab);
    EXPECT_EQ(t, 4);
    s.accept_token(t);
    l = favour(3);
    t = s.sample(l, {}, kVocab);
    EXPECT_EQ(t, 3);
    s.accept_token(t);
    EXPECT_TRUE(s.grammar_active());
    l = favour(4);         // after the closer the grammar holds: "a", then "b"
    t = s.sample(l, {}, kVocab);
    EXPECT_EQ(t, 0);
    s.accept_token(t);
    l = favour(4);
    EXPECT_EQ(s.sample(l, {}, kVocab), 1);
    s.accept_token(1);
    EXPECT_TRUE(g->is_accepting_state());
}

TEST(GrammarGateTest, AFirstTokenThatIsNotTheOpenerActivatesTheGrammar) {
    auto g = qinf::GrammarVocab::parse_impl("root ::= \"ab\"");
    GreedySampler s;
    s.set_grammar(g.get());
    s.build_token_trie(kVocab);
    s.set_grammar_gate(2, 3, false);
    auto l = favour(0);
    EXPECT_EQ(s.sample(l, {}, kVocab), 0);
    s.accept_token(0);
    l = favour(2);         // the opener is no longer allowed: only "b"
    EXPECT_EQ(s.sample(l, {}, kVocab), 1);
}

TEST(GrammarGateTest, StartInsideIsUnconstrainedUntilTheCloser) {
    auto g = qinf::GrammarVocab::parse_impl("root ::= \"ab\"");
    GreedySampler s;
    s.set_grammar(g.get());
    s.build_token_trie(kVocab);
    s.set_grammar_gate(2, 3, /*start_inside=*/true);
    EXPECT_FALSE(s.grammar_active());
    EXPECT_TRUE(s.peek_valid_set().empty());   // no sparse set while ungated text runs
    auto l = favour(1);
    EXPECT_EQ(s.sample(l, {}, kVocab), 1);
    s.accept_token(1);
    s.accept_token(3);
    EXPECT_TRUE(s.grammar_active());
    l = favour(1);
    EXPECT_EQ(s.sample(l, {}, kVocab), 0);
}
