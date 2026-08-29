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
