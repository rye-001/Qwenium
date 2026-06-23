// test_inference_server.cpp — warm chat-prefix slot routing, model-free.
//
// Exercises InferenceServer::warm_prefix_pick — the pure prefix-aware slot
// selector behind --chat-prefix-cache (docs/plan-chat-prefix-cache.md). The rest
// of the warm path (KV retain/clear, suffix prefill at the engine position) is a
// model-on-Metal concern covered by tests/smoke/server_chat_cache_smoke.sh; this
// file pins the matching logic that decides warm-vs-cold and which slot to reuse.
//
// Contract under test:
//   - strict-prefix-append ONLY: a slot is reused iff its ENTIRE retained token
//     stream is a prefix of the request AND strictly shorter than it. A shorter
//     common prefix (divergence) is a MISS, never a partial rewind — that would
//     corrupt overwrite-semantics recurrent state on hybrid recipes (CLAUDE.md).
//   - longest match wins (most KV reused).
//   - only FREE slots are eligible; active slots are skipped.
//   - no match ⇒ first free slot, lcp 0 (cold). no free slot ⇒ {-1, 0}.

#include <vector>

#include <gtest/gtest.h>

#include "../../src/server/inference_server.h"

using qwenium::InferenceServer;
using Pick = InferenceServer::WarmPick;
using Tokens = std::vector<int32_t>;

namespace {

// active[i] == 0 ⇒ free. Helper to express "all slots free".
std::vector<char> all_free(size_t n) { return std::vector<char>(n, 0); }

}  // namespace

TEST(WarmPrefixPick, NoFreeSlotReturnsMinusOne) {
    std::vector<Tokens> residents = {{1, 2}, {3, 4}};
    std::vector<char> active = {1, 1};  // both busy
    Pick p = InferenceServer::warm_prefix_pick(residents, active, {1, 2, 3});
    EXPECT_EQ(p.slot_id, -1);
    EXPECT_EQ(p.lcp, 0u);
}

TEST(WarmPrefixPick, AllFreeNoResidentsPicksFirstCold) {
    std::vector<Tokens> residents = {{}, {}, {}};
    Pick p = InferenceServer::warm_prefix_pick(residents, all_free(3), {1, 2, 3});
    EXPECT_EQ(p.slot_id, 0);
    EXPECT_EQ(p.lcp, 0u);  // cold
}

TEST(WarmPrefixPick, StrictPrefixIsAWarmHit) {
    // slot 1 holds [10,20,30], the request is [10,20,30,40] → reuse all 3.
    std::vector<Tokens> residents = {{}, {10, 20, 30}, {}};
    Pick p = InferenceServer::warm_prefix_pick(residents, all_free(3),
                                               {10, 20, 30, 40});
    EXPECT_EQ(p.slot_id, 1);
    EXPECT_EQ(p.lcp, 3u);
}

TEST(WarmPrefixPick, LongestMatchWins) {
    // Both slot 0 ([10,20]) and slot 2 ([10,20,30]) are prefixes of the request;
    // slot 2 reuses more KV, so it wins.
    std::vector<Tokens> residents = {{10, 20}, {99}, {10, 20, 30}};
    Pick p = InferenceServer::warm_prefix_pick(residents, all_free(3),
                                               {10, 20, 30, 40, 50});
    EXPECT_EQ(p.slot_id, 2);
    EXPECT_EQ(p.lcp, 3u);
}

TEST(WarmPrefixPick, ExactEqualIsNotReused) {
    // resident == request (no new turn to append): NOT a strict prefix → cold.
    // (A warm hit must leave a non-empty suffix to prefill and a final token to
    // sample; an exact match has neither.)
    std::vector<Tokens> residents = {{10, 20, 30}};
    Pick p = InferenceServer::warm_prefix_pick(residents, all_free(1),
                                               {10, 20, 30});
    EXPECT_EQ(p.slot_id, 0);
    EXPECT_EQ(p.lcp, 0u);  // cold
}

TEST(WarmPrefixPick, DivergenceIsNotReused) {
    // resident [10,20,30] shares [10,20] with the request but diverges at idx 2
    // (30 vs 99). Strict-prefix-append refuses it — reusing would need a rewind.
    std::vector<Tokens> residents = {{10, 20, 30}};
    Pick p = InferenceServer::warm_prefix_pick(residents, all_free(1),
                                               {10, 20, 99, 40});
    EXPECT_EQ(p.slot_id, 0);
    EXPECT_EQ(p.lcp, 0u);  // cold (the slot's stale KV will be cleared)
}

TEST(WarmPrefixPick, ResidentLongerThanRequestIsNotReused) {
    // resident is longer than the request → cannot be a prefix of it → cold.
    std::vector<Tokens> residents = {{10, 20, 30, 40, 50}};
    Pick p = InferenceServer::warm_prefix_pick(residents, all_free(1),
                                               {10, 20, 30});
    EXPECT_EQ(p.slot_id, 0);
    EXPECT_EQ(p.lcp, 0u);
}

TEST(WarmPrefixPick, ActiveMatchingSlotIsSkipped) {
    // slot 0 holds the matching prefix but is BUSY; the only free slot (1) has no
    // resident → cold on slot 1 (we never reuse an active slot's live KV).
    std::vector<Tokens> residents = {{10, 20, 30}, {}};
    std::vector<char> active = {1, 0};  // slot 0 busy, slot 1 free
    Pick p = InferenceServer::warm_prefix_pick(residents, active,
                                               {10, 20, 30, 40});
    EXPECT_EQ(p.slot_id, 1);
    EXPECT_EQ(p.lcp, 0u);
}

TEST(WarmPrefixPick, NoFreeMatchFallsBackToFirstFree) {
    // No free slot's resident is a prefix; fall back to the first FREE slot (1),
    // cold. Slot 0 is busy and must not be chosen even though it matches.
    std::vector<Tokens> residents = {{10, 20, 30}, {7, 7}, {}};
    std::vector<char> active = {1, 0, 0};
    Pick p = InferenceServer::warm_prefix_pick(residents, active,
                                               {10, 20, 30, 40});
    EXPECT_EQ(p.slot_id, 1);  // first free, not the busy match
    EXPECT_EQ(p.lcp, 0u);
}
