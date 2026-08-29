// test_rope_divergence.cpp — the per-slot rows-vs-positions bookkeeping on
// ForwardPassBase (get_rope_pos / note_span_rows_vs_positions /
// has_rope_divergence), tested WITHOUT a model.
//
// The bookkeeping is pure arithmetic over one number the recipe already owns
// (the KV row count), so a stub recipe that does nothing but count rows
// exercises it completely — no GGUF, no Metal, microseconds.
//
// It earns its own file because the bug it gates is not hypothetical: on the
// HTTP server, image request #1 was coherent and #2 was token soup on a Qwen
// (M-RoPE) recipe, with zero OOM markers. The slot is cleared between requests,
// but `note_span_rows_vs_positions` accumulated the second image's delta on top
// of the first image's stale record, so every decode position after it was ~1024
// too low. P6 of docs/plan-qwen35-vision-impl.md. Gemma never saw it: a scalar
// recipe has n_rows == n_pos, so no record is ever written.

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "engine/model.h"
#include "../../src/models/forward_pass_base.h"

namespace {

// Minimal recipe: a per-slot row counter and nothing else. Every graph-building
// pure virtual is a hard error — reaching one would mean the test drifted from
// bookkeeping into inference.
class RowCounterRecipe : public ForwardPassBase {
public:
    // ForwardPassBase's ctor only allocates its metadata ggml context; it stores
    // the model + metadata as references and never dereferences them. So an
    // empty pair is enough to build one, which is what makes this file
    // model-free.
    RowCounterRecipe(const Model& m, const ModelMetadata* md)
        : ForwardPassBase(m, md) {}

    void advance_cache(uint32_t n_tokens, uint32_t slot_idx) override {
        rows_[slot_idx] += n_tokens;
    }
    void clear_slot(uint32_t slot_idx) override { rows_[slot_idx] = 0; }
    void set_cache_pos(uint32_t pos, uint32_t slot_idx) override {
        rows_[slot_idx] = pos;
    }
    uint32_t get_cache_pos(uint32_t slot_idx) const override {
        const auto it = rows_.find(slot_idx);
        return it == rows_.end() ? 0u : it->second;
    }

    ggml_cgraph* build_prefill_graph(const std::vector<int32_t>&, int, uint32_t,
                                     bool) override {
        throw std::logic_error("RowCounterRecipe builds no graphs");
    }
    void clone_slot(uint32_t, uint32_t, uint32_t) override {
        throw std::logic_error("RowCounterRecipe builds no graphs");
    }
    ggml_cgraph* build_decoding_graph(const std::vector<int32_t>&,
                                      const std::vector<uint32_t>&,
                                      const std::vector<int32_t>&) override {
        throw std::logic_error("RowCounterRecipe builds no graphs");
    }

private:
    std::unordered_map<uint32_t, uint32_t> rows_;
};

// One image turn on `slot`: `pre` text rows, then an image of `img_rows` KV rows
// advancing only `img_pos` positions, then `post` text rows. Mirrors the chunk
// sequence drive_prefill_chunks feeds (note BEFORE the span is written).
void feed_image_turn(RowCounterRecipe& fp, uint32_t slot, uint32_t pre,
                     uint32_t img_rows, uint32_t img_pos, uint32_t post) {
    fp.note_span_rows_vs_positions(slot, pre, pre);
    fp.advance_cache(pre, slot);
    fp.note_span_rows_vs_positions(slot, img_rows, img_pos);
    fp.advance_cache(img_rows, slot);
    fp.note_span_rows_vs_positions(slot, post, post);
    fp.advance_cache(post, slot);
}

// An empty model + metadata pair, shared by every case: the base ctor keeps
// references to them and reads neither, so one file-scope pair is safe.
Model& empty_model() { static Model m; return m; }
ModelMetadata& empty_metadata() { static ModelMetadata md; return md; }
RowCounterRecipe make_recipe() {
    return RowCounterRecipe(empty_model(), &empty_metadata());
}

constexpr uint32_t kPre = 15, kImgRows = 1024, kImgPos = 32, kPost = 6;
// rows 15 + 1024 + 6 = 1045; positions 15 + 32 + 6 = 53.
constexpr int32_t kRowsAfterTurn = kPre + kImgRows + kPost;
constexpr int32_t kPosAfterTurn  = kPre + kImgPos  + kPost;

}  // namespace

// A scalar recipe (Gemma: one soft token, one position) never records anything,
// so get_rope_pos is get_cache_pos — byte-identical to the pre-M-RoPE world.
TEST(RopeDivergence, ScalarSpanRecordsNothing) {
    RowCounterRecipe fp = make_recipe();
    feed_image_turn(fp, 0, kPre, /*img_rows=*/256, /*img_pos=*/256, kPost);
    EXPECT_FALSE(fp.has_rope_divergence(0));
    EXPECT_EQ(fp.get_rope_pos(0), static_cast<int32_t>(fp.get_cache_pos(0)));
}

// One M-RoPE image turn: rows and positions diverge by exactly rows − pos.
TEST(RopeDivergence, OneImageSpanSeparatesRowsFromPositions) {
    RowCounterRecipe fp = make_recipe();
    feed_image_turn(fp, 0, kPre, kImgRows, kImgPos, kPost);
    EXPECT_TRUE(fp.has_rope_divergence(0));
    EXPECT_EQ(static_cast<int32_t>(fp.get_cache_pos(0)), kRowsAfterTurn);
    EXPECT_EQ(fp.get_rope_pos(0), kPosAfterTurn);
}

// THE REGRESSION. A cleared slot has outlived its record, so the next image turn
// must start from a clean delta. Before the fix the second turn's delta landed
// on top of the first's (992 + 992) and get_rope_pos came back 992 low — in fact
// NEGATIVE (-939) — which is the server's "image request #1 coherent, #2 token
// soup".
//
// Load-bearing detail: NOTHING may read the record between the clear and the
// next turn. The readers self-heal, so an assertion in that gap silently repairs
// the state and the test passes even unfixed (observed — this case was written
// that way first). The server reads nothing there either, so the strict
// clear→feed→assert order is also the faithful one.
TEST(RopeDivergence, ClearedSlotDoesNotAccumulateTheNextImagesDelta) {
    RowCounterRecipe fp = make_recipe();
    feed_image_turn(fp, 0, kPre, kImgRows, kImgPos, kPost);
    ASSERT_EQ(fp.get_rope_pos(0), kPosAfterTurn);

    fp.clear_slot(0);                                    // what the server does
    feed_image_turn(fp, 0, kPre, kImgRows, kImgPos, kPost);  // no reader between

    EXPECT_EQ(static_cast<int32_t>(fp.get_cache_pos(0)), kRowsAfterTurn);
    EXPECT_EQ(fp.get_rope_pos(0), kPosAfterTurn)
        << "second image turn must not inherit the first turn's delta";
}

// The reader-side self-heal, on its own: a cleared slot reports no divergence
// and position 0. This is what USED to be relied on to keep the writer honest;
// it is necessary but not sufficient, hence the case above.
TEST(RopeDivergence, ClearedSlotReadsAsUndiverged) {
    RowCounterRecipe fp = make_recipe();
    feed_image_turn(fp, 0, kPre, kImgRows, kImgPos, kPost);
    fp.clear_slot(0);
    EXPECT_FALSE(fp.has_rope_divergence(0));
    EXPECT_EQ(fp.get_rope_pos(0), 0);
}

// Same slot, many requests: the invariant is per-turn, not just for turn two.
TEST(RopeDivergence, RepeatedImageRequestsOnOneSlotStayCorrect) {
    RowCounterRecipe fp = make_recipe();
    for (int i = 0; i < 5; ++i) {
        fp.clear_slot(0);
        feed_image_turn(fp, 0, kPre, kImgRows, kImgPos, kPost);
        EXPECT_EQ(fp.get_rope_pos(0), kPosAfterTurn) << "request " << i + 1;
    }
}

// Slots are independent: clearing one must not disturb another's record.
TEST(RopeDivergence, SlotsDoNotShareDivergenceRecords) {
    RowCounterRecipe fp = make_recipe();
    feed_image_turn(fp, 0, kPre, kImgRows, kImgPos, kPost);
    feed_image_turn(fp, 1, kPre, kImgRows, kImgPos, kPost);
    fp.clear_slot(0);
    EXPECT_FALSE(fp.has_rope_divergence(0));
    EXPECT_TRUE(fp.has_rope_divergence(1));
    EXPECT_EQ(fp.get_rope_pos(1), kPosAfterTurn);
}

// A multi-image turn (Phase 7) accumulates span by span WITHIN one live history
// — the `+=` is right, it is only the stale record that must not be inherited.
TEST(RopeDivergence, TwoSpansInOneLiveHistoryAccumulate) {
    RowCounterRecipe fp = make_recipe();
    fp.note_span_rows_vs_positions(0, kImgRows, kImgPos);
    fp.advance_cache(kImgRows, 0);
    fp.note_span_rows_vs_positions(0, kImgRows, kImgPos);
    fp.advance_cache(kImgRows, 0);
    EXPECT_EQ(static_cast<int32_t>(fp.get_cache_pos(0)), 2 * kImgRows);
    EXPECT_EQ(fp.get_rope_pos(0), 2 * kImgPos);
}

// An explicit reset drops the record even while the slot's rows still stand
// (restore_slot's belt-and-braces call).
TEST(RopeDivergence, ResetRopePosDropsALiveRecord) {
    RowCounterRecipe fp = make_recipe();
    feed_image_turn(fp, 0, kPre, kImgRows, kImgPos, kPost);
    ASSERT_TRUE(fp.has_rope_divergence(0));
    fp.reset_rope_pos(0);
    EXPECT_FALSE(fp.has_rope_divergence(0));
    EXPECT_EQ(fp.get_rope_pos(0), kRowsAfterTurn);
}

// Fail-loud contract: a span cannot advance more positions than it wrote rows.
TEST(RopeDivergence, MorePositionsThanRowsFailsLoud) {
    RowCounterRecipe fp = make_recipe();
    EXPECT_THROW(fp.note_span_rows_vs_positions(0, 8, 9), std::runtime_error);
}
