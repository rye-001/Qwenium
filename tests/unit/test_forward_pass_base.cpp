// test_forward_pass_base.cpp — P1/A1 gate for the Qemmi-Lens attention tap.
//
// The lens tap (ForwardPassBase::set_attention_taps / mark_attention_taps /
// get_attention_taps, docs/plan-qemmi-lens.md) exposes each requested attention
// layer's post-softmax row `kq_soft.<il>` as a retained graph output. It is the
// productization of the QDOCS probe's interpose; these tests are the gate the
// plan asks for:
//
//   1. TapOffByteIdentical — decode logits are byte-for-byte identical with the
//      tap disarmed vs armed. Marking an existing node as an output adds no
//      compute (same argument as D3 set_output_hidden), so the default path is
//      unchanged. This is the load-bearing "byte-inert off" claim.
//   2. TapRowsAreRealSoftmax — armed rows exist, have the right shape
//      ([n_kv, n_head]), and every head's row sums to ~1.0 (proves live data,
//      not a reused/garbage galloc buffer — the probe's Q1).
//   3. UnknownTapLayerFailsLoud — arming a non-attention layer throws a
//      contract error naming the tensor (fail-loud at the module boundary).
//
// Hosted on the Qwen3.6 recipe (the pinned lens model); the seam itself is
// recipe-agnostic. Requires QWEN36_MODEL_PATH → a qwen35moe GGUF; skips without.

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include "../../src/core/model.h"
#include "../../src/models/qwen36.h"
#include "../../src/models/model_registry.h"

static std::string get_model_path() {
    const char* p = std::getenv("QWEN36_MODEL_PATH");
    return p ? std::string(p) : "";
}

#define SKIP_IF_NO_MODEL()                                             \
    do {                                                               \
        if (get_model_path().empty())                                  \
            GTEST_SKIP() << "QWEN36_MODEL_PATH not set — skipping";    \
    } while (0)

class ForwardPassTapTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (get_model_path().empty()) return;
        register_builtin_models();
        model_ = std::make_unique<Model>();
        model_->load_metadata(get_model_path());
        model_->load_tensors();
    }
    static void TearDownTestSuite() { model_.reset(); }
    static std::unique_ptr<Model> model_;

    // Attention layers of this qwen36 GGUF, derived from metadata exactly as the
    // recipe/probe does (il % fai == fai-1). The frozen lens constants are the
    // citation head's layer 3 and the coverage source layer 11.
    static std::vector<int> attention_layers(const ModelMetadata& meta) {
        const uint32_t fai = meta.raw_kv.get_uint32("qwen35moe.full_attention_interval");
        std::vector<int> out;
        for (uint32_t il = 0; il < meta.block_count; ++il)
            if (fai > 0 && (il % fai) == (fai - 1)) out.push_back((int)il);
        return out;
    }

    // Run one greedy decode step over `prompt` on a freshly-cleared slot,
    // capturing the last-position logits (and, if taps are armed on `fp`, the
    // tapped rows). Independent state per call: clear → prefill → one decode.
    static std::vector<float> decode_once(
            Qwen36ForwardPass& fp, ggml_backend_sched_t sched,
            const std::vector<int32_t>& prompt,
            std::vector<ForwardPassBase::AttentionTap>* taps_out) {
        fp.clear_slot(0);
        fp.set_cache_pos(0, 0);
        fp.run_prefill(prompt, 0, 0, sched);

        std::vector<int32_t>  tokens    = {prompt.back()};
        std::vector<uint32_t> slots     = {0};
        std::vector<int32_t>  positions = {(int)fp.get_cache_pos(0)};

        ggml_cgraph* gf = fp.build_decoding_graph(tokens, slots, positions);
        fp.mark_attention_taps(gf);        // no-op when the tap set is empty
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf);
        fp.set_decode_inputs(gf, tokens, slots, positions);
        ggml_backend_sched_graph_compute(sched, gf);

        if (taps_out) *taps_out = fp.get_attention_taps(gf);
        return fp.get_output_logits(gf);
    }
};

std::unique_ptr<Model> ForwardPassTapTest::model_ = nullptr;

// A short deterministic prompt is enough — the tap machinery is prompt-agnostic.
static const std::vector<int32_t> kPrompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

// ── 1. Byte-inert off ────────────────────────────────────────────────────────
TEST_F(ForwardPassTapTest, TapOffByteIdentical) {
    SKIP_IF_NO_MODEL();
    const auto& meta = model_->get_metadata();
    ggml_backend_sched_t sched = model_->get_scheduler();

    Qwen36ForwardPass fp(*model_, &meta, 512, 1);

    // Disarmed (today's decode path).
    std::vector<float> logits_off = decode_once(fp, sched, kPrompt, nullptr);

    // Armed on the frozen lens layers, if this GGUF has them.
    auto attn = attention_layers(meta);
    std::vector<int> taps;
    for (int want : {3, 11})
        if (std::find(attn.begin(), attn.end(), want) != attn.end()) taps.push_back(want);
    if (taps.empty())
        GTEST_SKIP() << "no frozen lens layers (3/11) in this GGUF's attention set";
    fp.set_attention_taps(taps);

    std::vector<ForwardPassBase::AttentionTap> tapped;
    std::vector<float> logits_on = decode_once(fp, sched, kPrompt, &tapped);

    ASSERT_EQ(logits_off.size(), logits_on.size());
    // Byte-for-byte: marking outputs is a liveness-only change to the graph.
    EXPECT_EQ(logits_off, logits_on)
        << "arming the attention tap perturbed decode logits — the tap is NOT "
           "byte-inert; this violates the P1/A1 off-path identity gate";
    EXPECT_EQ(tapped.size(), taps.size());
}

// ── 2. Armed rows are live softmax distributions ─────────────────────────────
TEST_F(ForwardPassTapTest, TapRowsAreRealSoftmax) {
    SKIP_IF_NO_MODEL();
    const auto& meta = model_->get_metadata();
    ggml_backend_sched_t sched = model_->get_scheduler();

    auto attn = attention_layers(meta);
    std::vector<int> taps;
    for (int want : {3, 11})
        if (std::find(attn.begin(), attn.end(), want) != attn.end()) taps.push_back(want);
    if (taps.empty())
        GTEST_SKIP() << "no frozen lens layers (3/11) in this GGUF's attention set";

    Qwen36ForwardPass fp(*model_, &meta, 512, 1);
    fp.set_attention_taps(taps);

    std::vector<ForwardPassBase::AttentionTap> tapped;
    decode_once(fp, sched, kPrompt, &tapped);

    ASSERT_EQ(tapped.size(), taps.size());
    // prefill wrote 0..N-1; the decode step writes the query token's own K/V
    // before the softmax, so the row attends over N+1 positions.
    const int expect_kv = (int)kPrompt.size() + 1;
    for (size_t i = 0; i < tapped.size(); ++i) {
        const auto& tap = tapped[i];
        EXPECT_EQ(tap.layer, taps[i]);
        EXPECT_EQ(tap.n_head, (int)meta.attention_head_count);
        EXPECT_EQ(tap.n_kv, expect_kv);
        ASSERT_EQ(tap.rows.size(), (size_t)tap.n_kv * tap.n_head);
        for (int h = 0; h < tap.n_head; ++h) {
            double s = 0.0;
            for (int j = 0; j < tap.n_kv; ++j) s += tap.rows[(size_t)h * tap.n_kv + j];
            EXPECT_NEAR(s, 1.0, 1e-3)
                << "layer " << tap.layer << " head " << h
                << " softmax row does not sum to 1 — tapped buffer is not the "
                   "live attention distribution";
        }
    }
}

// ── 3. Fail-loud on a non-attention layer ────────────────────────────────────
TEST_F(ForwardPassTapTest, UnknownTapLayerFailsLoud) {
    SKIP_IF_NO_MODEL();
    const auto& meta = model_->get_metadata();
    Qwen36ForwardPass fp(*model_, &meta, 512, 1);
    fp.set_attention_taps({99999});  // no such attention layer ⇒ no kq_soft.99999

    std::vector<int32_t>  tokens    = {kPrompt.back()};
    std::vector<uint32_t> slots     = {0};
    std::vector<int32_t>  positions = {0};
    fp.clear_slot(0);
    fp.set_cache_pos(0, 0);
    fp.run_prefill(kPrompt, 0, 0, model_->get_scheduler());
    positions[0] = (int)fp.get_cache_pos(0);

    ggml_cgraph* gf = fp.build_decoding_graph(tokens, slots, positions);
    EXPECT_THROW(fp.mark_attention_taps(gf), std::runtime_error);
}
