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
// RECIPE-AGNOSTIC BY CONSTRUCTION (2026-09-01). The seam is the tensor NAME
// `kq_soft.<il>`, which layers/attention.cpp assigns inside build_attn_mha —
// the single funnel every attention builder (plain/gated × prefill/decode)
// passes through. So this gate must not be able to *only* ask the question of
// one recipe. It previously could: it constructed `Qwen36ForwardPass` directly
// and derived attention layers from the `qwen35moe.full_attention_interval`
// metadata key, so it was structurally incapable of taking the cross-family
// comparison — the same defect architecture.md §10 records for the July
// QINF_BATCH_IDENTICAL control. Two changes fix that:
//
//   * the recipe comes from `create_forward_pass` (the registry factory), so
//     any registered architecture is hostable here; and
//   * the attention layers are discovered by SCANNING THE BUILT GRAPH for
//     `kq_soft.<il>` rather than by reading an architecture-specific hparam.
//     That is the seam's own definition, needs no per-family knowledge, and
//     automatically excludes blocks held out of the decode stack (a qwen35
//     GGUF's trailing NextN/MTP head) because they build no nodes.
//
// Legs are env-gated and each self-skips when its model is absent, so a green
// run with skips is normal — check the reported Skipped list, not just the
// failures. Set exactly one at a time: these are 5–13 GB models and load peak
// is ~2× model size (architecture.md §5).
//
//   QWEN36_MODEL_PATH  — qwen35moe hybrid (the pinned lens model)
//   QWEN35_MODEL_PATH  — qwen35 hybrid   (hosts the Qwen 3.5 and 3.8 releases)
//   QWEN3_MODEL_PATH   — qwen3 pure transformer
//   GEMMA3_MODEL_PATH  — gemma3 (interleaved local/global attention)
//   GEMMA4_MODEL_PATH  — gemma4 (the cross-family falsifier)
//
// A passing leg answers the MECHANICAL question only — "does the tap return
// live attention rows on this recipe". It says nothing about whether the lens
// CONSTANTS transfer (citations pinned to L3H13, coverage to layer 11 at
// ≥0.705, all probed on Qwen 3.6); that is a probe campaign, not a check, and
// architecture.md §12 is explicit that there are no lens claims for Gemma.

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/forward_pass_base.h"

namespace {

// The env vars this gate knows about, in the order they are reported.
struct Leg {
    const char* env;
    const char* what;
};
const Leg kLegs[] = {
    {"QWEN36_MODEL_PATH", "qwen35moe hybrid (pinned lens model)"},
    {"QWEN35_MODEL_PATH", "qwen35 hybrid (Qwen 3.5 / 3.8)"},
    {"QWEN3_MODEL_PATH",  "qwen3 pure transformer"},
    {"GEMMA3_MODEL_PATH", "gemma3"},
    {"GEMMA4_MODEL_PATH", "gemma4"},
};

// The parameter space is EVERY leg, unconditionally — never "the legs whose env
// var happens to be set". gtest_discover_tests enumerates by running this binary
// at BUILD time, so an env-dependent parameter list would make the ctest test
// list depend on the environment of the build, and a leg would vanish from the
// suite rather than appear as a skip. A recipe that silently isn't in the list
// is the failure mode this whole file was rewritten to remove; each leg instead
// self-skips at run time, naming itself in the Skipped list.
std::vector<std::string> leg_params() {
    std::vector<std::string> out;
    for (const Leg& l : kLegs) out.push_back(l.env);
    return out;
}

// One model resident at a time. These are 5–13 GB files on a machine where
// memory, not code, is the binding constraint (architecture.md §5: load peak is
// ~2× model size), so a second load must never overlap the first.
class ModelCache {
public:
    static Model* get(const std::string& path) {
        if (path.empty()) return nullptr;
        if (path != path_) {
            model_.reset();          // release before loading the next
            register_builtin_models();
            auto m = std::make_unique<Model>();
            m->load_metadata(path);
            m->load_tensors();
            model_ = std::move(m);
            path_  = path;
        }
        return model_.get();
    }
    static void clear() { model_.reset(); path_.clear(); }
private:
    static std::unique_ptr<Model> model_;
    static std::string path_;
};
std::unique_ptr<Model> ModelCache::model_ = nullptr;
std::string ModelCache::path_;

// The cache MUST be released while main() is still running. Left to static
// destruction it races ggml-metal's own static device vector, and losing that
// race aborts the process AFTER every test has passed:
//   ggml_metal_device_free: GGML_ASSERT([rsets->data count] == 0) failed
// — the model's Metal buffers are still resident when the device is torn down.
// A green gtest summary followed by a non-zero exit is exactly the shape of
// failure that reads as "flaky infrastructure", so it is closed here.
class ModelCacheEnv : public ::testing::Environment {
public:
    void TearDown() override { ModelCache::clear(); }
};
const bool kModelCacheEnvRegistered =
    (::testing::AddGlobalTestEnvironment(new ModelCacheEnv), true);

// A short deterministic prompt is enough — the tap machinery is prompt-agnostic.
// Ids 1..12 are valid in every vocabulary here and their text is irrelevant.
const std::vector<int32_t> kPrompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

constexpr uint32_t kCtx = 512;

}  // namespace

class ForwardPassTapTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string env = GetParam();
        const char* p = std::getenv(env.c_str());
        if (!p || !*p) GTEST_SKIP() << env << " not set — skipping this leg";
        path_  = p;
        model_ = ModelCache::get(path_);
        ASSERT_NE(model_, nullptr) << env << " named '" << path_
                                   << "' but no model was loaded";
    }

    std::unique_ptr<ForwardPassBase> make_fp() const {
        return create_forward_pass(*model_, &model_->get_metadata(), kCtx, 1);
    }

    // Run one greedy decode step over `prompt` on a freshly-cleared slot,
    // capturing the last-position logits (and, if taps are armed on `fp`, the
    // tapped rows). Independent state per call: clear → prefill → one decode.
    static std::vector<float> decode_once(
            ForwardPassBase& fp, ggml_backend_sched_t sched,
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

    // Which layers of THIS recipe's decode graph carry a tap, discovered from
    // the graph itself (the seam is the tensor name). No architecture hparam is
    // read, so the check cannot be silently gated to one family.
    std::vector<int> discover_tap_layers(ForwardPassBase& fp,
                                         ggml_backend_sched_t sched) const {
        fp.clear_slot(0);
        fp.set_cache_pos(0, 0);
        fp.run_prefill(kPrompt, 0, 0, sched);

        std::vector<int32_t>  tokens    = {kPrompt.back()};
        std::vector<uint32_t> slots     = {0};
        std::vector<int32_t>  positions = {(int)fp.get_cache_pos(0)};
        ggml_cgraph* gf = fp.build_decoding_graph(tokens, slots, positions);

        std::vector<int> out;
        const uint32_t n_block = model_->get_metadata().block_count;
        for (uint32_t il = 0; il < n_block; ++il) {
            const std::string nm = "kq_soft." + std::to_string(il);
            if (ggml_graph_get_tensor(gf, nm.c_str())) out.push_back((int)il);
        }
        return out;
    }

    std::string path_;
    Model*      model_ = nullptr;
};

// ── 0. The seam exists on this recipe at all ─────────────────────────────────
// Reported first and separately: "no kq_soft in the decode graph" is a
// different fact from "the rows are wrong", and only the former means the
// recipe cannot host the lens.
TEST_P(ForwardPassTapTest, RecipeMaterializesAttentionRows) {
    auto fp = make_fp();
    auto layers = discover_tap_layers(*fp, model_->get_scheduler());
    const auto& meta = model_->get_metadata();
    std::cout << "[tap] " << GetParam() << " arch=" << meta.architecture
              << " blocks=" << meta.block_count
              << " n_head=" << meta.attention_head_count
              << " tapped attention layers=" << layers.size() << ":";
    for (int il : layers) std::cout << " " << il;
    std::cout << std::endl;
    EXPECT_FALSE(layers.empty())
        << "no `kq_soft.<il>` tensor in the decode graph of arch '"
        << meta.architecture << "' — this recipe does not materialize attention "
           "and cannot host the lens tap";
}

// ── 1. Byte-inert off ────────────────────────────────────────────────────────
TEST_P(ForwardPassTapTest, TapOffByteIdentical) {
    ggml_backend_sched_t sched = model_->get_scheduler();

    auto fp = make_fp();
    // Disarmed (today's decode path).
    std::vector<float> logits_off = decode_once(*fp, sched, kPrompt, nullptr);

    auto layers = discover_tap_layers(*fp, sched);
    if (layers.empty())
        GTEST_SKIP() << "recipe materializes no attention rows (see "
                        "RecipeMaterializesAttentionRows)";
    // Arm EVERY attention layer, not just the two frozen lens layers: the
    // liveness claim is about marking outputs in general, and a recipe with
    // interleaved attention (Gemma) exercises more of the graph this way.
    fp->set_attention_taps(layers);

    std::vector<ForwardPassBase::AttentionTap> tapped;
    std::vector<float> logits_on = decode_once(*fp, sched, kPrompt, &tapped);

    ASSERT_EQ(logits_off.size(), logits_on.size());
    // Byte-for-byte: marking outputs is a liveness-only change to the graph.
    EXPECT_EQ(logits_off, logits_on)
        << "arming the attention tap perturbed decode logits on arch '"
        << model_->get_metadata().architecture << "' — the tap is NOT byte-inert; "
           "this violates the P1/A1 off-path identity gate";
    EXPECT_EQ(tapped.size(), layers.size());
}

// ── 2. Armed rows are live softmax distributions ─────────────────────────────
TEST_P(ForwardPassTapTest, TapRowsAreRealSoftmax) {
    const auto& meta = model_->get_metadata();
    ggml_backend_sched_t sched = model_->get_scheduler();

    auto fp = make_fp();
    auto layers = discover_tap_layers(*fp, sched);
    if (layers.empty())
        GTEST_SKIP() << "recipe materializes no attention rows (see "
                        "RecipeMaterializesAttentionRows)";
    fp->set_attention_taps(layers);

    std::vector<ForwardPassBase::AttentionTap> tapped;
    decode_once(*fp, sched, kPrompt, &tapped);

    ASSERT_EQ(tapped.size(), layers.size());
    // prefill wrote 0..N-1; the decode step writes the query token's own K/V
    // before the softmax, so the row attends over N+1 positions.
    const int expect_kv = (int)kPrompt.size() + 1;
    for (size_t i = 0; i < tapped.size(); ++i) {
        const auto& tap = tapped[i];
        EXPECT_EQ(tap.layer, layers[i]);
        EXPECT_EQ(tap.n_head, (int)meta.attention_head_count);
        EXPECT_EQ(tap.n_kv, expect_kv);
        ASSERT_EQ(tap.rows.size(), (size_t)tap.n_kv * tap.n_head);
        for (int h = 0; h < tap.n_head; ++h) {
            double s = 0.0;
            for (int j = 0; j < tap.n_kv; ++j) s += tap.rows[(size_t)h * tap.n_kv + j];
            EXPECT_NEAR(s, 1.0, 1e-3)
                << "arch " << meta.architecture << " layer " << tap.layer
                << " head " << h
                << " softmax row does not sum to 1 — tapped buffer is not the "
                   "live attention distribution";
        }
    }
}

// ── 3. Fail-loud on a non-attention layer ────────────────────────────────────
TEST_P(ForwardPassTapTest, UnknownTapLayerFailsLoud) {
    auto fp = make_fp();
    fp->set_attention_taps({99999});  // no such attention layer ⇒ no kq_soft.99999

    std::vector<int32_t>  tokens    = {kPrompt.back()};
    std::vector<uint32_t> slots     = {0};
    std::vector<int32_t>  positions = {0};
    fp->clear_slot(0);
    fp->set_cache_pos(0, 0);
    fp->run_prefill(kPrompt, 0, 0, model_->get_scheduler());
    positions[0] = (int)fp->get_cache_pos(0);

    ggml_cgraph* gf = fp->build_decoding_graph(tokens, slots, positions);
    EXPECT_THROW(fp->mark_attention_taps(gf), std::runtime_error);
}

// ── 4. D8: is a disarmed tap inert for the NEXT request? ─────────────────────
// TapOffByteIdentical proves the graph is byte-identical when the tap is
// disarmed *within one decode*. That is a weaker property than a server needs.
// The consolidation loop found (2026-09-02) that running POST /v1/extract makes
// the NEXT /v1/completions differ from the same request run before it, while an
// intervening ordinary completion does not — and only the FIRST request after an
// extract is affected, reproducibly. That is D8.
//
// These two tests localize it to the engine or exonerate the engine:
//   * TapArmedThenDisarmedIsInertForNextDecode — arm, run a tapped pass on a
//     DIFFERENT prompt, disarm, then re-run the original decode. Its logits must
//     equal the pre-tap run.
//   * ExtraUntappedPassIsInertForNextDecode — the CONTROL: identical sequence
//     with the taps never armed. If the control passes and the armed one fails,
//     the tap (not merely the extra pass) is the cause.
TEST_P(ForwardPassTapTest, TapArmedThenDisarmedIsInertForNextDecode) {
    ggml_backend_sched_t sched = model_->get_scheduler();
    auto fp = make_fp();

    const std::vector<float> before = decode_once(*fp, sched, kPrompt, nullptr);

    auto layers = discover_tap_layers(*fp, sched);
    if (layers.empty()) GTEST_SKIP() << "recipe materializes no attention rows";

    // A tapped pass over a DIFFERENT prompt, as /v1/extract does.
    const std::vector<int32_t> other = {21, 22, 23, 24, 25, 26, 27, 28};
    fp->set_attention_taps(layers);
    std::vector<ForwardPassBase::AttentionTap> taps;
    decode_once(*fp, sched, other, &taps);
    ASSERT_FALSE(taps.empty());
    fp->set_attention_taps({});          // disarm, exactly as run_lens_tapped_decode does

    const std::vector<float> after = decode_once(*fp, sched, kPrompt, nullptr);

    ASSERT_EQ(before.size(), after.size());
    EXPECT_EQ(before, after)
        << "a disarmed attention tap perturbed the NEXT decode on arch '"
        << model_->get_metadata().architecture << "' — D8. The tap seam claims to "
           "be byte-inert when disarmed (architecture.md §12) and receipts-grade "
           "determinism claims byte-reproducible greedy decode at B=1 (§11); this "
           "shows the inertness does not survive across passes.";
}

TEST_P(ForwardPassTapTest, ExtraUntappedPassIsInertForNextDecode) {
    ggml_backend_sched_t sched = model_->get_scheduler();
    auto fp = make_fp();

    const std::vector<float> before = decode_once(*fp, sched, kPrompt, nullptr);
    const std::vector<int32_t> other = {21, 22, 23, 24, 25, 26, 27, 28};
    decode_once(*fp, sched, other, nullptr);          // same extra pass, NO taps
    const std::vector<float> after = decode_once(*fp, sched, kPrompt, nullptr);

    ASSERT_EQ(before.size(), after.size());
    EXPECT_EQ(before, after)
        << "an ordinary intervening decode perturbed the next decode — the "
           "perturbation is NOT tap-specific, so D8's cause is more general "
           "(scheduler/galloc replanning across shapes, not the tap seam).";
}

// ── 5. D8, refined: does a LENS-SHAPED pass perturb the next decode? ─────────
// Tests 4 above both passed, which exonerated the tap — but they were weak in a
// way that matters: both passes used the SAME decode_once path, hence the same
// graph shape, so galloc never had to re-plan. The lens driver instead runs its
// own long prefill plus MANY single-slot decode steps with a growing n_kv, a
// shape the server's batched decode path never builds. Alternating shapes on one
// scheduler is a documented hazard here (server-image-multirequest-bug.md: "the
// image-prefill graph runs on the SAME scheduler as text prefill and decode;
// galloc re-plans across those alternating shapes"). This reproduces that shape
// alternation WITHOUT any tap, to separate "the tap perturbs" from "an unusual
// graph shape perturbs".
TEST_P(ForwardPassTapTest, LensShapedUntappedPassIsInertForNextDecode) {
    ggml_backend_sched_t sched = model_->get_scheduler();
    auto fp = make_fp();

    const std::vector<float> before = decode_once(*fp, sched, kPrompt, nullptr);

    // A long prefill + a multi-step decode loop, exactly the lens driver's shape,
    // but with the tap never armed.
    std::vector<int32_t> longer;
    for (int i = 0; i < 120; ++i) longer.push_back(30 + (i % 900));
    fp->clear_slot(0);
    fp->set_cache_pos(0, 0);
    fp->run_prefill(longer, 0, 0, sched);
    int32_t cur = longer.back();
    for (int step = 0; step < 24; ++step) {
        std::vector<int32_t>  tks = {cur};
        std::vector<uint32_t> sl  = {0};
        std::vector<int32_t>  pos = {(int)fp->get_cache_pos(0)};
        ggml_cgraph* gf = fp->build_decoding_graph(tks, sl, pos);
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf);
        fp->set_decode_inputs(gf, tks, sl, pos);
        ggml_backend_sched_graph_compute(sched, gf);
        std::vector<float> lg = fp->get_output_logits(gf);
        int32_t best = 0;
        for (int j = 1; j < (int)lg.size(); ++j) if (lg[j] > lg[best]) best = j;
        cur = best;
        fp->advance_cache(1, 0);
    }

    const std::vector<float> after = decode_once(*fp, sched, kPrompt, nullptr);
    ASSERT_EQ(before.size(), after.size());
    EXPECT_EQ(before, after)
        << "a lens-SHAPED but UNTAPPED pass perturbed the next decode — so D8 is "
           "graph-shape/galloc driven, not caused by the attention tap.";
}

INSTANTIATE_TEST_SUITE_P(
    Recipes, ForwardPassTapTest, ::testing::ValuesIn(leg_params()),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; });
