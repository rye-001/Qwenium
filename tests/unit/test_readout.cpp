// test_readout.cpp — co-located test for src/engine/readout.cpp
// (docs/plan-readout-api.md, phase A1).
//
// readout::prefill / decode are an EXTRACTION of the sequence the lens wrote by
// hand (../Qemmi-lens-server/src/server/server_lens.cpp). So the gate is
// bit-identity against that hand-written sequence, run here verbatim, on every
// recipe leg — Qwen AND Gemma (CLAUDE.md cross-family rule):
//   QWEN36_MODEL_PATH  qwen35moe hybrid   QWEN35_MODEL_PATH  qwen35 hybrid
//   QWEN3_MODEL_PATH   qwen3 transformer  GEMMA3_MODEL_PATH  gemma3
//   GEMMA4_MODEL_PATH  gemma4             GEMMA4_MOE_MODEL_PATH (routing legs)
// Each leg self-skips when its variable is unset. One model resident at a time.
//
// And the reason the API exists: every override a Spec sets is restored after
// the call — also when the call throws.

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "engine/model.h"
#include "engine/readout.h"
#include "engine/graph_compute.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/forward_pass_base.h"
#include "../../src/layers/routing_trace.h"

namespace {

namespace ro = qinf::engine::readout;
using AttnImpl = ForwardPassBase::AttnImpl;
using Tap = ForwardPassBase::AttentionTap;

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
            path_ = path;
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

// Released while main() runs: left to static destruction it races ggml-metal's
// device teardown (see test_forward_pass_base.cpp, ModelCacheEnv).
class ModelCacheEnv : public ::testing::Environment {
public:
    void TearDown() override { ModelCache::clear(); }
};
const bool kModelCacheEnvRegistered = (::testing::AddGlobalTestEnvironment(new ModelCacheEnv), true);

constexpr uint32_t kCtx = 512;

std::vector<int32_t> prompt_of(int n, int32_t first = 1) {
    std::vector<int32_t> p;
    for (int32_t t = first; t < first + n; ++t) p.push_back(t);
    return p;
}

bool same_floats(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void expect_same_taps(const std::vector<Tap>& want, const std::vector<Tap>& got, const std::string& what) {
    ASSERT_EQ(want.size(), got.size()) << what << ": tap count";
    for (size_t i = 0; i < want.size(); ++i) {
        EXPECT_EQ(want[i].layer, got[i].layer) << what << ": tap " << i << " layer";
        EXPECT_EQ(want[i].n_kv, got[i].n_kv) << what << ": tap " << i << " n_kv";
        EXPECT_EQ(want[i].n_q, got[i].n_q) << what << ": tap " << i << " n_q";
        EXPECT_EQ(want[i].heads, got[i].heads) << what << ": tap " << i << " heads";
        EXPECT_TRUE(same_floats(want[i].rows, got[i].rows))
            << what << ": tap " << i << " (layer " << want[i].layer << ") rows expected bitwise-identical";
    }
}

}  // namespace

class ReadoutTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const char* p = std::getenv(GetParam().c_str());
        if (!p || !*p) GTEST_SKIP() << GetParam() << " not set — skipping this leg";
        model_ = ModelCache::get(p);
        ASSERT_NE(model_, nullptr);
        fp_ = create_forward_pass(*model_, &model_->get_metadata(), kCtx, 1);
        sched_ = model_->get_scheduler();
        tap_layers_ = discover_tap_layers();
        ASSERT_FALSE(tap_layers_.empty()) << GetParam() << ": no kq_soft.<il> in the prefill graph";
    }

    // Tap layers from the prefill graph itself (the seam is the tensor name).
    std::vector<int> discover_tap_layers() {
        fp_->clear_slot(0); fp_->set_cache_pos(0, 0);
        ggml_cgraph* gf = fp_->build_prefill_graph(prompt_of(8), 0, 0, false);
        std::vector<int> out;
        for (uint32_t il = 0; il < model_->get_metadata().block_count; ++il)
            if (ggml_graph_get_tensor(gf, ("kq_soft." + std::to_string(il)).c_str())) out.push_back((int)il);
        ggml_backend_sched_reset(sched_);
        fp_->clear_slot(0);
        return out;
    }

    // The lens's hand-written tapped prefill (run_lens_locate / run_lens_verify),
    // verbatim, with its restore done by hand.
    std::vector<Tap> raw_prefill(const std::vector<int32_t>& toks, int pos, int cut,
                                 std::vector<int> layers, std::vector<int> heads, bool want_logits,
                                 std::vector<float>* logits, RoutingTrace* capture = nullptr) {
        fp_->set_truncate_after_layer(cut);
        fp_->set_attention_taps(layers, heads);
        fp_->set_prefill_attn_impl(AttnImpl::Materialized);
        fp_->set_routing_capture(capture);
        ggml_cgraph* gf = fp_->build_prefill_graph(toks, pos, 0, want_logits);
        fp_->mark_attention_taps(gf);
        if (capture) fp_->mark_moe_routing(gf);
        fp_->alloc_readback_graph(sched_, gf);
        fp_->set_prefill_inputs(gf, toks, pos);
        qinf::engine::require_compute_success(ggml_backend_sched_graph_compute(sched_, gf), "raw_prefill");
        if (capture) fp_->read_moe_routing(gf, *capture, pos);
        std::vector<Tap> taps;
        if (!layers.empty()) taps = fp_->get_attention_taps(gf);
        if (logits) *logits = fp_->get_output_logits(gf);
        fp_->advance_cache((uint32_t)toks.size(), 0);
        fp_->set_attention_taps({});
        fp_->set_truncate_after_layer(-1);
        fp_->set_routing_capture(nullptr);
        return taps;
    }

    // The lens's hand-written tapped decode step (run_lens_tapped_decode).
    std::vector<Tap> raw_decode(int32_t token, std::vector<int> layers, std::vector<float>* logits) {
        fp_->set_attention_taps(layers);
        fp_->set_sparse_decode_ids({});
        std::vector<int32_t> tks = {token};
        std::vector<uint32_t> slots = {0};
        std::vector<int32_t> positions = {fp_->get_rope_pos(0)};
        ggml_cgraph* gf = fp_->build_decoding_graph(tks, slots, positions);
        fp_->mark_attention_taps(gf);
        fp_->alloc_readback_graph(sched_, gf);
        fp_->set_decode_inputs(gf, tks, slots, positions);
        qinf::engine::require_compute_success(ggml_backend_sched_graph_compute(sched_, gf), "raw_decode");
        std::vector<Tap> taps = fp_->get_attention_taps(gf);
        *logits = fp_->get_output_logits(gf);
        fp_->advance_cache(1, 0);
        fp_->set_attention_taps({});
        return taps;
    }

    void fresh() { fp_->clear_slot(0); fp_->set_cache_pos(0, 0); }

    Model* model_ = nullptr;
    std::unique_ptr<ForwardPassBase> fp_;
    ggml_backend_sched_t sched_ = nullptr;
    std::vector<int> tap_layers_;
};

// Full stack, two tap layers (request order, not layer order — the lens arms
// {27, 11} on Qwen 3.8), last-row logits.
TEST_P(ReadoutTest, PrefillIsBitwiseTheHandWrittenSequence) {
    const std::vector<int32_t> p = prompt_of(40);
    const std::vector<int> layers = {tap_layers_.back(), tap_layers_.front()};
    fresh();
    std::vector<float> want_logits;
    const std::vector<Tap> want = raw_prefill(p, 0, -1, layers, {}, true, &want_logits);
    const uint32_t want_pos = fp_->get_cache_pos(0);

    fresh();
    ro::Spec s;
    s.tap_layers = layers;
    s.logits = true;
    s.prefill_attn = AttnImpl::Materialized;
    const ro::Result got = ro::prefill(*fp_, sched_, 0, 0, p, s);
    expect_same_taps(want, got.taps, GetParam() + " full-stack prefill");
    EXPECT_TRUE(same_floats(want_logits, got.logits)) << GetParam() << ": logits expected bitwise-identical";
    ASSERT_GT(got.logits_width, 0u);
    EXPECT_EQ(0u, got.logits.size() % got.logits_width) << "logits expected whole rows";
    EXPECT_EQ(want_pos, fp_->get_cache_pos(0)) << "cache position after the prefill";
}

// The locate shape: an untapped, truncated document pass, then a truncated,
// head-selected tapped pass over the rows after it.
TEST_P(ReadoutTest, SplitTruncatedHeadSelectedPrefillIsBitwiseTheHandWrittenSequence) {
    const std::vector<int32_t> doc = prompt_of(30), rows = prompt_of(10, 31);
    const int cut = tap_layers_.front();
    const std::vector<int> heads = {1, 0};
    fresh();
    raw_prefill(doc, 0, cut, {}, {}, false, nullptr);
    const std::vector<Tap> want = raw_prefill(rows, 30, cut, {cut}, heads, false, nullptr);

    fresh();
    ro::Spec doc_spec;
    doc_spec.last_layer = cut;
    doc_spec.prefill_attn = AttnImpl::Materialized;
    ro::prefill(*fp_, sched_, 0, 0, doc, doc_spec);
    ro::Spec tap_spec = doc_spec;
    tap_spec.tap_layers = {cut};
    tap_spec.tap_heads = heads;
    const ro::Result got = ro::prefill(*fp_, sched_, 0, 30, rows, tap_spec);
    expect_same_taps(want, got.taps, GetParam() + " split truncated head-selected prefill");
    EXPECT_EQ(40u, fp_->get_cache_pos(0));
    EXPECT_EQ(-1, fp_->truncate_after_layer()) << "the layer cut must not leak past the call";
}

// One tapped decode step after an identical prefill.
TEST_P(ReadoutTest, DecodeIsBitwiseTheHandWrittenSequence) {
    const std::vector<int32_t> p = prompt_of(24);
    const std::vector<int> layers = {tap_layers_.front(), tap_layers_.back()};
    fresh();
    fp_->run_prefill(p, 0, 0, sched_);
    std::vector<float> want_logits;
    const std::vector<Tap> want = raw_decode(p.back(), layers, &want_logits);

    fresh();
    fp_->run_prefill(p, 0, 0, sched_);
    ro::Spec s;
    s.tap_layers = layers;
    s.logits = true;
    const ro::Result got = ro::decode(*fp_, sched_, 0, fp_->get_rope_pos(0), p.back(), s);
    expect_same_taps(want, got.taps, GetParam() + " decode");
    EXPECT_TRUE(same_floats(want_logits, got.logits)) << GetParam() << ": decode logits expected bitwise-identical";
    EXPECT_EQ(25u, fp_->get_cache_pos(0));
}

// Every override is put back — after a normal call and after a throwing one.
TEST_P(ReadoutTest, OverridesAreRestoredAfterReturnAndAfterThrow) {
    fp_->set_truncate_after_layer(-1);
    fp_->set_attention_taps({});
    fp_->set_prefill_attn_impl(AttnImpl::Materialized);
    auto unchanged = [&](const char* when) {
        EXPECT_EQ(-1, fp_->truncate_after_layer()) << when;
        EXPECT_TRUE(fp_->attention_taps().empty()) << when;
        EXPECT_TRUE(fp_->attention_tap_heads().empty()) << when;
        EXPECT_EQ(AttnImpl::Materialized, fp_->prefill_attn_impl()) << when;
        EXPECT_EQ(nullptr, fp_->routing_capture()) << when;
        EXPECT_EQ(nullptr, fp_->routing_replay()) << when;
    };
    fresh();
    ro::Spec s;
    s.last_layer = tap_layers_.front();
    s.tap_layers = {tap_layers_.front()};
    s.tap_heads = {0};
    ro::prefill(*fp_, sched_, 0, 0, prompt_of(12), s);
    unchanged("after a normal call");

    // A head outside [0, n_head) fails loud at mark time — inside the call.
    fresh();
    ro::Spec bad = s;
    bad.tap_heads = {(int)model_->get_metadata().attention_head_count + 7};
    EXPECT_THROW(ro::prefill(*fp_, sched_, 0, 0, prompt_of(12), bad), std::runtime_error);
    unchanged("after a throwing call");
    ggml_backend_sched_reset(sched_);
}

TEST_P(ReadoutTest, RefusesEmptyTokensAndACutDecode) {
    fresh();
    EXPECT_THROW(ro::prefill(*fp_, sched_, 0, 0, {}, ro::Spec{}), std::runtime_error);
    ro::Spec cut;
    cut.last_layer = 0;
    EXPECT_THROW(ro::decode(*fp_, sched_, 0, 0, 1, cut), std::runtime_error);
    EXPECT_EQ(-1, fp_->truncate_after_layer());
}

namespace {
std::vector<std::string> legs() {
    return {"QWEN36_MODEL_PATH", "QWEN35_MODEL_PATH", "QWEN3_MODEL_PATH", "GEMMA3_MODEL_PATH",
            "GEMMA4_MODEL_PATH", "GEMMA4_MOE_MODEL_PATH"};
}
}  // namespace
INSTANTIATE_TEST_SUITE_P(Recipes, ReadoutTest, ::testing::ValuesIn(legs()),
                         [](const ::testing::TestParamInfo<std::string>& i) { return i.param; });

// ── MoE routing capture through the readout ──────────────────────────────────
// The trace a readout captures equals the hand-written capture's, and replaying
// it through the readout reproduces the logits bit for bit. Dense legs skip.
class ReadoutRoutingTest : public ReadoutTest {};

TEST_P(ReadoutRoutingTest, CaptureIsTheHandWrittenCaptureAndReplaysBitwise) {
    const std::vector<int32_t> p = prompt_of(40);
    fresh();
    RoutingTrace want;
    std::vector<float> want_logits;
    raw_prefill(p, 0, -1, {}, {}, true, &want_logits, &want);
    if (want.by_layer().empty()) GTEST_SKIP() << "dense recipe: no routing to capture";

    fresh();
    RoutingTrace got;
    ro::Spec s;
    s.logits = true;
    s.capture = &got;
    const ro::Result r = ro::prefill(*fp_, sched_, 0, 0, p, s);
    EXPECT_EQ(want.top_k(), got.top_k());
    EXPECT_EQ(want.by_layer(), got.by_layer()) << "captured expert selections expected identical";
    EXPECT_TRUE(same_floats(want_logits, r.logits));
    EXPECT_EQ(nullptr, fp_->routing_capture()) << "capture must not leak past the call";

    fresh();
    ro::Spec rp;
    rp.logits = true;
    rp.replay = &got;
    const ro::Result replayed = ro::prefill(*fp_, sched_, 0, 0, p, rp);
    EXPECT_TRUE(same_floats(want_logits, replayed.logits)) << "replay expected bitwise-identical logits";
    EXPECT_EQ(nullptr, fp_->routing_replay()) << "replay must not leak past the call";
}

INSTANTIATE_TEST_SUITE_P(MoeRecipes, ReadoutRoutingTest,
                         ::testing::Values(std::string("QWEN36_MODEL_PATH"), std::string("GEMMA4_MOE_MODEL_PATH")),
                         [](const ::testing::TestParamInfo<std::string>& i) { return i.param; });
