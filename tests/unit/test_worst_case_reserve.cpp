// test_worst_case_reserve.cpp — co-located test for src/engine/worst_case_reserve.cpp.
//
// The contract both servers rely on: after the reservation every slot is empty
// again (it advanced them to the context ceiling only to size the decode graph),
// and the scheduler then computes ordinary steps — prefill and decode — with the
// same result as on a pass that was never reserved. Qwen and Gemma legs
// (CLAUDE.md cross-family rule); each self-skips when its variable is unset.
//   QWEN3_MODEL_PATH  qwen3 transformer   QWEN35_MODEL_PATH  qwen35 hybrid
//   GEMMA3_MODEL_PATH gemma3

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "engine/model.h"
#include "engine/worst_case_reserve.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/forward_pass_base.h"

namespace {

constexpr int kSlots = 4;
constexpr int kCtx = 256;

std::unique_ptr<Model> g_model;
std::string g_path;

Model* load(const std::string& path) {
    if (path != g_path) {
        g_model.reset();
        register_builtin_models();
        g_model = std::make_unique<Model>();
        g_model->load_metadata(path);
        g_model->load_tensors();
        g_path = path;
    }
    return g_model.get();
}

class ReleaseEnv : public ::testing::Environment {
public:
    void TearDown() override { g_model.reset(); g_path.clear(); }   // before ggml-metal's statics
};
const bool kEnv = (::testing::AddGlobalTestEnvironment(new ReleaseEnv), true);

std::vector<float> prefill_then_decode(ForwardPassBase& fp, ggml_backend_sched_t sched) {
    std::vector<int32_t> p;
    for (int32_t t = 1; t <= 20; ++t) p.push_back(t);
    fp.clear_slot(0);
    fp.set_cache_pos(0, 0);
    fp.run_prefill(p, 0, 0, sched);
    std::vector<int32_t> tks = {p.back()};
    std::vector<uint32_t> slots = {0};
    std::vector<int32_t> pos = {(int32_t)fp.get_cache_pos(0)};
    ggml_cgraph* gf = fp.build_decoding_graph(tks, slots, pos);
    fp.alloc_readback_graph(sched, gf);
    fp.set_decode_inputs(gf, tks, slots, pos);
    ggml_backend_sched_graph_compute(sched, gf);
    std::vector<float> out = fp.get_output_logits(gf);
    fp.clear_slot(0);
    return out;
}

}  // namespace

class WorstCaseReserveTest : public ::testing::TestWithParam<std::string> {};

TEST_P(WorstCaseReserveTest, LeavesEverySlotEmptyAndStepsUnchanged) {
    const char* p = std::getenv(GetParam().c_str());
    if (!p || !*p) GTEST_SKIP() << GetParam() << " not set — skipping this leg";
    Model* m = load(p);
    ggml_backend_sched_t sched = m->get_scheduler();

    auto plain = create_forward_pass(*m, &m->get_metadata(), kCtx, kSlots);
    const std::vector<float> want = prefill_then_decode(*plain, sched);
    plain.reset();

    auto fp = create_forward_pass(*m, &m->get_metadata(), kCtx, kSlots);
    qinf::engine::reserve_worst_case(*fp, sched, kSlots, kCtx);
    for (uint32_t s = 0; s < (uint32_t)kSlots; ++s)
        EXPECT_EQ(0u, fp->get_cache_pos(s)) << "slot " << s << " expected empty after the reservation";
    const std::vector<float> got = prefill_then_decode(*fp, sched);
    ASSERT_EQ(want.size(), got.size());
    EXPECT_EQ(0, std::memcmp(want.data(), got.data(), want.size() * sizeof(float)))
        << GetParam() << ": a step after the reservation expected bitwise-identical logits";
}

INSTANTIATE_TEST_SUITE_P(Recipes, WorstCaseReserveTest,
                         ::testing::Values(std::string("QWEN3_MODEL_PATH"), std::string("QWEN35_MODEL_PATH"),
                                           std::string("GEMMA3_MODEL_PATH")),
                         [](const ::testing::TestParamInfo<std::string>& i) { return i.param; });
