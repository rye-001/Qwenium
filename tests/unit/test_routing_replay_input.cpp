// test_routing_replay_input.cpp — fills every "moe_routing.<il>" slot from a
// RoutingTrace, by ABSOLUTE position, and refuses every shape it cannot honor.

#include <gtest/gtest.h>

#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "../../src/graph_inputs/routing_replay_input.h"

namespace {

struct Harness {
    ggml_context*         ctx = nullptr;
    ggml_cgraph*          gf  = nullptr;
    ggml_backend_t        be  = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor*> slots;

    // One I32 [top_k, n_rows] tensor per layer, named like MoELayer names them.
    Harness(const std::vector<int>& layers, int top_k, int n_rows) {
        ggml_init_params p{ ggml_tensor_overhead() * (layers.size() + 4) +
                            ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(p);
        gf  = ggml_new_graph(ctx);
        for (int il : layers) {
            ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, n_rows);
            ggml_set_input(t);
            ggml_set_name(t, ("moe_routing." + std::to_string(il)).c_str());
            ggml_build_forward_expand(gf, t);
            slots.push_back(t);
        }
        be  = ggml_backend_cpu_init();
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    }
    ~Harness() {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(be);
        ggml_free(ctx);
    }
    std::vector<int32_t> read(size_t i, size_t n) {
        std::vector<int32_t> got(n);
        ggml_backend_tensor_get(slots[i], got.data(), 0, n * sizeof(int32_t));
        return got;
    }
};

RoutingTrace trace_for(const std::vector<int>& layers, int top_k, int n_positions) {
    RoutingTrace t;
    for (int il : layers)
        for (int pos = 0; pos < n_positions; ++pos) {
            std::vector<int32_t> sel(top_k);
            for (int k = 0; k < top_k; ++k) sel[k] = il * 1000 + pos * 10 + k;
            t.write(il, (size_t)pos, sel.data(), top_k);
        }
    return t;
}

}  // namespace

TEST(RoutingReplayInput, FillsEverySlotFromTheTrace) {
    Harness h({0, 5}, /*top_k=*/2, /*n_rows=*/3);
    RoutingTrace t = trace_for({0, 5}, 2, 3);
    std::vector<int32_t> toks(3, 0);

    StepContext step;
    step.gf = h.gf; step.tokens = &toks; step.pos = 0;

    RoutingReplayInput in(&t);
    in.set_input(step);

    EXPECT_EQ(h.read(0, 6), (std::vector<int32_t>{0, 1, 10, 11, 20, 21}));
    EXPECT_EQ(h.read(1, 6), (std::vector<int32_t>{5000, 5001, 5010, 5011, 5020, 5021}));
}

// The reason positions are absolute: a chunked prefill's second batch holds
// sequence positions 3..4, not 0..1, and replaying row indices would silently
// route those tokens with some other token's experts.
TEST(RoutingReplayInput, ReplaysByAbsolutePositionNotRowIndex) {
    Harness h({0}, /*top_k=*/2, /*n_rows=*/2);
    RoutingTrace t = trace_for({0}, 2, 5);
    std::vector<int32_t> toks(2, 0);

    StepContext step;
    step.gf = h.gf; step.tokens = &toks; step.pos = 3;   // a later chunk

    RoutingReplayInput in(&t);
    in.set_input(step);
    EXPECT_EQ(h.read(0, 4), (std::vector<int32_t>{30, 31, 40, 41}));
}

TEST(RoutingReplayInput, RefusesAShapeItCannotHonor) {
    Harness h({0}, /*top_k=*/4, /*n_rows=*/2);   // graph wants top_k 4
    RoutingTrace t = trace_for({0}, 2, 4);       // trace only has 2
    std::vector<int32_t> toks(2, 0);
    StepContext step;
    step.gf = h.gf; step.tokens = &toks;

    RoutingReplayInput in(&t);
    EXPECT_THROW(in.set_input(step), std::runtime_error);
}

// A truncated prefill (the lens verify path builds only the first N blocks)
// must replay the layers it DID build and ignore the rest. The deeper layers'
// absence is correct, not an error.
TEST(RoutingReplayInput, ReplaysOnlyTheLayersThisPassBuilt) {
    Harness h({0}, /*top_k=*/2, /*n_rows=*/1);      // graph built layer 0 only
    RoutingTrace t = trace_for({0, 9}, 2, 1);       // trace captured 0 and 9
    std::vector<int32_t> toks(1, 0);
    StepContext step;
    step.gf = h.gf; step.tokens = &toks;

    RoutingReplayInput in(&t);
    EXPECT_NO_THROW(in.set_input(step));
    EXPECT_EQ(h.read(0, 2), (std::vector<int32_t>{0, 1}));
}

// The converse is the documented gap, and it is worth pinning so a future
// change notices it: a layer the graph built but the trace never captured is
// left UNFILLED rather than refused, because a graph input lives in the
// leafs and there is no public way to enumerate the slots the graph created.
// The capture path cannot produce such a trace (it records every MoE layer at
// full depth), which is why this is a documented contract and not a defect.
TEST(RoutingReplayInput, ALayerAbsentFromTheTraceIsNotCaught) {
    Harness h({0, 9}, /*top_k=*/2, /*n_rows=*/1);
    RoutingTrace t = trace_for({0}, 2, 1);          // layer 9 never recorded
    std::vector<int32_t> toks(1, 0);
    StepContext step;
    step.gf = h.gf; step.tokens = &toks;

    RoutingReplayInput in(&t);
    EXPECT_NO_THROW(in.set_input(step));            // fills 0, says nothing about 9
    EXPECT_EQ(h.read(0, 2), (std::vector<int32_t>{0, 1}));
}

TEST(RoutingReplayInput, RefusesANullOrEmptyTrace) {
    Harness h({0}, 2, 1);
    std::vector<int32_t> toks(1, 0);
    StepContext step;
    step.gf = h.gf; step.tokens = &toks;

    RoutingReplayInput none(nullptr);
    EXPECT_THROW(none.set_input(step), std::runtime_error);

    RoutingTrace empty;
    RoutingReplayInput blank(&empty);
    EXPECT_THROW(blank.set_input(step), std::runtime_error);
}

// A graph built with RoutingSource::Router has no such slots. Filling nothing
// silently would mean a "replayed" pass that actually re-chose its experts.
TEST(RoutingReplayInput, RefusesAGraphThatHasNoRoutingSlots) {
    Harness h({}, 2, 1);
    std::vector<int32_t> toks(1, 0);
    StepContext step;
    step.gf = h.gf; step.tokens = &toks;

    RoutingTrace t = trace_for({0}, 2, 1);
    RoutingReplayInput in(&t);
    EXPECT_THROW(in.set_input(step), std::runtime_error);
}
