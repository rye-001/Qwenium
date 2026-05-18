// test_graph_input.cpp — GraphInputSet fan-out + conservative can_reuse,
// and the fail-loud require_tensor contract (slot / expected / actual).

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "../../src/graph_inputs/graph_input.h"
#include "../../src/graph_inputs/tokens_input.h"
#include "../../src/graph_inputs/positions_input.h"

TEST(GraphInputSet, FansOutSetInputAndIsConservative) {
    std::vector<int32_t> toks{3, 1, 4};

    ggml_init_params p{ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
                        nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ggml_tensor* tk = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, toks.size());
    ggml_set_input(tk); ggml_set_name(tk, "tokens");
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, toks.size());
    ggml_set_input(pos); ggml_set_name(pos, "inp_pos");
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, tk);
    ggml_build_forward_expand(gf, pos);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    GraphInputSet set;
    set.add(std::make_unique<TokensInput>());
    set.add(std::make_unique<PositionsInput>());
    EXPECT_FALSE(set.empty());

    StepContext step;
    step.gf = gf; step.tokens = &toks; step.pos = 10;
    set.set_input(step);

    std::vector<int32_t> gk(3), gp(3);
    ggml_backend_tensor_get(tk, gk.data(), 0, gk.size()*sizeof(int32_t));
    ggml_backend_tensor_get(pos, gp.data(), 0, gp.size()*sizeof(int32_t));
    EXPECT_EQ(gk, toks);
    EXPECT_EQ(gp, (std::vector<int32_t>{10,11,12}));

    // Phase 1: every input defaults can_reuse=false, so the set ANDs to false.
    EXPECT_FALSE(set.can_reuse(step));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
}

TEST(GraphInputSet, EmptySetCannotReuse) {
    GraphInputSet set;
    StepContext step;
    EXPECT_TRUE(set.empty());
    EXPECT_FALSE(set.can_reuse(step));
}

TEST(GraphInput, RequireTensorFailsLoudOnTypeMismatch) {
    std::vector<int32_t> toks{1};
    ggml_init_params p{ ggml_tensor_overhead()*4 + ggml_graph_overhead(),
                        nullptr, true };
    ggml_context* ctx = ggml_init(p);
    // "tokens" exists but as F32, not the I32 TokensInput requires.
    ggml_tensor* wrong = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(wrong); ggml_set_name(wrong, "tokens");
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, wrong);

    StepContext step;
    step.gf = gf; step.tokens = &toks;
    TokensInput in;
    EXPECT_THROW(in.set_input(step), std::runtime_error);
    ggml_free(ctx);
}
