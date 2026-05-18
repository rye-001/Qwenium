// test_tokens_input.cpp — TokensInput copies the token id list verbatim.

#include <gtest/gtest.h>

#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "../../src/graph_inputs/tokens_input.h"

TEST(TokensInput, CopiesTokenList) {
    std::vector<int32_t> toks{5, 9, 1, 4, 2};

    ggml_init_params p{ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
                        nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, toks.size());
    ggml_set_input(t);
    ggml_set_name(t, "tokens");
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    StepContext step;
    step.gf = gf;
    step.tokens = &toks;

    TokensInput in;
    in.set_input(step);

    std::vector<int32_t> got(toks.size());
    ggml_backend_tensor_get(t, got.data(), 0, got.size() * sizeof(int32_t));
    EXPECT_EQ(got, toks);
    EXPECT_STREQ(in.slot_name(), "tokens");
    EXPECT_FALSE(in.can_reuse(step));  // Phase 1: always conservative

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
}

TEST(TokensInput, FailLoudWhenTokensNull) {
    ggml_init_params p{ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
                        nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    ggml_set_input(t);
    ggml_set_name(t, "tokens");
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);

    StepContext step;
    step.gf = gf;
    step.tokens = nullptr;

    TokensInput in;
    EXPECT_THROW(in.set_input(step), std::runtime_error);
    ggml_free(ctx);
}
