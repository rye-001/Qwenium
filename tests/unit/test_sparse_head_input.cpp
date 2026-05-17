// test_sparse_head_input.cpp — uploads the grammar valid-id set into the
// "valid_indices" slot; fail-loud when armed with no/empty set.

#include <gtest/gtest.h>

#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "../../src/graph_inputs/sparse_head_input.h"

TEST(SparseHeadInput, UploadsValidIds) {
    std::vector<int32_t> ids{11, 250, 4096, 7};

    ggml_init_params p{ ggml_tensor_overhead()*4 + ggml_graph_overhead(),
                        nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ids.size());
    ggml_set_input(t);
    ggml_set_name(t, "valid_indices");
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    std::vector<int32_t> toks(1, 0);
    StepContext step;
    step.gf = gf;
    step.tokens = &toks;
    step.sparse_ids = &ids;

    SparseHeadInput in;
    in.set_input(step);

    std::vector<int32_t> got(ids.size());
    ggml_backend_tensor_get(t, got.data(), 0, got.size()*sizeof(int32_t));
    EXPECT_EQ(got, ids);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
}

TEST(SparseHeadInput, FailLoudWhenNotArmed) {
    ggml_init_params p{ ggml_tensor_overhead()*4 + ggml_graph_overhead(),
                        nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    ggml_set_input(t);
    ggml_set_name(t, "valid_indices");
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);

    std::vector<int32_t> toks(1, 0);
    StepContext step;
    step.gf = gf;
    step.tokens = &toks;
    step.sparse_ids = nullptr;  // not armed -> must throw, never silent

    SparseHeadInput in;
    EXPECT_THROW(in.set_input(step), std::runtime_error);
    ggml_free(ctx);
}
