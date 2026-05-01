// test_rope_pruned.cpp — PR G4.5
//
// Unit tests for build_rope_pruned (p-RoPE).
//
// p-RoPE rotates only the first n_rot dimensions of each attention head;
// the remaining (head_dim - n_rot) dimensions pass through unchanged.
// Used by Gemma 4 global layers (rope_dim < head_dim_k).
//
// Tests:
//   1. ZeroPositionIsIdentity   — at pos=0, RoPE returns input unchanged
//                                 (sanity check for ggml_rope_ext shape).
//   2. TailPassesThrough         — with n_rot < head_dim, the trailing
//                                 (head_dim - n_rot) elements of every head
//                                 equal the input element-for-element at any
//                                 position.  This is the load-bearing p-RoPE
//                                 contract.
//   3. FullEqualsStandard        — with n_rot == head_dim, build_rope_pruned
//                                 is bit-identical to a direct ggml_rope_ext
//                                 call (it MUST be — same kernel underneath).
//   4. HeadDimsActuallyRotate    — with n_rot < head_dim and pos > 0, at
//                                 least one of the first n_rot elements
//                                 differs from the input (cheap regression
//                                 against accidentally degenerating to a
//                                 no-op).
//   5. NeoxPairOracle            — analytical oracle for a single pair using
//                                 the documented NEOX rotation formula.
//
// No model file required.
// Build target: rope-pruned-tests (see tests/CMakeLists.txt).

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "../../src/layers/attention.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// Synthetic dimensions kept small so oracles fit on the page.
static constexpr int RP_HEAD_DIM = 8;
static constexpr int RP_N_HEAD   = 1;
static constexpr int RP_N_TOKENS = 1;
static constexpr int RP_N_CTX    = 32;

namespace {

// One-shot harness: builds a graph that calls build_rope_pruned, computes it,
// and returns the output as a plain float vector along with the input the
// graph was driven with.
struct RopeRun {
    std::vector<float> input;
    std::vector<float> output;
};

RopeRun run_pruned(int n_rot, int pos, float freq_base,
                   const std::vector<float>& seed_input)
{
    ggml_backend_t backend = ggml_backend_cpu_init();
    EXPECT_NE(backend, nullptr);

    const size_t ctx_bytes = 256 * ggml_tensor_overhead()
                           + 32 * RP_HEAD_DIM * RP_N_HEAD * RP_N_TOKENS * sizeof(float);
    ggml_init_params p = { ctx_bytes, nullptr, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(p);

    ggml_cgraph* gf = ggml_new_graph(ctx);

    ggml_tensor* x = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, RP_HEAD_DIM, RP_N_HEAD, RP_N_TOKENS);
    ggml_set_input(x);
    ggml_set_name(x, "x");

    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, RP_N_TOKENS);
    ggml_set_input(inp_pos);
    ggml_set_name(inp_pos, "inp_pos");

    ggml_tensor* y = build_rope_pruned(
        ctx, x, inp_pos, n_rot, /*context_len=*/RP_N_CTX, freq_base);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    EXPECT_EQ(seed_input.size(),
              static_cast<size_t>(RP_HEAD_DIM * RP_N_HEAD * RP_N_TOKENS));
    ggml_backend_tensor_set(x, seed_input.data(), 0,
                            seed_input.size() * sizeof(float));
    int32_t pv[RP_N_TOKENS] = { pos };
    ggml_backend_tensor_set(inp_pos, pv, 0, sizeof(pv));

    ggml_backend_graph_compute(backend, gf);

    RopeRun out;
    out.input = seed_input;
    out.output.resize(seed_input.size());
    ggml_backend_tensor_get(y, out.output.data(), 0,
                            out.output.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return out;
}

// Same harness, but invoking ggml_rope_ext directly with n_rot == HEAD_DIM,
// so we have a reference to bit-identity-check against build_rope_pruned.
RopeRun run_standard(int pos, float freq_base,
                     const std::vector<float>& seed_input)
{
    ggml_backend_t backend = ggml_backend_cpu_init();
    EXPECT_NE(backend, nullptr);

    const size_t ctx_bytes = 256 * ggml_tensor_overhead()
                           + 32 * RP_HEAD_DIM * RP_N_HEAD * RP_N_TOKENS * sizeof(float);
    ggml_init_params p = { ctx_bytes, nullptr, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(p);

    ggml_cgraph* gf = ggml_new_graph(ctx);

    ggml_tensor* x = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, RP_HEAD_DIM, RP_N_HEAD, RP_N_TOKENS);
    ggml_set_input(x);
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, RP_N_TOKENS);
    ggml_set_input(inp_pos);

    ggml_tensor* y = ggml_rope_ext(ctx, x, inp_pos, nullptr,
        RP_HEAD_DIM, GGML_ROPE_TYPE_NEOX, RP_N_CTX,
        freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(x, seed_input.data(), 0,
                            seed_input.size() * sizeof(float));
    int32_t pv[RP_N_TOKENS] = { pos };
    ggml_backend_tensor_set(inp_pos, pv, 0, sizeof(pv));

    ggml_backend_graph_compute(backend, gf);

    RopeRun out;
    out.input = seed_input;
    out.output.resize(seed_input.size());
    ggml_backend_tensor_get(y, out.output.data(), 0,
                            out.output.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return out;
}

std::vector<float> ascending_seed() {
    std::vector<float> v(RP_HEAD_DIM * RP_N_HEAD * RP_N_TOKENS);
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i + 1);
    return v;
}

} // namespace

// Test 1: pos=0 → output equals input (RoPE rotation is the identity at p=0).
// This guards against the harness silently misconfiguring the call (e.g.
// passing the wrong rope mode).
TEST(RopePrunedTest, ZeroPositionIsIdentity) {
    auto seed = ascending_seed();
    auto run = run_pruned(/*n_rot=*/4, /*pos=*/0,
                          /*freq_base=*/10000.0f, seed);
    ASSERT_EQ(run.output.size(), seed.size());
    for (size_t i = 0; i < seed.size(); ++i) {
        EXPECT_FLOAT_EQ(run.output[i], seed[i])
            << "pos=0 should be identity at index " << i;
    }
}

// Test 2: with n_rot < head_dim, the trailing (head_dim - n_rot) elements of
// every head pass through unchanged regardless of position.  This is the
// defining property of p-RoPE.
TEST(RopePrunedTest, TailPassesThrough) {
    auto seed = ascending_seed();
    constexpr int N_ROT = 4;  // half of HEAD_DIM=8

    auto run = run_pruned(N_ROT, /*pos=*/3, /*freq_base=*/10000.0f, seed);
    ASSERT_EQ(run.output.size(), seed.size());
    for (int t = 0; t < RP_N_TOKENS; ++t) {
        for (int h = 0; h < RP_N_HEAD; ++h) {
            for (int d = N_ROT; d < RP_HEAD_DIM; ++d) {
                const size_t idx = t * RP_N_HEAD * RP_HEAD_DIM
                                 + h * RP_HEAD_DIM + d;
                EXPECT_FLOAT_EQ(run.output[idx], seed[idx])
                    << "tail dim " << d << " should pass through";
            }
        }
    }
}

// Test 3: build_rope_pruned with n_rot == head_dim is bit-identical to a
// direct ggml_rope_ext call.  This confirms the wrapper is a thin pass-through
// for the standard-RoPE case (recipes that always rotate the full head dim,
// i.e. RopeKind::Standard, suffer zero behavior change).
TEST(RopePrunedTest, FullEqualsStandard) {
    auto seed = ascending_seed();
    auto pruned   = run_pruned(/*n_rot=*/RP_HEAD_DIM, /*pos=*/2,
                                /*freq_base=*/10000.0f, seed);
    auto standard = run_standard(/*pos=*/2, /*freq_base=*/10000.0f, seed);

    ASSERT_EQ(pruned.output.size(), standard.output.size());
    for (size_t i = 0; i < pruned.output.size(); ++i) {
        EXPECT_FLOAT_EQ(pruned.output[i], standard.output[i])
            << "pruned full-rot must match standard at index " << i;
    }
}

// Test 4: with n_rot < head_dim and pos > 0, at least one rotated element
// must differ from the input — guards against the implementation accidentally
// degenerating to a no-op.
TEST(RopePrunedTest, HeadDimsActuallyRotate) {
    auto seed = ascending_seed();
    constexpr int N_ROT = 4;
    auto run = run_pruned(N_ROT, /*pos=*/1, /*freq_base=*/10000.0f, seed);

    bool changed = false;
    for (int d = 0; d < N_ROT; ++d) {
        if (std::fabs(run.output[d] - seed[d]) > 1e-5f) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed)
        << "pruned RoPE at pos=1 should rotate the leading n_rot dims";
}

// Test 5: analytical oracle for the leading pair under NEOX rotation.
// NEOX pairs (i, i + n_rot/2) and rotates with theta_i = base^(-2i/n_rot).
// Pair 0 with n_rot=4 has theta_0 = 1, so at pos=1:
//     out[0] = q[0] * cos(1) - q[2] * sin(1)
//     out[2] = q[0] * sin(1) + q[2] * cos(1)
// We seed q[0]=1, q[2]=3 (from ascending_seed) so the values are concrete.
TEST(RopePrunedTest, NeoxPairOracle) {
    auto seed = ascending_seed();  // [1,2,3,4,5,6,7,8]
    constexpr int N_ROT = 4;
    auto run = run_pruned(N_ROT, /*pos=*/1, /*freq_base=*/10000.0f, seed);

    const double c = std::cos(1.0);
    const double s = std::sin(1.0);
    const double q0 = 1.0;
    const double q2 = 3.0;
    const float expect_0 = static_cast<float>(q0 * c - q2 * s);
    const float expect_2 = static_cast<float>(q0 * s + q2 * c);

    EXPECT_NEAR(run.output[0], expect_0, 1e-4f)
        << "oracle mismatch on out[0]";
    EXPECT_NEAR(run.output[2], expect_2, 1e-4f)
        << "oracle mismatch on out[2]";
}
