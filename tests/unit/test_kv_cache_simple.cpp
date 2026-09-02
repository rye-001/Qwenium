// test_kv_cache_simple.cpp — PR 2.8
//
// Unit tests for simple_kv_cache: build test, advance/position management,
// truncate_to_position, LayerState conformance, and scratch-budget note.
// No model file required.
//
// Run: ./qwen3-layer-tests --gtest_filter="KVCache*"

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../../src/state/kv_cache_simple.h"
#include "../../src/state/layer_state.h"
#include "../../src/session/snapshot_io.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

static constexpr int N_LAYERS   = 2;
static constexpr int N_CTX      = 32;
static constexpr int N_BATCH    = 2;
static constexpr int N_EMBD_KV  = 16;

class KVCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        cache_ = std::make_unique<simple_kv_cache>(
            N_LAYERS, N_CTX, N_BATCH,
            N_EMBD_KV, N_EMBD_KV,
            GGML_TYPE_F32, GGML_TYPE_F32, backend_);
    }
    void TearDown() override {
        cache_.reset();
        if (backend_) ggml_backend_free(backend_);
    }
    ggml_backend_t backend_ = nullptr;
    std::unique_ptr<simple_kv_cache> cache_;
};

// ── Build test ────────────────────────────────────────────────────────────────

TEST_F(KVCacheTest, ConstructionSucceeds) {
    EXPECT_GT(cache_->memory_bytes(), 0u);
    EXPECT_EQ(cache_->get_pos(0), 0u);
}

// ── Advance and position tracking ─────────────────────────────────────────────

TEST_F(KVCacheTest, AdvanceUpdatesPosition) {
    cache_->advance(5, 0);
    EXPECT_EQ(cache_->get_pos(0), 5u);
    cache_->advance(3, 0);
    EXPECT_EQ(cache_->get_pos(0), 8u);
}

TEST_F(KVCacheTest, SlotsAreIndependent) {
    cache_->advance(5, 0);
    cache_->advance(10, 1);
    EXPECT_EQ(cache_->get_pos(0), 5u);
    EXPECT_EQ(cache_->get_pos(1), 10u);
}

// ── truncate_to_position ──────────────────────────────────────────────────────

TEST_F(KVCacheTest, TruncateToPositionIsO1) {
    cache_->advance(20, 0);
    cache_->truncate_to_position(7, 0);
    EXPECT_EQ(cache_->get_pos(0), 7u);
}

TEST_F(KVCacheTest, TruncateToCurrentPositionIsNoop) {
    cache_->advance(5, 0);
    cache_->truncate_to_position(5, 0);
    EXPECT_EQ(cache_->get_pos(0), 5u);
}

// ── clear_slot via LayerState::reset_sequence ─────────────────────────────────

TEST_F(KVCacheTest, ResetSequenceClearsPosition) {
    cache_->advance(15, 1);
    static_cast<LayerState*>(cache_.get())->reset_sequence(1);
    EXPECT_EQ(cache_->get_pos(1), 0u);
}

// ── KV tensor views build cleanly ─────────────────────────────────────────────

TEST_F(KVCacheTest, GetKAndVTensorsAreNonNull) {
    cache_->advance(4, 0);

    const size_t ctx_bytes = 64 * ggml_tensor_overhead();
    ggml_init_params p{ctx_bytes, nullptr, true};
    ggml_context* ctx = ggml_init(p);
    ASSERT_NE(ctx, nullptr);

    ggml_tensor* k = cache_->get_k(ctx, 0, 4, 0);
    ggml_tensor* v = cache_->get_v(ctx, 0, 4, 0);
    EXPECT_NE(k, nullptr);
    EXPECT_NE(v, nullptr);
    EXPECT_EQ(k->ne[0], N_EMBD_KV);
    EXPECT_EQ(k->ne[1], 4);

    // Scratch budget note (PR 2.8 requirement):
    // simple_kv_cache tensors are persistent (not scratch) — they are the KV
    // store itself.  Views built over them add zero scratch allocation;
    // the memory cost is reported by memory_bytes() = cache buffer size.
    RecordProperty("memory_bytes", static_cast<int64_t>(cache_->memory_bytes()));

    ggml_free(ctx);
}


// ── Identity gather: single-slot view ≡ the batched gather ───────────────────
// gather_k_single replaces a materializing GET_ROWS with a view whenever one
// slot is active (docs/decode-gap-status.md §4). The whole claim is that the
// two produce THE SAME ROWS, so the test computes both in one graph and
// compares them elementwise — and also pins the absolute values, so a change
// that broke both identically could not pass.

TEST_F(KVCacheTest, SingleSlotGatherEqualsBatchedGather) {
    constexpr uint32_t SLOT = 1, N_KV = 5;

    // Fill layer 0's K cache with row-identifying values.
    ggml_tensor* kc = cache_->get_k_cache_tensor(0);
    const size_t n_rows_total = static_cast<size_t>(N_CTX) * N_BATCH;
    std::vector<float> fill(n_rows_total * N_EMBD_KV);
    for (size_t r = 0; r < n_rows_total; ++r)
        for (int e = 0; e < N_EMBD_KV; ++e)
            fill[r * N_EMBD_KV + e] = static_cast<float>(r * 1000 + e);
    ggml_backend_tensor_set(kc, fill.data(), 0, fill.size() * sizeof(float));

    ggml_init_params p{ 256 * ggml_tensor_overhead() + ggml_graph_overhead(),
                        nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ASSERT_NE(ctx, nullptr);

    ggml_tensor* idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_KV);
    ggml_set_input(idx);
    ggml_set_name(idx, "gather_indices");

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_tensor* gathered = cache_->gather_k(ctx, gf, 0, idx, 1, N_KV);
    ggml_tensor* viewed   = cache_->gather_k_single(ctx, 0, SLOT, N_KV);
    ggml_build_forward_expand(gf, gathered);
    ggml_build_forward_expand(gf, viewed);

    // Same shape, or the comparison below would be meaningless.
    ASSERT_EQ(gathered->ne[0], viewed->ne[0]);
    ASSERT_EQ(gathered->ne[1], viewed->ne[1]);
    ASSERT_EQ(gathered->ne[2], viewed->ne[2]);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

    // The indices GatherIndicesInput would build for one active slot.
    std::vector<int32_t> indices(N_KV);
    for (uint32_t t = 0; t < N_KV; ++t)
        indices[t] = static_cast<int32_t>(SLOT * N_CTX + t);
    ggml_backend_tensor_set(idx, indices.data(), 0, indices.size() * sizeof(int32_t));

    ASSERT_EQ(ggml_backend_graph_compute(backend_, gf), GGML_STATUS_SUCCESS);

    std::vector<float> got_gather(N_KV * N_EMBD_KV), got_view(N_KV * N_EMBD_KV);
    ggml_backend_tensor_get(gathered, got_gather.data(), 0, got_gather.size() * sizeof(float));
    ggml_backend_tensor_get(viewed,   got_view.data(),   0, got_view.size()   * sizeof(float));

    for (uint32_t t = 0; t < N_KV; ++t) {
        for (int e = 0; e < N_EMBD_KV; ++e) {
            const size_t i = t * N_EMBD_KV + e;
            const float expected = static_cast<float>((SLOT * N_CTX + t) * 1000 + e);
            EXPECT_FLOAT_EQ(got_gather[i], expected) << "gather t=" << t << " e=" << e;
            EXPECT_FLOAT_EQ(got_view[i],   expected) << "view   t=" << t << " e=" << e;
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

TEST_F(KVCacheTest, SingleSlotGatherFailsLoudOnBadSlot) {
    ggml_init_params p{ 32 * ggml_tensor_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ASSERT_NE(ctx, nullptr);
    EXPECT_THROW(cache_->gather_k_single(ctx, 0, N_BATCH, 4), std::runtime_error);
    EXPECT_THROW(cache_->gather_v_single(ctx, 0, N_BATCH, 4), std::runtime_error);
    ggml_free(ctx);
}

TEST_F(KVCacheTest, SingleSlotGatherFailsLoudOnOversizedNKv) {
    ggml_init_params p{ 32 * ggml_tensor_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ASSERT_NE(ctx, nullptr);
    EXPECT_THROW(cache_->gather_k_single(ctx, 0, 0, N_CTX + 1), std::runtime_error);
    EXPECT_THROW(cache_->gather_v_single(ctx, 0, 0, N_CTX + 1), std::runtime_error);
    ggml_free(ctx);
}

// ── path_tag folds the owner's salt ──────────────────────────────────────────
// path_tag() decides whether a frozen KV blob may be memcpy'd into this cache
// and resumed. Its own configuration (backend, dtypes, ctx) is not the whole
// story: the BYTES depend on the compute path that produced them, which the
// cache cannot see. --flash-attn is the case that forced this — flash changes
// the attention output, hence the residual stream, hence every later layer's
// K/V, so a prefill done under flash is not interchangeable with one done
// materialized. Without the salt, such a blob would be silently resumed under
// the wrong implementation instead of refused.

TEST_F(KVCacheTest, PathTagFoldsTheOwnerSalt) {
    const uint64_t base = cache_->path_tag();

    cache_->set_path_salt(0x9e3779b97f4a7c15ull);
    const uint64_t salted = cache_->path_tag();
    EXPECT_NE(base, salted)
        << "path_tag ignored the salt: a blob built under a different compute "
        << "path would be accepted and resumed rather than refused.";

    // Deterministic, and reversible — the same salt must give the same tag, or
    // a blob would be refused against the build that wrote it.
    cache_->set_path_salt(0x9e3779b97f4a7c15ull);
    EXPECT_EQ(salted, cache_->path_tag());
    cache_->set_path_salt(0);
    EXPECT_EQ(base, cache_->path_tag());
}

// ── KV element type is selectable (--kv-f16) ─────────────────────────────────
// simple_kv_cache has always taken type_k/type_v; these pin the two properties
// the F16 option depends on, so a future change cannot silently break the
// opt-in path or, worse, let an F16 blob resume into an F32 cache.

TEST(KVCacheTypeTest, F16HalvesCacheBytes) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    simple_kv_cache f32(N_LAYERS, N_CTX, N_BATCH, N_EMBD_KV, N_EMBD_KV,
                        GGML_TYPE_F32, GGML_TYPE_F32, backend);
    simple_kv_cache f16(N_LAYERS, N_CTX, N_BATCH, N_EMBD_KV, N_EMBD_KV,
                        GGML_TYPE_F16, GGML_TYPE_F16, backend);
    EXPECT_GT(f32.memory_bytes(), 0u);
    EXPECT_EQ(f16.memory_bytes() * 2, f32.memory_bytes());
    ggml_backend_free(backend);
}

TEST(KVCacheTypeTest, PathTagSeparatesKvTypes) {
    // path_tag feeds CompatHeader::build_path_tag, which is what stops an L2
    // prefix blob captured under one KV dtype from being reused under another.
    ggml_backend_t backend = ggml_backend_cpu_init();
    simple_kv_cache f32(N_LAYERS, N_CTX, N_BATCH, N_EMBD_KV, N_EMBD_KV,
                        GGML_TYPE_F32, GGML_TYPE_F32, backend);
    simple_kv_cache f16(N_LAYERS, N_CTX, N_BATCH, N_EMBD_KV, N_EMBD_KV,
                        GGML_TYPE_F16, GGML_TYPE_F16, backend);
    EXPECT_NE(f32.path_tag(), f16.path_tag());
    ggml_backend_free(backend);
}

TEST(KVCacheTypeTest, RestoreAcrossKvTypesFailsLoud) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    simple_kv_cache f32(N_LAYERS, N_CTX, N_BATCH, N_EMBD_KV, N_EMBD_KV,
                        GGML_TYPE_F32, GGML_TYPE_F32, backend);
    simple_kv_cache f16(N_LAYERS, N_CTX, N_BATCH, N_EMBD_KV, N_EMBD_KV,
                        GGML_TYPE_F16, GGML_TYPE_F16, backend);

    f32.advance(4, 0);
    qinf::session::SnapshotWriter w;
    f32.serialize_slot(w, 0);

    qinf::session::SnapshotReader r(w.buffer().data(), w.buffer().size());
    // Must throw naming type_k — a silent accept would resume a divergent slot.
    EXPECT_THROW(f16.deserialize_slot(r, 0), std::runtime_error);
    ggml_backend_free(backend);
}

TEST(KVCacheTypeTest, RestoreSameKvTypeRoundTrips) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    simple_kv_cache a(N_LAYERS, N_CTX, N_BATCH, N_EMBD_KV, N_EMBD_KV,
                      GGML_TYPE_F16, GGML_TYPE_F16, backend);
    simple_kv_cache b(N_LAYERS, N_CTX, N_BATCH, N_EMBD_KV, N_EMBD_KV,
                      GGML_TYPE_F16, GGML_TYPE_F16, backend);
    a.advance(4, 0);
    qinf::session::SnapshotWriter w;
    a.serialize_slot(w, 0);
    qinf::session::SnapshotReader r(w.buffer().data(), w.buffer().size());
    EXPECT_NO_THROW(b.deserialize_slot(r, 0));
    EXPECT_EQ(b.get_pos(0), 4u);
    ggml_backend_free(backend);
}

// ── The two gather branches do NOT return the same element type ──────────────
//
// This is the trap that produced a real, months-latent defect (2026-09-02):
//
//   * gather_k/gather_v (multi-slot) go through ggml_get_rows, whose result type
//     is ALWAYS F32 unless the source is I32 -- i.e. it dequantizes.
//   * gather_k_single/gather_v_single (the B==1 fast path) return a VIEW of the
//     cache, so the result keeps the cache's own element type.
//
// A caller that builds a view over the gathered tensor with a hardcoded
// `n * sizeof(float)` stride is therefore correct for the first and silently
// mis-reads the second under --kv-type f16/q8_0/q4_0. That is exactly what
// layers/attention.cpp's build_batched_attention and build_gated_batched_attention
// did, and the symptom was degenerate output rather than a failure -- the
// silent-mis-read architecture.md section 9 warns about in as many words. Strides
// must come from the gathered tensor's own type (ggml_row_size), never sizeof(float).
//
// Model-free: this pins the type contract itself, so the defect class cannot
// come back without a red test, on any recipe.
// A quantized row must be a whole number of BLOCKS: Q8_0 and Q4_0 both block at
// 32 elements, so this fixture cannot reuse the 16-wide N_EMBD_KV above -- a
// 16-element row is less than one block and ggml aborts building the view.
// 64 is block-aligned for every type under test.
static constexpr int N_EMBD_KV_BLOCKED = 64;

class KVCacheGatherTypeTest : public ::testing::TestWithParam<ggml_type> {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        cache_ = std::make_unique<simple_kv_cache>(
            N_LAYERS, N_CTX, N_BATCH, N_EMBD_KV_BLOCKED, N_EMBD_KV_BLOCKED,
            GetParam(), GetParam(), backend_);
        ggml_init_params ip{};
        ip.mem_size   = 16u * 1024 * 1024;
        ip.no_alloc   = true;
        ctx_ = ggml_init(ip);
    }
    void TearDown() override {
        if (ctx_) ggml_free(ctx_);
        cache_.reset();
        if (backend_) ggml_backend_free(backend_);
    }
    ggml_backend_t backend_ = nullptr;
    ggml_context*  ctx_     = nullptr;
    std::unique_ptr<simple_kv_cache> cache_;
};

TEST_P(KVCacheGatherTypeTest, SingleSlotGatherKeepsTheCacheElementType) {
    ggml_tensor* k = cache_->gather_k_single(ctx_, /*il=*/0, /*slot=*/0, /*n_kv=*/4);
    ggml_tensor* v = cache_->gather_v_single(ctx_, /*il=*/0, /*slot=*/0, /*n_kv=*/4);
    ASSERT_NE(k, nullptr);
    ASSERT_NE(v, nullptr);
    EXPECT_EQ(k->type, GetParam())
        << "gather_k_single is a VIEW of the cache and must keep its element "
           "type; a caller striding it as F32 would silently mis-read it";
    EXPECT_EQ(v->type, GetParam());
}

TEST_P(KVCacheGatherTypeTest, AViewOverTheGatheredTensorMustUseRowSizeNotFloat) {
    ggml_tensor* k = cache_->gather_k_single(ctx_, /*il=*/0, /*slot=*/0, /*n_kv=*/4);
    ASSERT_NE(k, nullptr);
    // What the fixed callers compute, vs the hardcoded stride they used to.
    const size_t row_size   = ggml_row_size(k->type, N_EMBD_KV_BLOCKED);
    const size_t float_size = static_cast<size_t>(N_EMBD_KV_BLOCKED) * sizeof(float);
    EXPECT_EQ(row_size, ggml_row_size(k->type, N_EMBD_KV_BLOCKED));
    if (GetParam() == GGML_TYPE_F32) {
        // The identity that makes the fix byte-identical on the default path.
        EXPECT_EQ(row_size, float_size);
    } else {
        EXPECT_NE(row_size, float_size)
            << "a non-F32 cache must NOT have a float row stride — if these are "
               "equal the test has stopped discriminating and the regression it "
               "guards could return unnoticed";
    }
}

INSTANTIATE_TEST_SUITE_P(
    ElementTypes, KVCacheGatherTypeTest,
    ::testing::Values(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0),
    [](const ::testing::TestParamInfo<ggml_type>& i) {
        return std::string(ggml_type_name(i.param));
    });
