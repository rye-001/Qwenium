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
