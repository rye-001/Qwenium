// test_kv_write_setrows.cpp — P1 gate of docs/plan-persistent-decode-graph.md.
//
// THE DIFFERENTIAL: decode with KvWriteMode::Cpy (legacy baked-offset write)
// vs KvWriteMode::SetRows (value-driven ggml_set_rows write). SAME route
// (build_decoding_graph), SAME batch shape (N=1), same tokens/positions —
// the write op is the ONLY difference, and it writes the same bytes to the
// same cache rows. So the gate is BITWISE, on both the per-step logits and
// the written K/V cache region, and it RUNS strict (no Metal batch-shape
// fork applies: nothing about the graph's matmul shapes changes).
//
// Cross-family per the standing rule: Qwen3.5 (gated helper, hybrid with
// recurrent state) and Gemma 3 (plain helper, sliding-window masks).
// Model legs self-skip when the GGUF is absent.
//
// Pure-unit legs (no model): KvWriteIndicesInput row math and fail-loud
// range/shape checks, on a minimal CPU-backend graph.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/qwen35.h"
#include "../../src/models/gemma3.h"
#include "../../src/graph_inputs/kv_write_indices_input.h"
#include "../../src/state/kv_cache_simple.h"

// ── Pure-unit: KvWriteIndicesInput ───────────────────────────────────────────

namespace {
struct H {
    ggml_context* ctx; ggml_backend_t be; ggml_backend_buffer_t buf;
    ggml_cgraph* gf; ggml_tensor* t;
    explicit H(size_t n) {
        ggml_init_params p{ ggml_tensor_overhead()*4 + ggml_graph_overhead(),
                            nullptr, true };
        ctx = ggml_init(p);
        t = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n);
        ggml_set_input(t); ggml_set_name(t, KvWriteIndicesInput::slot_);
        gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, t);
        be = ggml_backend_cpu_init();
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    }
    ~H(){ ggml_backend_buffer_free(buf); ggml_backend_free(be); ggml_free(ctx); }
};
}  // namespace

TEST(KvWriteIndicesInput, SlotStrideTimesCtxPlusPosition) {
    const uint32_t n_batch = 3, n_ctx_max = 100;
    std::vector<int32_t>  toks(n_batch, 0);
    std::vector<uint32_t> slots{1, 0, 4};
    std::vector<int32_t>  positions{7, 99, 0};
    H h(n_batch);
    StepContext step;
    step.gf = h.gf; step.tokens = &toks; step.slots = &slots;
    step.positions = &positions;

    KvWriteIndicesInput in(n_ctx_max);
    in.set_input(step);

    std::vector<int64_t> got(n_batch);
    ggml_backend_tensor_get(h.t, got.data(), 0, got.size()*sizeof(int64_t));
    for (uint32_t b = 0; b < n_batch; ++b)
        EXPECT_EQ(got[b], (int64_t)slots[b]*n_ctx_max + positions[b]);
}

TEST(KvWriteIndicesInput, FailLoudWhenPositionOutOfRange) {
    const uint32_t n_batch = 1, n_ctx_max = 8;
    std::vector<int32_t>  toks(n_batch, 0);
    std::vector<uint32_t> slots{0};
    std::vector<int32_t>  positions{8};  // == n_ctx_max ⇒ out of range
    H h(n_batch);
    StepContext step;
    step.gf = h.gf; step.tokens = &toks; step.slots = &slots;
    step.positions = &positions;
    KvWriteIndicesInput in(n_ctx_max);
    EXPECT_THROW(in.set_input(step), std::runtime_error);
}

TEST(KvWriteIndicesInput, FailLoudWhenSlotsNull) {
    std::vector<int32_t> toks(2, 0);
    H h(2);
    StepContext step;
    step.gf = h.gf; step.tokens = &toks; step.slots = nullptr;
    KvWriteIndicesInput in(64);
    EXPECT_THROW(in.set_input(step), std::runtime_error);
}

TEST(KvWriteIndicesInput, FailLoudWhenRowCountMismatch) {
    std::vector<int32_t>  toks(3, 0);   // 3 batch rows…
    std::vector<uint32_t> slots{0, 1, 2};
    std::vector<int32_t>  positions{0, 0, 0};
    H h(2);                             // …but a 2-row tensor
    StepContext step;
    step.gf = h.gf; step.tokens = &toks; step.slots = &slots;
    step.positions = &positions;
    KvWriteIndicesInput in(64);
    EXPECT_THROW(in.set_input(step), std::runtime_error);
}

// ── Model differential: Cpy vs SetRows, bitwise ──────────────────────────────

static std::string env_or(const char* var, const char* fallback) {
    if (const char* e = std::getenv(var))
        if (e[0]) return std::string(e);
    return std::string(fallback);
}

static bool file_exists(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

// Deterministic in-vocab token ids — no tokenizer variance.
static std::vector<int32_t> mk_tokens(int base, int n) {
    std::vector<int32_t> v;
    for (int i = 0; i < n; ++i)
        v.push_back(static_cast<int32_t>((base + i * 7 + 3) % 1000));
    return v;
}

// One decode step via the unified decode graph (rebuild path).
template <typename FP>
static std::vector<float> decode_one(FP& fp, ggml_backend_sched_t sched,
                                     int32_t token, uint32_t slot, int pos) {
    const std::vector<int32_t>  tokens    = {token};
    const std::vector<uint32_t> slots     = {slot};
    const std::vector<int32_t>  positions = {pos};
    ggml_backend_sched_reset(sched);
    ggml_cgraph* gf = fp.build_decoding_graph(tokens, slots, positions);
    ggml_backend_sched_alloc_graph(sched, gf);
    fp.set_decode_inputs(gf, tokens, slots, positions);
    ggml_backend_sched_graph_compute(sched, gf);
    fp.advance_cache(1, slot);
    return fp.get_output_logits(gf);
}

// Read the written K/V span (slot 0, rows [0, pos_end)) of every cache layer
// as raw bytes. Layout: [n_embd, n_ctx, n_slots], rows contiguous per slot,
// slot 0 at offset 0 — one contiguous read per layer per cache.
static std::vector<std::vector<uint8_t>> read_kv_bytes(simple_kv_cache* kv,
                                                       uint32_t pos_end) {
    std::vector<std::vector<uint8_t>> out;
    for (uint32_t il = 0; il < kv->get_n_layers(); ++il) {
        for (ggml_tensor* t : {kv->get_k_cache_tensor(static_cast<int>(il)),
                               kv->get_v_cache_tensor(static_cast<int>(il))}) {
            std::vector<uint8_t> bytes(static_cast<size_t>(pos_end) * t->nb[1]);
            ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
            out.push_back(std::move(bytes));
        }
    }
    return out;
}

// The differential proper. FP is the recipe's forward-pass class.
template <typename FP>
static void run_write_mode_differential(const std::string& path) {
    if (!file_exists(path))
        GTEST_SKIP() << "model absent: " << path;

    register_builtin_models();
    Model model;
    model.load_metadata(path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    ggml_backend_sched_t sched = model.get_scheduler();

    const std::vector<int32_t> init  = mk_tokens(11, 12);
    const std::vector<int32_t> feeds = mk_tokens(500, 8);
    const int P = static_cast<int>(init.size());

    // Both at EXACT n_kv (bucket 0): this gate isolates the write mode, so the
    // read width must not vary between the legs (bucketing is a separate,
    // token-stable-not-bitwise fork — test_decode_kv_bucket).
    // Reference: legacy baked-offset cpy write.
    FP fpCpy(model, &meta, 1024, 1);
    fpCpy.set_kv_write_mode(ForwardPassBase::KvWriteMode::Cpy);
    fpCpy.set_decode_kv_bucket(0);
    fpCpy.run_prefill(init, 0, 0, sched);

    // Candidate: value-driven set_rows write.
    FP fpSet(model, &meta, 1024, 1);
    fpSet.set_kv_write_mode(ForwardPassBase::KvWriteMode::SetRows);
    fpSet.set_decode_kv_bucket(0);
    fpSet.run_prefill(init, 0, 0, sched);

    for (size_t i = 0; i < feeds.size(); ++i) {
        const int pos = P + static_cast<int>(i);
        std::vector<float> a = decode_one(fpCpy, sched, feeds[i], 0, pos);
        std::vector<float> b = decode_one(fpSet, sched, feeds[i], 0, pos);
        ASSERT_EQ(a.size(), b.size()) << "step " << i;

        size_t mismatches = 0;
        float  max_diff   = 0.0f;
        for (size_t j = 0; j < a.size(); ++j) {
            if (std::memcmp(&a[j], &b[j], sizeof(float)) != 0) {
                ++mismatches;
                float d = std::fabs(a[j] - b[j]);
                if (d > max_diff) max_diff = d;
            }
        }
        EXPECT_EQ(mismatches, 0u)
            << "step " << i << ": SetRows decode logits are NOT bitwise-"
            << "identical to Cpy decode. mismatched=" << mismatches << "/"
            << a.size() << " max_abs_diff=" << max_diff << ". The write op is "
            << "the only difference between these graphs; a mismatch means "
            << "set_rows wrote different bytes (or rows) than cpy.";
        if (mismatches != 0) return;  // first divergent step tells the story
    }

    // Written-region byte equality: same rows, same bytes, every layer.
    const uint32_t pos_end = fpCpy.get_cache_pos(0);
    ASSERT_EQ(pos_end, fpSet.get_cache_pos(0));
    simple_kv_cache* kvA = fpCpy.snapshot_kv_cache();
    simple_kv_cache* kvB = fpSet.snapshot_kv_cache();
    ASSERT_NE(kvA, nullptr);
    ASSERT_NE(kvB, nullptr);
    auto bytesA = read_kv_bytes(kvA, pos_end);
    auto bytesB = read_kv_bytes(kvB, pos_end);
    ASSERT_EQ(bytesA.size(), bytesB.size());
    for (size_t i = 0; i < bytesA.size(); ++i)
        EXPECT_EQ(bytesA[i], bytesB[i])
            << "K/V cache span " << i << " (layer " << i / 2 << ", "
            << (i % 2 == 0 ? "K" : "V") << "): written bytes differ between "
            << "Cpy and SetRows writes over rows [0, " << pos_end << ").";
}

TEST(KvWriteSetRows, Qwen35_BitwiseLogitsAndCacheBytes) {
    run_write_mode_differential<Qwen35ForwardPass>(
        env_or("QWEN35_MODEL_PATH", "./models/Qwen3.5-0.8B-BF16.gguf"));
}

TEST(KvWriteSetRows, Gemma3_BitwiseLogitsAndCacheBytes) {
    run_write_mode_differential<Gemma3ForwardPass>(
        env_or("GEMMA3_MODEL_PATH", "./models/gemma-3-1b-it-BF16.gguf"));
}
