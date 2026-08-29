#include "decode_graph_cache.h"
#include "graph_compute.h"

#include <stdexcept>
#include <string>

#include "model.h"
#include "../models/forward_pass_base.h"
#include "../state/kv_cache_simple.h"

#include "ggml.h"

void enable_persistent_decode(ForwardPassBase* fp) {
    if (!fp->supports_persistent_decode())
        throw std::runtime_error(
            "enable_persistent_decode: supports_persistent_decode expected "
            "true, actual false — this recipe has no value-driven decode graph "
            "(--persistent-graph requires qwen35/qwen36/gemma3). Refusing "
            "rather than building an un-bucketed graph the cache would mis-key.");
    fp->set_kv_write_mode(ForwardPassBase::KvWriteMode::SetRows);
    fp->set_decode_kv_bucket(DecodeGraphCache::kBucket);
}

DecodeGraphCache::DecodeGraphCache(Model& model, ForwardPassBase* fp) : fp_(fp) {
    // Dedicated scheduler over the same backends — isolates the persistent
    // decode allocation from main-scheduler graph work (MTP/image-prefill
    // precedent). parallel flag matches Model's own scheduler construction.
    if (model.has_metal_backend()) {
        ggml_backend_t backends[] = {model.get_backend_metal(),
                                     model.get_backend_cpu()};
        sched_ = ggml_backend_sched_new(backends, nullptr, 2, FP_GRAPH_SIZE,
                                        /*parallel=*/true, /*op_offload=*/false);
    } else {
        ggml_backend_t backends[] = {model.get_backend_cpu()};
        sched_ = ggml_backend_sched_new(backends, nullptr, 1, FP_GRAPH_SIZE,
                                        /*parallel=*/false, /*op_offload=*/false);
    }
    if (!sched_)
        throw std::runtime_error(
            "DecodeGraphCache: ggml_backend_sched_new expected a scheduler, "
            "got null");
    owns_sched_ = true;
}

DecodeGraphCache::~DecodeGraphCache() {
    if (owns_sched_ && sched_) ggml_backend_sched_free(sched_);
}

ggml_cgraph* DecodeGraphCache::step(const std::vector<int32_t>& tokens,
                                    const std::vector<uint32_t>& slots,
                                    const std::vector<int32_t>& positions) {
    // Bucketed KV width for this step's deepest slot — the key axis besides the
    // active-slot set. Mirrors the recipe's own decode_kv_len(); a divergence
    // here would mis-key the cache, so it is verified fail-loud after build.
    uint32_t max_pos = 0;
    for (uint32_t s : slots) {
        uint32_t p = fp_->get_cache_pos(s);
        if (p > max_pos) max_pos = p;
    }
    // Bucket is the fp's own decode bucket (single source of truth — the recipe
    // sizes its mask/gather with the same value via decode_kv_len).
    // enable_persistent_decode() sets it to kBucket; the gate uses a small
    // bucket to cross boundaries cheaply. Bucket 0 (exact) degrades to
    // rebuild-per-step.
    const uint32_t bucket = fp_->decode_kv_bucket();
    uint32_t want_nkv = (bucket == 0)
        ? (max_pos + 1)
        : ((max_pos + 1 + bucket - 1) / bucket) * bucket;
    // Match decode_kv_len's cap so the key equals the recipe's built width even
    // near context-full (else the fail-loud mask-width check would spuriously
    // fire on the last bucket).
    if (simple_kv_cache* kv = fp_->snapshot_kv_cache()) {
        const uint32_t n_ctx_max = kv->get_n_ctx_max();
        if (want_nkv > n_ctx_max) want_nkv = n_ctx_max;
    }

    const bool key_hit = has_cached_ &&
                         key_slots_ == slots &&
                         key_bucket_nkv_ == want_nkv;

    if (!key_hit) {
        // MISS: reset + build + alloc on the dedicated scheduler (~13 ms).
        ggml_backend_sched_reset(sched_);
        gf_ = fp_->build_decoding_graph(tokens, slots, positions);
        if (!ggml_backend_sched_alloc_graph(sched_, gf_))
            throw std::runtime_error(
                "DecodeGraphCache::step: ggml_backend_sched_alloc_graph "
                "expected success, got failure (persistent decode graph, "
                "n_kv bucket " + std::to_string(want_nkv) + ")");
        // Fail-loud: the recipe must have honored the bucket we keyed on, or a
        // later hit would set inputs into a mis-sized graph.
        ggml_tensor* mask = ggml_graph_get_tensor(gf_, "kq_mask_b");
        if (!mask) {  // Gemma names masks per window (kq_mask.w*)
            mask = ggml_graph_get_tensor(gf_, "kq_mask.w0");
        }
        if (mask && static_cast<uint32_t>(mask->ne[0]) != want_nkv)
            throw std::runtime_error(
                "DecodeGraphCache::step: keyed n_kv bucket expected " +
                std::to_string(want_nkv) + ", recipe built mask width " +
                std::to_string(mask->ne[0]) + " — bucket seam disagreement");
        key_slots_      = slots;
        key_bucket_nkv_ = want_nkv;
        has_cached_     = true;
        ++rebuilds_;
    } else {
        ++reuses_;
    }

    // HIT and MISS both end the same way: refill typed inputs + recompute. On a
    // hit this is the ONLY work — no reset, no build, no alloc.
    fp_->set_decode_inputs(gf_, tokens, slots, positions);
    qinf::engine::require_compute_success(
        ggml_backend_sched_graph_compute(sched_, gf_), "decode_graph_cache");
    return gf_;
}
