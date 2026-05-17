#include "decode_step.h"
#include "ggml-backend.h"
#include <cstdlib>
#include <cstring>

// Sparse LM head fires when |valid_set| < vocab / SPARSE_HEAD_FRACTION_DENOM.
// On Qwen 3.6 (vocab ~152k) this is ~19k — well above typical grammar-constrained
// sets (~tens to low hundreds). Tune this with measured Phase A benchmarks.
static constexpr uint32_t SPARSE_HEAD_FRACTION_DENOM = 8;

// Diagnostic: read once at process start. When QWENIUM_FORCE_DENSE=1 in the
// environment, all decode_step calls take the dense path regardless of how
// narrow the grammar set is. Lets us compare sparse-on/off tok/s with one
// binary.
static bool env_force_dense() {
    static const bool flag = [] {
        const char* e = std::getenv("QWENIUM_FORCE_DENSE");
        return e && std::strcmp(e, "1") == 0;
    }();
    return flag;
}

int32_t decode_step(
    ForwardPassBase*       fp,
    ggml_backend_sched_t   scheduler,
    qwenium::Sampler*      sampler,
    int32_t                token,
    uint32_t               slot,
    const std::vector<int32_t>&    history,
    const std::vector<std::string>& vocab,
    uint32_t               vocab_size,
    bool                   force_dense
) {
    // 0. BRIDGE: route to the legacy single-token run_prefill path when the
    //    unified decode graph can't be used. Two reasons, same destination:
    //      (a) tq_active()       — build_decoding_graph has no TurboQuant
    //          support (kv_cache_ is null under TQ → would segfault); the
    //          legacy run_prefill IS TQ-aware. (Option 1 pending.)
    //      (b) !has_decode_graph() — recipe's build_decoding_graph is
    //          unimplemented and throws (Gemma 1–4); single-token run_prefill
    //          is that recipe's only decode mechanism.
    //    This is exactly how decode worked before decode_step existed.
    //    NOTE: no sparse LM head on this path — run_prefill builds the full
    //    output. Grammar masking still applies via sampler->sample(). Deferred:
    //    collapse the two predicates into one intent-named query so a third
    //    reason doesn't spawn a third boolean.
    if (fp->tq_active() || !fp->has_decode_graph()) {
        const int32_t pos = static_cast<int32_t>(fp->get_cache_pos(slot));
        std::vector<float> logits =
            fp->run_prefill({token}, pos, slot, scheduler);
        // run_prefill returns logits for the single decoded position; take the
        // first vocab_size (matches the pre-decode_step chat.cpp behavior).
        std::vector<float> last(logits.begin(), logits.begin() + vocab_size);
        int32_t next_token =
            static_cast<int32_t>(sampler->sample(last, history, vocab));
        sampler->accept_token(next_token);
        return next_token;
    }

    // 1. Query grammar constraints before the forward pass.
    std::vector<int32_t> valid_ids = sampler->peek_valid_set();
    const bool use_sparse = !force_dense &&
                            !env_force_dense() &&
                            !valid_ids.empty() &&
                            valid_ids.size() < vocab_size / SPARSE_HEAD_FRACTION_DENOM;

    // 2. Arm sparse indices (or clear for dense path).
    fp->set_sparse_decode_ids(use_sparse ? valid_ids : std::vector<int32_t>{});

    // 3. Build → alloc → upload sparse indices → set inputs → compute.
    const int32_t           pos       = static_cast<int32_t>(fp->get_cache_pos(slot));
    const std::vector<int32_t>  tokens    = {token};
    const std::vector<uint32_t> slots     = {slot};
    const std::vector<int32_t>  positions = {pos};

    ggml_backend_sched_reset(scheduler);
    ggml_cgraph* gf = fp->build_decoding_graph(tokens, slots, positions);
    ggml_backend_sched_alloc_graph(scheduler, gf);
    fp->set_decode_inputs(gf, tokens, slots, positions);
    ggml_backend_sched_graph_compute(scheduler, gf);
    fp->advance_cache(1, slot);

    // 4. Extract logits and sample.
    std::vector<float> logits = fp->get_output_logits(gf);

    int32_t next_token;
    if (use_sparse) {
        next_token = sampler->sample_sparse(logits, valid_ids, history);
    } else {
        next_token = static_cast<int32_t>(sampler->sample(logits, history, vocab));
    }

    // 5. Advance grammar state.
    sampler->accept_token(next_token);

    return next_token;
}
