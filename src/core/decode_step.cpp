#include "decode_step.h"
#include "decode_plan.h"
#include "decode_graph_cache.h"
#include "ggml-backend.h"
#include <algorithm>

// Sparse LM head fires when |valid_set| < vocab / SPARSE_HEAD_FRACTION_DENOM.
// On Qwen 3.6 (vocab ~152k) this is ~19k — well above typical grammar-constrained
// sets (~tens to low hundreds).
static constexpr uint32_t SPARSE_HEAD_FRACTION_DENOM = 8;

int32_t decode_step(
    ForwardPassBase*       fp,
    ggml_backend_sched_t   scheduler,
    qwenium::Sampler*      sampler,
    int32_t                token,
    uint32_t               slot,
    const std::vector<int32_t>&    history,
    const std::vector<std::string>& vocab,
    uint32_t               vocab_size,
    bool                   force_dense,
    std::vector<int32_t>*  forced_run,
    DecodeGraphCache*      graph_cache
) {
    if (forced_run) forced_run->clear();

    // Resolve the decode-path decision ONCE. Every branch below reads plan.*
    const DecodePlan plan =
        resolve_decode_plan(fp, /*forced_run_enabled=*/forced_run != nullptr,
                            force_dense);

    // Forced-token elision: if the sampler (usually because of the grammar) leaves exactly one valid
    // token, the next token is determined; we do NOT need a forward pass to
    // know it. Roll the deterministic run, then advance model state over it
    // in ONE feed_tokens dispatch (KV append + recurrent overwrite). 
    // Skips forward, head, r sample. The forced path never builds the decode graph.
    std::vector<int32_t> valid_ids;
    bool have_valid = false;

    if (plan.allow_forced_elision) {
        std::vector<int32_t> valid_set = sampler->peek_valid_set();
        if (valid_set.size() == 1) {
            const auto& stop = sampler->eos_token_ids();
            auto is_stop = [&](int32_t t) {
                return std::find(stop.begin(), stop.end(), t) != stop.end();
            };
            // Lets bound the number of forced runs (grammar single tokens)
            constexpr size_t kForcedRunCap = 64;

            std::vector<int32_t> run;
            int32_t t = valid_set[0];
            while (true) {
                run.push_back(t);
                sampler->accept_token(t);          // advance grammar
                if (is_stop(t) || run.size() >= kForcedRunCap) break;
                valid_set = sampler->peek_valid_set();
                if (valid_set.size() != 1) break;          // reached a branch
                t = valid_set[0];
            }

            // Ingest the input `token` plus every forced token except the last.
            std::vector<int32_t> feed;
            feed.reserve(run.size());
            feed.push_back(token);
            feed.insert(feed.end(), run.begin(), run.end() - 1);
            // feed_tokens rebuilds a prefill graph → resets fp's context, so any
            // retained persistent decode graph is now dangling.
            if (graph_cache) graph_cache->invalidate();
            fp->feed_tokens(feed, slot, scheduler);

            forced_run->assign(run.begin(), run.end() - 1);
            return run.back();
        }
        // Reuse the valid set, don't peek again
        valid_ids = std::move(valid_set);
        have_valid = true;
    }

    // 0. BRIDGE route: single-token run_prefill
    // No sparse LM head here — run_prefill builds the full dense output;
    if (plan.route == DecodeRoute::Bridge) {
        if (graph_cache) graph_cache->invalidate();  // run_prefill resets fp ctx
        const int32_t pos = static_cast<int32_t>(fp->get_cache_pos(slot));
        std::vector<float> logits =
            fp->run_prefill({token}, pos, slot, scheduler);
        // run_prefill returns logits for the single decoded position; take the
        // first vocab_size
        std::vector<float> last(logits.begin(), logits.begin() + vocab_size);
        int32_t next_token =
            static_cast<int32_t>(sampler->sample(last, history, vocab));
        sampler->accept_token(next_token);
        return next_token;
    }

    // 1. Query next valid tokens before the forward pass (we don't do it again if elision already did it)
    if (!have_valid) valid_ids = sampler->peek_valid_set();
    const bool use_sparse = plan.sparse_head_allowed &&
                            !valid_ids.empty() &&
                            valid_ids.size() < vocab_size / SPARSE_HEAD_FRACTION_DENOM;

    // 2. Arm sparse indices (or clear for dense path).
    fp->set_sparse_decode_ids(use_sparse ? valid_ids : std::vector<int32_t>{});

    // 3. Build -> alloc -> set inputs (sparse indices uploaded as part of
    //    the typed input set via SparseHeadInput) -> compute.
    //
    // Persistent path (opt-in --persistent-graph): dense steps reuse one
    // built+allocated graph across a bucket of positions on a dedicated
    // scheduler (DecodeGraphCache), skipping the per-step ~12 ms galloc replan.
    // Sparse steps have a per-step structure (grammar-narrowed head), so they
    // always take the rebuild path — and MUST invalidate the retained graph,
    // because build_decoding_graph resets fp's ggml_context out from under it.
    const int32_t           pos       = static_cast<int32_t>(fp->get_cache_pos(slot));
    const std::vector<int32_t>  tokens    = {token};
    const std::vector<uint32_t> slots     = {slot};
    const std::vector<int32_t>  positions = {pos};

    ggml_cgraph* gf;
    if (graph_cache && !use_sparse) {
        gf = graph_cache->step(tokens, slots, positions);  // dedicated scheduler
    } else {
        if (graph_cache) graph_cache->invalidate();
        ggml_backend_sched_reset(scheduler);
        gf = fp->build_decoding_graph(tokens, slots, positions);
        ggml_backend_sched_alloc_graph(scheduler, gf);
        fp->set_decode_inputs(gf, tokens, slots, positions);
        ggml_backend_sched_graph_compute(scheduler, gf);
    }
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
