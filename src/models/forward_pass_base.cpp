#include "forward_pass_base.h"

#include <cstdlib>
#include <cstdio>

#include "../graph_inputs/routing_replay_input.h"
#include "../layers/attention.h"
#include "../layers/ffn.h"
#include "../layers/norm.h"
#include "../graph_inputs/sparse_head_input.h"
#include "../graph_inputs/output_ids_input.h"
#include "../graph_inputs/attn_mask_input.h"
#include "../graph_inputs/image_embeddings_input.h"

#include <map>
#include <memory>
#include <string>

#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <cmath>
#include <stdexcept>

// The arena default-constructs its buffer and context, so the constructor only
// binds the model references now.
ForwardPassBase::ForwardPassBase(const Model& model, const ModelMetadata* metadata)
    : meta_(*metadata), model_(model)
{
}

// The arena's destructor frees the context.
ForwardPassBase::~ForwardPassBase() = default;

void ForwardPassBase::reset_context() {
    image_spliced_ = false;
    arena_.reset();
    // NOTE: do NOT clear sparse_decode_ids_ here. reset_context is called
    // inside build_decoding_graph (between the caller's set_sparse_decode_ids
    // and build_output_head's read), so clearing would erase the indices the
    // caller just armed. Consume-on-use happens in set_prefill_inputs /
    // set_decode_inputs after SparseHeadInput uploads it.
}

ggml_cgraph* ForwardPassBase::new_graph() {
    return arena_.new_graph();
}

ggml_tensor* ForwardPassBase::embedding(ggml_cgraph* gf, const std::vector<int32_t>& tokens) {
    const size_t n_tokens = tokens.size();

    // 1. Create a 1D tensor from the input token IDs
    struct ggml_tensor* tokens_tensor = ggml_new_tensor_1d(
        arena_.ctx(),
        GGML_TYPE_I32,
        n_tokens
    );
    
    ggml_set_input(tokens_tensor);
    set_tensor_name(gf, tokens_tensor, "tokens");
    ggml_build_forward_expand(gf, tokens_tensor);
    // memcpy(tokens_tensor->data, tokens.data(), ggml_nbytes(tokens_tensor));

    // 2. Perform the embedding lookup using ggml_get_rows
    ggml_tensor * cur = ggml_get_rows(
        arena_.ctx(),
        model_.get_token_embedding_weight(),
        tokens_tensor
    );

    ggml_set_name(cur, "embed_lookup");
    return cur;
}

ggml_tensor* ForwardPassBase::build_norm(
    ggml_cgraph* gf,
    ggml_tensor* cur,
    ggml_tensor* mw,
    int il) const
{
    return build_rms_norm(arena_.ctx(), cur, mw, meta_.rms_norm_eps, il);
}


void ForwardPassBase::build_output_head(ggml_cgraph* gf, ggml_tensor* cur, ggml_tensor* valid_idx, bool gemma_final_norm, float final_softcap) {
    // Auto-create the sparse row-selection tensor from host-side ids if the
    // caller didn't supply one. Do NOT clear sparse_decode_ids_ here — it is
    // uploaded later by SparseHeadInput (via set_prefill/decode_inputs) and
    // cleared there (consume-on-use).
    if (valid_idx == nullptr && !sparse_decode_ids_.empty()) {
        valid_idx = ggml_new_tensor_1d(arena_.ctx(), GGML_TYPE_I32,
                                       static_cast<int64_t>(sparse_decode_ids_.size()));
        ggml_set_input(valid_idx);
        ggml_set_name(valid_idx, "valid_indices");
        ggml_build_forward_expand(gf, valid_idx);
        // Generalizes the former set_sparse_decode_ids/upload_sparse_indices
        // one-off into the typed-input set. Recipes populate graph_inputs_ in
        // their build_*_graph and call build_output_head after; this appends
        // the sparse slot only when the sparse path is armed.
        graph_inputs_.add(std::make_unique<SparseHeadInput>());
    }

    // Gemma's final norm is (x / rms(x)) * (1 + w); every other recipe uses
    // x * w. Default false keeps Qwen and all non-Gemma recipes byte-identical.
    cur = gemma_final_norm
        ? build_rms_norm_gemma(arena_.ctx(), cur, model_.get_output_norm_weight(),
                               meta_.rms_norm_eps, /*il=*/-1)
        : build_norm(gf, cur, model_.get_output_norm_weight(), -1);
    set_tensor_name(gf, cur, "final_norm");

    ggml_tensor* weight = model_.get_output_weight()
        ? model_.get_output_weight()
        : model_.get_token_embedding_weight();

    if (valid_idx) {
        weight = ggml_get_rows(arena_.ctx(), weight, valid_idx);
        ggml_set_name(weight, "output_weight_k");
    }
    cur = ggml_mul_mat(arena_.ctx(), weight, cur);
    // Gemma 2 final logit soft-capping (cap == 0 → off, byte-identical for all
    // non-Gemma-2 recipes). Applied before the "logits" name so get_output_logits
    // reads the capped values, matching the recipe's prefill head.
    if (final_softcap > 0.0f) {
        cur = build_softcap(arena_.ctx(), cur, final_softcap);
    }
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
}

ggml_tensor* ForwardPassBase::build_out_ids_slice(ggml_cgraph* gf, ggml_tensor* cur) {
    // Explicit dense-reference path (differential seam). Not a silent fallback:
    // the caller chose it via set_slice_prefill_head(false).
    if (!policy_.slice_prefill_head) {
        return cur;
    }

    // "out_ids": int32 token-position selector. Width 1 — last token only
    // (OutputIdsInput fills it with n_rows-1 at set time). Gathering on the
    // [hidden, n_tokens] hidden state elides the discarded first n_tokens-1
    // head rows. Composes with build_output_head's vocab-axis weight slice:
    // different tensor, different axis, order-independent.
    ggml_tensor* out_ids = ggml_new_tensor_1d(arena_.ctx(), GGML_TYPE_I32, 1);
    ggml_set_input(out_ids);
    ggml_set_name(out_ids, "out_ids");
    ggml_build_forward_expand(gf, out_ids);
    graph_inputs_.add(std::make_unique<OutputIdsInput>());

    ggml_tensor* sliced = ggml_get_rows(arena_.ctx(), cur, out_ids);
    ggml_set_name(sliced, "out_ids_slice");
    return sliced;
}

ggml_tensor* ForwardPassBase::build_image_substitution(
    ggml_cgraph* gf, ggml_tensor* inpL, std::vector<float>&& embd,
    int32_t span_start, uint32_t n_img, int hidden_dim, size_t n_tokens)
{
    const size_t want = static_cast<size_t>(hidden_dim) * n_img;
    if (embd.size() != want)
        throw std::runtime_error(
            "build_image_substitution: slot 'image_embeddings': expected " +
            std::to_string(want) + " floats (hidden_dim=" +
            std::to_string(hidden_dim) + " * n_img=" + std::to_string(n_img) +
            "), got: " + std::to_string(embd.size()));
    if (span_start < 0 ||
        static_cast<size_t>(span_start) + n_img > n_tokens)
        throw std::runtime_error(
            "build_image_substitution: slot 'image_span': expected span within "
            "[0, " + std::to_string(n_tokens) + "), got: start=" +
            std::to_string(span_start) + " n_img=" + std::to_string(n_img));

    ggml_tensor* img_in = ggml_new_tensor_2d(
        arena_.ctx(), GGML_TYPE_F32, hidden_dim, static_cast<int64_t>(n_img));
    ggml_set_input(img_in);
    set_tensor_name(gf, img_in, "image_embeddings");
    ggml_build_forward_expand(gf, img_in);

    // inpL[:, span : span+n_img] = img_in (one op; the surviving text columns
    // keep their sqrt(d_model) scale, the image columns enter unscaled).
    inpL = ggml_set_2d(arena_.ctx(), inpL, img_in, inpL->nb[1],
                       static_cast<size_t>(span_start) * inpL->nb[1]);
    set_tensor_name(gf, inpL, "inpL_image_subst");
    // Pin the substituted residual as a graph output. Without this, galloc
    // reuses this intermediate's buffer across the server's alternating graph
    // shapes, so the 2nd+ image request reads stale memory here and degenerates
    // (token-soup). Marking it an output keeps its buffer live and makes
    // multi-request image prefill deterministic. The single owner of this pin —
    // every vision recipe routes through here, so it cannot be forgotten by one.
    ggml_set_output(inpL);
    ggml_build_forward_expand(gf, inpL);

    graph_inputs_.add(std::make_unique<ImageEmbeddingsInput>(std::move(embd)));
    image_spliced_ = true;  // guarded in set_prefill_inputs
    return inpL;
}

std::vector<ggml_tensor*> ForwardPassBase::build_decode_layer_masks(
    ggml_cgraph* gf,
    const std::vector<uint32_t>& layer_windows,
    uint32_t n_kv_len, uint32_t n_tokens)
{
    std::vector<ggml_tensor*>        per_layer(layer_windows.size(), nullptr);
    std::map<uint32_t, ggml_tensor*> by_window;  // distinct window -> shared mask

    for (size_t il = 0; il < layer_windows.size(); ++il) {
        const uint32_t w  = layer_windows[il];
        auto           it = by_window.find(w);
        if (it == by_window.end()) {
            // One tensor + one typed input per distinct window. The mask body is
            // identical for every layer of this window within a decode step
            // (same positions/slots/n_kv), so sharing is bit-for-bit equivalent
            // to the former tensor-per-layer while collapsing the input count.
            ggml_tensor* m = ggml_new_tensor_4d(arena_.ctx(), GGML_TYPE_F32,
                                                n_kv_len, 1, 1, n_tokens);
            ggml_set_input(m);
            const std::string name = "kq_mask.w" + std::to_string(w);
            ggml_set_name(m, name.c_str());
            ggml_build_forward_expand(gf, m);
            graph_inputs_.add(std::make_unique<AttnMaskInput>(name, w));
            it = by_window.emplace(w, m).first;
        }
        per_layer[il] = it->second;
    }
    return per_layer;
}

void ForwardPassBase::set_tensor_name(ggml_cgraph* gf, ggml_tensor* tensor, const char* name, int il) const {
    if (il != -1) {
        char new_name[128];
        snprintf(new_name, sizeof(new_name), "%s.%d", name, il);
        ggml_set_name(tensor, new_name);
    } else {
        ggml_set_name(tensor, name);
    }
}

// Get output from GPU
std::vector<float> ForwardPassBase::get_output_logits(ggml_cgraph* gf) {
    ggml_tensor* logits_gpu = ggml_graph_get_tensor(gf, "logits");
    if (!logits_gpu) {
        throw std::runtime_error("logits tensor not found in graph");
    }
    
    size_t logits_size = ggml_nbytes(logits_gpu);
    std::vector<float> logits_cpu(logits_size / sizeof(float));
    ggml_backend_tensor_get(logits_gpu, logits_cpu.data(), 0, logits_size);

    return logits_cpu;
}

std::vector<float> ForwardPassBase::get_output_hidden(ggml_cgraph* gf) {
    ggml_tensor* h = ggml_graph_get_tensor(gf, "hidden_out");
    if (!h) {
        throw std::runtime_error(
            "get_output_hidden: 'hidden_out' tensor not found in graph — "
            "expected set_output_hidden(true) before build, actual absent");
    }
    size_t n = ggml_nbytes(h);
    std::vector<float> out(n / sizeof(float));
    ggml_backend_tensor_get(h, out.data(), 0, n);
    return out;
}

// ── Lens tap (docs/plan-qemmi-lens.md P1/A1) ─────────────────────────────────
// Mark each armed attention layer's post-softmax row as a retained graph output.
// The tap tensors are named `kq_soft.<il>` by layers/attention.cpp on every
// recipe; marking an existing node as an output adds no compute, so the tap-off
// path (empty layer set → this is a no-op) is byte-identical to today.
void ForwardPassBase::mark_attention_taps(ggml_cgraph* gf) {
    tap_plan_ = ReadbackPlan::None;
    tap_plan_graph_ = nullptr;
    const std::vector<int>& heads = policy_.attention_tap_heads;
    for (int il : policy_.attention_taps) {
        std::string nm = "kq_soft." + std::to_string(il);
        ggml_tensor* ts = ggml_graph_get_tensor(gf, nm.c_str());
        if (!ts)
            throw std::runtime_error(
                "mark_attention_taps: attention-tap tensor '" + nm +
                "' expected in graph, actual absent — layer " +
                std::to_string(il) + " is not an attention layer of this "
                "recipe (or the graph has no such block).");
        if (heads.empty()) {
            ggml_set_output(ts);
            ggml_build_forward_expand(gf, ts);
            continue;
        }
        // Head-selected tap. kq_soft is [n_kv, n_q, n_head, 1] and contiguous,
        // so one head is the contiguous block at offset h * nb[2]; ggml_cont
        // copies it into a tensor of its own. kq_soft stays an ordinary
        // intermediate, free for galloc to reuse once the copies are made.
        if (ts->ne[3] != 1 || !ggml_is_contiguous(ts))
            throw std::runtime_error(
                "mark_attention_taps: '" + nm + "' expected contiguous with ne[3] == 1 "
                "for a head-selected tap, actual ne[3]=" + std::to_string(ts->ne[3]) +
                (ggml_is_contiguous(ts) ? "" : " and non-contiguous"));
        for (size_t a = 0; a < heads.size(); ++a) {
            const int h = heads[a];
            if (h < 0 || h >= (int)ts->ne[2])
                throw std::runtime_error(
                    "mark_attention_taps: tap head expected within [0, " +
                    std::to_string(ts->ne[2]) + ") on layer " + std::to_string(il) +
                    ", actual " + std::to_string(h));
            for (size_t b = 0; b < a; ++b)
                if (heads[b] == h)
                    throw std::runtime_error(
                        "mark_attention_taps: tap heads expected distinct, actual head " +
                        std::to_string(h) + " listed twice");
            ggml_tensor* one = ggml_view_3d(arena_.ctx(), ts, ts->ne[0], ts->ne[1], 1,
                                            ts->nb[1], ts->nb[2], (size_t)h * ts->nb[2]);
            ggml_tensor* sel = ggml_cont(arena_.ctx(), one);
            const std::string sn = "kq_tap." + std::to_string(il) + "." + std::to_string(h);
            ggml_set_name(sel, sn.c_str());
            ggml_set_output(sel);
            ggml_build_forward_expand(gf, sel);
        }
    }
    if (!policy_.attention_taps.empty()) {
        tap_plan_ = ReadbackPlan::Marked;
        tap_plan_graph_ = gf;
    }
}

// A memory plan made for THIS tapped graph — see the header for why a plain
// ggml_backend_sched_alloc_graph can hand a tapped graph another graph's plan.
//
// How: first reserve a one-node graph, which replaces the scheduler's cached
// plan (and never shrinks its buffers — galloc only grows them); the tapped
// graph's node count then differs from the cached plan, so its ONE alloc makes
// a fresh plan from its own output flags. Not ggml_backend_sched_reserve(gf)
// followed by alloc(gf): each of those splits `gf`, and the split rewrites
// node->src[j] to per-backend input copies living in the scheduler's split
// context, which the second split frees — the graph then computes from stale
// copies (measured: TapOffByteIdentical logits off by up to 23 on every
// recipe). The one-node graph is built and freed per call, so the same
// rewrite can never leave a stale pointer behind in it.
void ForwardPassBase::alloc_readback_graph(ggml_backend_sched_t sched, ggml_cgraph* gf) {
    ggml_backend_sched_reset(sched);
    // What was MARKED on this graph decides, not what is armed: a caller may
    // prefill untapped with taps armed for its next decode (run_prefill does
    // not read taps). A reader whose graph was not marked and planned here is
    // refused at the read, which is where a wrong plan would do harm.
    const bool taps  = tap_plan_ == ReadbackPlan::Marked && tap_plan_graph_ == gf;
    const bool route = route_plan_ == ReadbackPlan::Marked && route_plan_graph_ == gf;
    if (!taps && !route) {
        if (!ggml_backend_sched_alloc_graph(sched, gf))
            throw std::runtime_error("alloc_readback_graph: graph alloc expected to succeed, actual failure (nothing marked)");
        return;
    }
    {
        ggml_init_params ip = {ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(4, false),
                               nullptr, /*no_alloc=*/true};
        ggml_context* ctx = ggml_init(ip);
        if (!ctx) throw std::runtime_error("alloc_readback_graph: a context for the one-node plan expected, actual null");
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_cgraph* one = ggml_new_graph_custom(ctx, 4, false);
        ggml_build_forward_expand(one, ggml_scale(ctx, a, 1.0f));
        const bool ok = ggml_backend_sched_reserve(sched, one);   // ends with a sched reset
        ggml_free(ctx);
        if (!ok) throw std::runtime_error("alloc_readback_graph: reserving the one-node plan expected to succeed, actual failure");
    }
    if (!ggml_backend_sched_alloc_graph(sched, gf))
        throw std::runtime_error("alloc_readback_graph: graph alloc on a fresh plan expected to succeed, actual failure");
    if (taps)  tap_plan_ = ReadbackPlan::Planned;
    if (route) route_plan_ = ReadbackPlan::Planned;
}

void ForwardPassBase::add_routing_replay_input() {
    if (policy_.routing_replay)
        graph_inputs_.add(std::make_unique<RoutingReplayInput>(policy_.routing_replay));
}

void ForwardPassBase::mark_moe_routing(ggml_cgraph* gf) {
    route_plan_ = ReadbackPlan::None;
    route_plan_graph_ = nullptr;
    int marked = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor* t = ggml_graph_node(gf, i);
        if (std::string(ggml_get_name(t)).rfind("moe_idx.", 0) != 0) continue;
        // `moe_idx.<il>` is a ggml_view_2d of the argsort output carrying the
        // PARENT's row stride. Pinning and reading the view directly copies
        // ggml_nbytes() contiguous bytes, which for a strided view is neither
        // the right bytes nor, at the last row, necessarily mapped memory. Pin
        // the contiguous parent and slice host-side in read_moe_routing.
        ggml_tensor* src = t->view_src ? t->view_src : t;
        ggml_set_output(src);
        ggml_build_forward_expand(gf, src);
        ++marked;
    }
    // A dense graph marks nothing and stays on the plain alloc path.
    if (marked > 0) {
        route_plan_ = ReadbackPlan::Marked;
        route_plan_graph_ = gf;
    }
}

int ForwardPassBase::read_moe_routing(ggml_cgraph* gf, RoutingTrace& trace, int pos) {
    bool has_routing = false;
    for (int i = 0; i < ggml_graph_n_nodes(gf) && !has_routing; ++i)
        has_routing = std::string(ggml_get_name(ggml_graph_node(gf, i))).rfind("moe_idx.", 0) == 0;
    if (has_routing) {
        if (route_plan_ != ReadbackPlan::Planned || route_plan_graph_ != gf)
            throw std::runtime_error(
                std::string("read_moe_routing: expected a graph marked with mark_moe_routing(gf) and "
                            "allocated with alloc_readback_graph(sched, gf), actual ") +
                (route_plan_ == ReadbackPlan::Marked && route_plan_graph_ == gf
                     ? "marked but allocated some other way"
                     : "a graph that was not marked for this read") +
                " — on a reused ggml memory plan moe_idx can hold another tensor's bytes.");
        route_plan_ = ReadbackPlan::None;
        route_plan_graph_ = nullptr;
    }
    int found = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor* t = ggml_graph_node(gf, i);
        const std::string nm = ggml_get_name(t);
        if (nm.rfind("moe_idx.", 0) != 0) continue;
        const int il    = std::stoi(nm.substr(8));
        const int top_k = (int)t->ne[0];
        const int n_tok = (int)t->ne[1];
        ggml_tensor* src = t->view_src ? t->view_src : t;
        const int n_exp = (int)src->ne[0];

        std::vector<int32_t> full((size_t)n_exp * n_tok);
        ggml_backend_tensor_get(src, full.data(), 0, ggml_nbytes(src));
        for (int r = 0; r < n_tok; ++r)
            trace.write(il, (size_t)(pos + r), full.data() + (size_t)r * n_exp, top_k);
        found++;
    }
    // Zero is a legitimate answer, not an error: a DENSE recipe has no routing
    // to capture. The dangerous case — asking to replay a graph that was built
    // with RoutingSource::Router — is caught on the other side, by
    // RoutingReplayInput, which refuses a graph with no 'moe_routing' slots.
    // Callers that require an MoE check the returned count.
    return found;
}

std::vector<ForwardPassBase::AttentionTap>
ForwardPassBase::get_attention_taps(ggml_cgraph* gf) {
    if (!policy_.attention_taps.empty()) {
        if (tap_plan_ != ReadbackPlan::Planned || tap_plan_graph_ != gf)
            throw std::runtime_error(
                std::string("get_attention_taps: expected a graph marked with mark_attention_taps(gf) and "
                            "allocated with alloc_readback_graph(sched, gf), actual ") +
                (tap_plan_ == ReadbackPlan::Marked && tap_plan_graph_ == gf
                     ? "marked but allocated some other way"
                     : "a graph that was not marked for this read") +
                " — on a reused ggml memory plan the tap can hold another layer's attention.");
        tap_plan_ = ReadbackPlan::None;
        tap_plan_graph_ = nullptr;
    }
    std::vector<AttentionTap> out;
    out.reserve(policy_.attention_taps.size());
    const std::vector<int>& heads = policy_.attention_tap_heads;
    for (int il : policy_.attention_taps) {
        std::string nm = "kq_soft." + std::to_string(il);
        AttentionTap tap;
        tap.layer = il;
        if (heads.empty()) {
            ggml_tensor* ts = ggml_graph_get_tensor(gf, nm.c_str());
            if (!ts)
                throw std::runtime_error(
                    "get_attention_taps: attention-tap tensor '" + nm +
                    "' expected in graph, actual absent — call "
                    "mark_attention_taps(gf) after build_decoding_graph and before "
                    "graph alloc.");
            tap.n_kv   = (int)ts->ne[0];
            tap.n_q    = (int)ts->ne[1];   // 1 at decode; >1 at a tapped prefill block
            tap.n_head = (int)ts->ne[2];   // shape [n_kv, n_q, n_head, 1]
            tap.rows.resize((size_t)tap.n_kv * tap.n_q * tap.n_head);
            ggml_backend_tensor_get(ts, tap.rows.data(), 0, ggml_nbytes(ts));
            tap.heads.resize((size_t)tap.n_head);
            for (int h = 0; h < tap.n_head; ++h) tap.heads[(size_t)h] = h;
        } else {
            // Head-selected: one `kq_tap.<il>.<h>` copy per head, laid out as
            // consecutive blocks — the same [block][q][kv] order as a full tap.
            tap.heads  = heads;
            tap.n_head = (int)heads.size();
            for (size_t b = 0; b < heads.size(); ++b) {
                const std::string sn = "kq_tap." + std::to_string(il) + "." + std::to_string(heads[b]);
                ggml_tensor* sel = ggml_graph_get_tensor(gf, sn.c_str());
                if (!sel)
                    throw std::runtime_error(
                        "get_attention_taps: head-selected tap '" + sn +
                        "' expected in graph, actual absent — call "
                        "mark_attention_taps(gf) after the graph build and before "
                        "graph alloc, with the same head list armed.");
                if (b == 0) {
                    tap.n_kv = (int)sel->ne[0];
                    tap.n_q  = (int)sel->ne[1];
                    tap.rows.resize((size_t)tap.n_kv * tap.n_q * tap.n_head);
                }
                const size_t block = (size_t)tap.n_kv * tap.n_q;
                if (ggml_nbytes(sel) != block * sizeof(float))
                    throw std::runtime_error(
                        "get_attention_taps: '" + sn + "' expected " +
                        std::to_string(block * sizeof(float)) + " bytes, actual " +
                        std::to_string(ggml_nbytes(sel)));
                ggml_backend_tensor_get(sel, tap.rows.data() + b * block, 0, ggml_nbytes(sel));
            }
        }
        // ── The tap must BE a softmax. Fail loud if it is not. ──────────────
        // Post-softmax attention weights live in [0, 1] — always, with or
        // without sinks. A value outside that range means these bytes are not
        // this pass's softmax output, and the only way that happens is memory
        // the graph reused underneath a tensor we flagged OUTPUT.
        //
        // That is not hypothetical. Measured 2026-09-18: on a --lens-verify-only
        // server, a preceding /v1/locate left a galloc plan in which
        // `kq_soft.<citation_layer>` was NOT an output (locate marks one layer,
        // verify marks two). ggml_gallocr_needs_realloc keys on node count and
        // node SIZES — never on the OUTPUT flag — and verify's graph has the
        // same node count and strictly smaller tensors, so galloc reused
        // locate's plan verbatim and handed the tap's block to DeltaNet layers
        // 4 and 6. Every later /v1/verify then returned a confident, well-formed
        // report computed from pre-softmax scores: body_mass went negative and
        // four of seven badges flipped. No error, no warning.
        //
        // The scheduler split that caused it is fixed at the call site. This
        // guard is the backstop, because the failure is SILENT and the product
        // is the receipt: a lens that cannot tell whether it read its own
        // attention should refuse, not report.
        for (size_t i = 0; i < tap.rows.size(); ++i) {
            const float v = tap.rows[i];
            if (v >= -1e-3f && v <= 1.0f + 1e-3f) continue;
            throw std::runtime_error(
                "get_attention_taps: tap 'kq_soft." + std::to_string(il) +
                "' expected post-softmax weights within [0, 1], actual " +
                std::to_string(v) + " at element " + std::to_string(i) + " of " +
                std::to_string(tap.rows.size()) + " (shape [" +
                std::to_string(tap.n_kv) + "," + std::to_string(tap.n_q) + "," +
                std::to_string(tap.n_head) + "]) — these bytes are not this "
                "pass's attention: the tap's memory was overwritten during the "
                "pass although alloc_readback_graph planned it for this graph. A "
                "backstop — report it; do not work around it.");
        }
        out.push_back(std::move(tap));
    }
    return out;
}

// Get output logits for a specific batch slot
std::vector<float> ForwardPassBase::get_output_logits_for_slot(ggml_cgraph* gf, uint32_t slot_index) {
    ggml_tensor* logits_gpu = ggml_graph_get_tensor(gf, "logits");
    if (!logits_gpu) {
        throw std::runtime_error("logits tensor not found in graph");
    }
    
    // logits shape: [vocab_size, batch_size]
    uint32_t vocab_size = logits_gpu->ne[0];
    uint32_t batch_size = logits_gpu->ne[1];
    
    if (slot_index >= batch_size) {
        throw std::out_of_range("slot_index out of bounds for logits tensor");
    }
    
    size_t offset_bytes = slot_index * vocab_size * sizeof(float);
    std::vector<float> logits(vocab_size);
    
    ggml_backend_tensor_get(logits_gpu, logits.data(), offset_bytes, vocab_size * sizeof(float));
    
    return logits;
}