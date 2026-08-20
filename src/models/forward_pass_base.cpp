#include "forward_pass_base.h"
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

ForwardPassBase::ForwardPassBase(const Model& model, const ModelMetadata* metadata)
    : meta_(*metadata), model_(model), ctx_(nullptr)
{
        // Pre-allocate persistent buffer for graph metadata
        ctx_buffer_.resize(FP_GRAPH_SIZE_METADATA);

        struct ggml_init_params params = {
            .mem_size   = ctx_buffer_.size(),
            .mem_buffer = ctx_buffer_.data(),
            .no_alloc   = true,
        };
        ctx_ = ggml_init(params);
}

ForwardPassBase::~ForwardPassBase() {
    if (ctx_) {
        ggml_free(ctx_);
    }
}

void ForwardPassBase::reset_context() {
    if (ctx_) {
        ggml_free(ctx_);
    }
    struct ggml_init_params params = {
        .mem_size   = ctx_buffer_.size(),
        .mem_buffer = ctx_buffer_.data(),
        .no_alloc   = true,
    };
    ctx_ = ggml_init(params);
    // NOTE: do NOT clear sparse_decode_ids_ here. reset_context is called
    // inside build_decoding_graph (between the caller's set_sparse_decode_ids
    // and build_output_head's read), so clearing would erase the indices the
    // caller just armed. Consume-on-use happens in set_prefill_inputs /
    // set_decode_inputs after SparseHeadInput uploads it.
}

ggml_cgraph* ForwardPassBase::new_graph() {
    return ggml_new_graph_custom(ctx_, FP_GRAPH_SIZE, false);
}

ggml_tensor* ForwardPassBase::embedding(ggml_cgraph* gf, const std::vector<int32_t>& tokens) {
    const size_t n_tokens = tokens.size();

    // 1. Create a 1D tensor from the input token IDs
    struct ggml_tensor* tokens_tensor = ggml_new_tensor_1d(
        ctx_,
        GGML_TYPE_I32,
        n_tokens
    );
    
    ggml_set_input(tokens_tensor);
    set_tensor_name(gf, tokens_tensor, "tokens");
    ggml_build_forward_expand(gf, tokens_tensor);
    // memcpy(tokens_tensor->data, tokens.data(), ggml_nbytes(tokens_tensor));

    // 2. Perform the embedding lookup using ggml_get_rows
    ggml_tensor * cur = ggml_get_rows(
        ctx_,
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
    return build_rms_norm(ctx_, cur, mw, meta_.rms_norm_eps, il);
}

// Thin wrapper — implementation lives in src/layers/attention.cpp.
// qwen35.cpp calls this via the base class; all new code
// should call the free function ::build_attn_mha directly.
ggml_tensor* ForwardPassBase::build_attn_mha(
    ggml_cgraph* gf,
    ggml_tensor* q,
    ggml_tensor* k,
    ggml_tensor* v,
    ggml_tensor* kq_mask,
    ggml_tensor* sinks,
    float kq_scale,
    uint32_t pos,
    int il) const
{
    return ::build_attn_mha(ctx_, gf, q, k, v, kq_mask, sinks, kq_scale, pos, il);
}

void ForwardPassBase::build_output_head(ggml_cgraph* gf, ggml_tensor* cur, ggml_tensor* valid_idx, bool gemma_final_norm, float final_softcap) {
    // Auto-create the sparse row-selection tensor from host-side ids if the
    // caller didn't supply one. Do NOT clear sparse_decode_ids_ here — it is
    // uploaded later by SparseHeadInput (via set_prefill/decode_inputs) and
    // cleared there (consume-on-use).
    if (valid_idx == nullptr && !sparse_decode_ids_.empty()) {
        valid_idx = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32,
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
        ? build_rms_norm_gemma(ctx_, cur, model_.get_output_norm_weight(),
                               meta_.rms_norm_eps, /*il=*/-1)
        : build_norm(gf, cur, model_.get_output_norm_weight(), -1);
    set_tensor_name(gf, cur, "final_norm");

    ggml_tensor* weight = model_.get_output_weight()
        ? model_.get_output_weight()
        : model_.get_token_embedding_weight();

    if (valid_idx) {
        weight = ggml_get_rows(ctx_, weight, valid_idx);
        ggml_set_name(weight, "output_weight_k");
    }
    cur = ggml_mul_mat(ctx_, weight, cur);
    // Gemma 2 final logit soft-capping (cap == 0 → off, byte-identical for all
    // non-Gemma-2 recipes). Applied before the "logits" name so get_output_logits
    // reads the capped values, matching the recipe's prefill head.
    if (final_softcap > 0.0f) {
        cur = build_softcap(ctx_, cur, final_softcap);
    }
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
}

ggml_tensor* ForwardPassBase::build_out_ids_slice(ggml_cgraph* gf, ggml_tensor* cur) {
    // Explicit dense-reference path (differential seam). Not a silent fallback:
    // the caller chose it via set_slice_prefill_head(false).
    if (!slice_prefill_head_) {
        return cur;
    }

    // "out_ids": int32 token-position selector. Width 1 — last token only
    // (OutputIdsInput fills it with n_rows-1 at set time). Gathering on the
    // [hidden, n_tokens] hidden state elides the discarded first n_tokens-1
    // head rows. Composes with build_output_head's vocab-axis weight slice:
    // different tensor, different axis, order-independent.
    ggml_tensor* out_ids = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, 1);
    ggml_set_input(out_ids);
    ggml_set_name(out_ids, "out_ids");
    ggml_build_forward_expand(gf, out_ids);
    graph_inputs_.add(std::make_unique<OutputIdsInput>());

    ggml_tensor* sliced = ggml_get_rows(ctx_, cur, out_ids);
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
        ctx_, GGML_TYPE_F32, hidden_dim, static_cast<int64_t>(n_img));
    ggml_set_input(img_in);
    set_tensor_name(gf, img_in, "image_embeddings");
    ggml_build_forward_expand(gf, img_in);

    // inpL[:, span : span+n_img] = img_in (one op; the surviving text columns
    // keep their sqrt(d_model) scale, the image columns enter unscaled).
    inpL = ggml_set_2d(ctx_, inpL, img_in, inpL->nb[1],
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
            ggml_tensor* m = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32,
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
    for (int il : attention_taps_) {
        std::string nm = "kq_soft." + std::to_string(il);
        ggml_tensor* ts = ggml_graph_get_tensor(gf, nm.c_str());
        if (!ts)
            throw std::runtime_error(
                "mark_attention_taps: attention-tap tensor '" + nm +
                "' expected in graph, actual absent — layer " +
                std::to_string(il) + " is not an attention layer of this "
                "recipe (or the graph has no such block).");
        ggml_set_output(ts);
        ggml_build_forward_expand(gf, ts);
    }
}

std::vector<ForwardPassBase::AttentionTap>
ForwardPassBase::get_attention_taps(ggml_cgraph* gf) {
    std::vector<AttentionTap> out;
    out.reserve(attention_taps_.size());
    for (int il : attention_taps_) {
        std::string nm = "kq_soft." + std::to_string(il);
        ggml_tensor* ts = ggml_graph_get_tensor(gf, nm.c_str());
        if (!ts)
            throw std::runtime_error(
                "get_attention_taps: attention-tap tensor '" + nm +
                "' expected in graph, actual absent — call "
                "mark_attention_taps(gf) after build_decoding_graph and before "
                "graph alloc.");
        AttentionTap tap;
        tap.layer  = il;
        tap.n_kv   = (int)ts->ne[0];
        tap.n_head = (int)ts->ne[2];   // shape [n_kv, 1, n_head, 1] at decode
        tap.rows.resize((size_t)tap.n_kv * tap.n_head);
        ggml_backend_tensor_get(ts, tap.rows.data(), 0, ggml_nbytes(ts));
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