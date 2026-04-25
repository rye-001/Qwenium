#include "forward_pass_base.h"
#include "../layers/attention.h"
#include "../layers/ffn.h"
#include "../layers/norm.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <cmath>
#include <stdexcept>

ForwardPassBase::ForwardPassBase(const Qwen3Model& model, const Qwen3Metadata* metadata)
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

void ForwardPassBase::build_output_head(ggml_cgraph* gf, ggml_tensor* cur) {
    cur = build_norm(gf, cur, model_.get_output_norm_weight(), -1);
    set_tensor_name(gf, cur, "final_norm");

    if (model_.get_output_weight() != nullptr) {
        cur = ggml_mul_mat(ctx_, model_.get_output_weight(), cur);
    } else {
        cur = ggml_mul_mat(ctx_, model_.get_token_embedding_weight(), cur);
    }
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
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