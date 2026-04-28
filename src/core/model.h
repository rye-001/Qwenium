#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf_value.h"

class GGUFLoader;
class Tokenizer;

struct TensorMetadata {
    std::string name;
    ggml_type type;
    std::vector<uint64_t> shape;
    uint64_t offset;
};

// Tokenizer-specific structures
enum class TokenType : uint8_t {
    NORMAL = 1,
    UNKNOWN = 2,
    CONTROL = 3,
    USER_DEFINED = 4,
    UNUSED = 5,
    BYTE = 6
};

struct ModelMetadata {
    // Model metadata
    std::string model_name;
    std::string architecture;
    uint32_t block_count;
    uint32_t attention_head_count;
    uint32_t attention_head_count_kv;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t context_length;
    uint32_t vocab_size;
    uint32_t attention_key_length;
    uint32_t attention_value_length;
    float rope_freq_base = 0.0f;
    float rms_norm_eps   = 0.0f;

    // Tokenizer metadata (from GGUF)
    std::string tokenizer_type;              // "gpt2" tokenizer.ggml.model
    std::string tokenizer_pre;               // "qwen2" tokenizer.ggml.pre
    std::vector<std::string> id_to_token;    // tokenizer.ggml.tokens [151936 elements]
    std::vector<TokenType> token_types;      // tokenizer.ggml.token_type [151936 elements]
    std::vector<std::string> merges;         // tokenizer.ggml.merges [151387 elements]
    std::vector<float>      scores;          // tokenizer.ggml.scores (llama/sentencepiece)

    // Special token IDs (from GGUF)
    int32_t eos_token_id = -1;       // 151645
    int32_t bos_token_id = -1;       // 151643
    int32_t padding_token_id = -1;   // 151643
    int32_t unknown_token_id = -1;   // tokenizer.ggml.unknown_token_id
    bool add_bos_token = false;
    
    std::unordered_map<std::string, TensorMetadata> tensor_inventory;

    // Raw GGUF KVs; family-specific fields read from here (typed members are universal-only).
    GGUFKVBag raw_kv;
};

enum class Qwen3ModelSize {
    QWEN3_0_6B,
    QWEN3_8B,
    QWEN3_32B
};

// Structs to hold the model's tensors
struct TransformerBlock {
    // Attention (used by full attention layers in all architectures)
    struct ggml_tensor* attn_norm_weight = nullptr;
    struct ggml_tensor* attn_q_weight = nullptr;
    struct ggml_tensor* attn_k_weight = nullptr;
    struct ggml_tensor* attn_v_weight = nullptr;
    struct ggml_tensor* attn_output_weight = nullptr;

    // Qwen3/Qwen35 full-attention-specific
    struct ggml_tensor* attn_q_norm_weight = nullptr;
    struct ggml_tensor* attn_k_norm_weight = nullptr;

    // Qwen2-specific
    struct ggml_tensor* attn_q_bias = nullptr;
    struct ggml_tensor* attn_k_bias = nullptr;
    struct ggml_tensor* attn_v_bias = nullptr;

    // Feed-forward (shared by all layer types)
    struct ggml_tensor* ffn_norm_weight = nullptr;   // "ffn_norm" for qwen2/3, "post_attention_norm" for qwen35
    struct ggml_tensor* ffn_gate_weight = nullptr;
    struct ggml_tensor* ffn_up_weight = nullptr;
    struct ggml_tensor* ffn_down_weight = nullptr;

    // === SSM/DeltaNet tensors (qwen35 and qwen35moe DeltaNet layers) ===
    struct ggml_tensor* ssm_a = nullptr;             // learned log-decay
    struct ggml_tensor* ssm_conv1d_weight = nullptr; // causal conv
    struct ggml_tensor* ssm_dt_bias = nullptr;       // timestep bias
    struct ggml_tensor* ssm_alpha_weight = nullptr;  // gate projection (decay)
    struct ggml_tensor* ssm_beta_weight = nullptr;   // gate projection (update)
    struct ggml_tensor* attn_qkv_weight = nullptr;   // fused QKV
    struct ggml_tensor* attn_gate_weight = nullptr;  // output gate
    struct ggml_tensor* ssm_norm_weight = nullptr;   // RMS norm on SSM output
    struct ggml_tensor* ssm_out_weight = nullptr;    // output projection

    // === MoE FFN tensors (qwen35moe — all layers) ===
    // Tensor name → GGUF key
    struct ggml_tensor* moe_router_weight     = nullptr; // ffn_gate_inp.weight       [n_embd, n_experts]
    struct ggml_tensor* moe_shexp_gate        = nullptr; // ffn_gate_inp_shexp.weight [n_embd]
    struct ggml_tensor* moe_exp_gate_weight   = nullptr; // ffn_gate_exps.weight      [n_embd, d_ffn, n_experts]
    struct ggml_tensor* moe_exp_up_weight     = nullptr; // ffn_up_exps.weight        [n_embd, d_ffn, n_experts]
    struct ggml_tensor* moe_exp_down_weight   = nullptr; // ffn_down_exps.weight      [d_ffn, n_embd, n_experts]
    struct ggml_tensor* moe_shexp_gate_w      = nullptr; // ffn_gate_shexp.weight     [n_embd, d_ffn]
    struct ggml_tensor* moe_shexp_up_weight   = nullptr; // ffn_up_shexp.weight       [n_embd, d_ffn]
    struct ggml_tensor* moe_shexp_down_weight = nullptr; // ffn_down_shexp.weight     [d_ffn, n_embd]
};
class Model {
public:
    Model();
    ~Model();

    void load_metadata(const std::string& model_path);
    void load_tensors();

    bool validate_architecture() const;
    Qwen3ModelSize detect_model_size() const;
    std::string get_model_size_string() const;
    void print_metadata() const;
    
    const ModelMetadata& get_metadata() const { return metadata_; }
    struct ggml_tensor* get_token_embedding_weight() const { return token_embd_weight_; }
    struct ggml_tensor* get_output_norm_weight() const { return output_norm_weight_; }
    struct ggml_tensor* get_output_weight() const { return output_weight_; }
    struct ggml_context* get_context() const { return model_context_; }
    const TransformerBlock& get_block(int i) const { return blocks_[i]; }
    const ggml_backend_t& get_backend_metal() const { return backend_metal_;}
    const ggml_backend_t& get_backend_cpu() const { return backend_cpu_;}
    
    // Phase 1: Backend getters (for future use in Phase 2/3)
    ggml_backend_sched_t get_scheduler() const { return sched_; }
    bool has_metal_backend() const { return backend_metal_ != nullptr; }

private:
    ModelMetadata metadata_;
    bool is_loaded_;
    std::unique_ptr<GGUFLoader> loader_;
    ggml_context* model_context_;

    // Phase 1: Backend infrastructure
    ggml_backend_t backend_cpu_;
    ggml_backend_t backend_metal_;
    ggml_backend_sched_t sched_;
    ggml_backend_buffer_t weights_buffer_;

    // Tensors
    struct ggml_tensor* token_embd_weight_;
    struct ggml_tensor* output_norm_weight_;
    struct ggml_tensor* output_weight_;

    std::vector<TransformerBlock> blocks_;
    
    // Helper method for tensor assignment (reduces code duplication)
    void assign_tensor_pointers(const std::unordered_map<std::string, ggml_tensor*>& tensors);

    // Tokenizer
    std::unique_ptr<Tokenizer> tokenizer_;
public:
    Tokenizer* get_tokenizer() const { return tokenizer_.get(); }
};