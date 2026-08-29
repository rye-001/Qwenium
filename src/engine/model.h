#pragma once
// model.h — the loaded model: metadata, tokenizer vocabulary, and weights.
//
// Responsibility: hold everything a GGUF file turns into — ModelMetadata (arch
//   hyperparameters, the tokenizer's vocab/merges/scores, special-token ids, the
//   raw KV bag for family-specific keys, the weights hash) and the bound
//   ggml_tensor* weights, laid out per layer in TransformerBlock. Owns the
//   backend and the load path; does NOT build graphs — recipes in src/models/
//   do that, reading their weights from here.
// State owned: the backend buffer holding all weights, and the metadata.
// Invariants:
//   - The file mapping is RELEASED after the weights are copied into the backend
//     buffer (release_file_mapping). Not tidiness: the copy faults in every page,
//     so holding the mapping keeps a second full copy of the weights resident for
//     the process lifetime — measured 5.31 -> 1.75 GB steady-state RSS on
//     Qwen3.5-0.8B. Afterwards the loader's tensor-data accessors throw rather
//     than dereference a released mapping. Load-time PEAK is unaffected (source
//     and destination are both live during the copy): plan loading against 2x
//     model size, serving against 1x.
//   - TransformerBlock is a union of every family's slots; a recipe reads only
//     the ones its architecture defines, and nullptr means "this family has no
//     such tensor", not "missing".
// Fail-loud gap (architecture.md §12): assign_tensor_pointers' newer branches
//   use require(), which names the architecture and the tensor; the older ones
//   still use tensors.at() and throw an unnamed std::out_of_range.
// Unit tests: tests/unit/test_loader.cpp, tests/unit/test_gguf_kv_bag.cpp

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "ggml.h"
#include "ggml-backend.h"
#include "loader/gguf_value.h"

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

    // All token IDs that signal end-of-generation: primary EOS plus any
    // end-of-turn token (e.g. <|im_end|> for Qwen, <end_of_turn> for Gemma).
    // Populated by the loader; callers should iterate this instead of
    // hard-coding token strings.
    std::vector<int32_t> stop_token_ids;
    
    std::unordered_map<std::string, TensorMetadata> tensor_inventory;

    // Content identity of the model weights: a hash of the tensor inventory
    // (name + ggml_type + shape + offset, name-sorted) folded with the
    // architecture string. Captures arch + shape + quant + layout cheaply (no
    // weight-data read). Feeds CompatHeader::weights_hash so a session / prefix
    // blob built against different weights is refused fail-loud. NOTE: two
    // finetunes with a byte-identical inventory hash the same — sufficient for
    // the single-node local prefix library (same file reused); a sampled-byte
    // digest is the cross-node hardening if ever needed. 0 = not computed.
    uint64_t weights_hash = 0;

    // Raw GGUF KVs; family-specific fields read from here (typed members are universal-only).
    GGUFKVBag raw_kv;

    // How many trailing blocks are an MTP draft head rather than real decode
    // layers (docs/plan-mtp-decode.md §4).  Qwen3.8 ships one — block_count 65
    // is 64 layers plus the head; Qwen3.5 ships none, and 0 just means "decode
    // all block_count blocks".  Validators reject a head bigger than the model.
    //
    // Each family spells the key with its own arch prefix, so compose it
    // ("qwen35.nextn_predict_layers", "qwen35moe.nextn_predict_layers", ...).
    uint32_t nextn_predict_layers() const {
        return raw_kv.get_uint32_opt(architecture + ".nextn_predict_layers").value_or(0);
    }
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

    // === NextN / MTP head tensors (MTP GGUFs only — qwen35moe, qwen35) ===
    // Present on the last `nextn_predict_layers` blocks. A NextN block is an
    // ordinary full-attention block (the fields above — with MoE FFN on
    // qwen35moe, dense FFN on qwen35) *plus* these four.
    // The head predicts token t+1 from the main model's last hidden state; see
    // docs/plan-mtp-decode.md §4. enorm/hnorm are Gemma-style (1+w) RMSNorms.
    struct ggml_tensor* nextn_eh_proj           = nullptr; // nextn.eh_proj.weight          [2*n_embd, n_embd]
    struct ggml_tensor* nextn_enorm             = nullptr; // nextn.enorm.weight            [n_embd]
    struct ggml_tensor* nextn_hnorm             = nullptr; // nextn.hnorm.weight            [n_embd]
    struct ggml_tensor* nextn_shared_head_norm  = nullptr; // nextn.shared_head_norm.weight [n_embd]
};
class Model {
public:
    Model();
    ~Model();

    // `allow_multimodal=true` opts out of the load-time multimodal-only
    // refusal — the caller is supplying a vision projector (mmproj) and will
    // run the vision pipeline, so a checkpoint carrying image-placeholder
    // tokens is expected, not a misuse. Default false keeps the fail-loud
    // guard for the text-only path.
    void load_metadata(const std::string& model_path, bool allow_multimodal = false);
    void load_tensors();


    // Total trainable parameters, summed from the tensor inventory (every
    // tensor the GGUF ships, MoE experts and any NextN head included). This is
    // the real count for whatever file was loaded -- it does not guess from
    // block_count, and it needs no per-model table to keep current.
    uint64_t parameter_count() const;
    // parameter_count() rendered as a human label ("9.0B", "595M"). Returns
    // "unknown" only when the inventory is empty, which cannot happen after a
    // successful load.
    std::string get_parameter_count_string() const;
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