#include "model.h"
#include "../loader/gguf_loader.h"
#include "../loader/tokenizer.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#ifndef QWEN_DEFAULT_GRAPH_SIZE
#define QWEN_DEFAULT_GRAPH_SIZE 16384
#endif

Model::Model()
    : is_loaded_(false),
      model_context_(nullptr),
      token_embd_weight_(nullptr),
      output_norm_weight_(nullptr),
      output_weight_(nullptr),
      backend_cpu_(nullptr),
      backend_metal_(nullptr),
      sched_(nullptr),
      weights_buffer_(nullptr)
{
    loader_ = std::make_unique<GGUFLoader>();

    // Phase 1: Initialize backend infrastructure
    std::cout << "[Phase 1] Initializing backends..." << std::endl;
    
    // Initialize CPU backend (always available)
    backend_cpu_ = ggml_backend_cpu_init();
    if (!backend_cpu_) {
        throw std::runtime_error("Failed to initialize CPU backend");
    }
    std::cout << "[Phase 1] CPU backend initialized" << std::endl;

#ifdef GGML_USE_METAL
    // Try to initialize Metal backend (optional)
    backend_metal_ = ggml_backend_metal_init();
    if (backend_metal_) {
        std::cout << "[Phase 1] Metal backend initialized - GPU acceleration available" << std::endl;
    } else {
        std::cout << "[Phase 1] Metal backend not available - using CPU only" << std::endl;
    }
#else
    std::cout << "[Phase 1] Metal support not compiled - using CPU only" << std::endl;
#endif

    // Create scheduler with available backends
    // NOTE: Scheduler will be used in Phase 3 for graph execution
    if (backend_metal_) {
        ggml_backend_t backends[] = {backend_metal_, backend_cpu_};
        sched_ = ggml_backend_sched_new(backends, nullptr, 2, QWEN_DEFAULT_GRAPH_SIZE, true, false);
        std::cout << "[Phase 1] Scheduler created with Metal + CPU backends" << std::endl;
    } else {
        ggml_backend_t backends[] = {backend_cpu_};
        sched_ = ggml_backend_sched_new(backends, nullptr, 1, QWEN_DEFAULT_GRAPH_SIZE, false, false);
        std::cout << "[Phase 1] Scheduler created with CPU backend only" << std::endl;
    }
    
    if (!sched_) {
        throw std::runtime_error("Failed to create backend scheduler");
    }
}

Model::~Model()
{
    // Free resources in reverse order of creation
    if (model_context_) {
        ggml_free(model_context_);
    }
    if (weights_buffer_) {
        ggml_backend_buffer_free(weights_buffer_);
    }
    if (sched_) {
        ggml_backend_sched_free(sched_);
    }
    if (backend_metal_) {
        ggml_backend_free(backend_metal_);
    }
    if (backend_cpu_) {
        ggml_backend_free(backend_cpu_);
    }
}

void Model::load_metadata(const std::string &model_path)
{
    loader_->load_model(model_path);
    loader_->extract_metadata(metadata_);
    is_loaded_ = true;
}

void Model::load_tensors()
{
    if (!is_loaded_) {
        throw GGUFLoadError("Metadata must be loaded before loading tensors.");
    }

    // Calculate memory needed for model weights
    size_t weights_size = loader_->calculate_tensors_memory_size();
    std::cout << "Calculated weights size: " << weights_size / (1024 * 1024) << " MB" << std::endl;

    // Choose loading path based on backend availability
    // Path A: CPU-only (backward compatible, original behavior)
    // Path B: Backend-based (new, for GPU offloading preparation)
    
    if (!backend_metal_) {
        // ===== PATH A: CPU-ONLY (Original behavior) =====
        std::cout << "[Load Path] Using CPU-only mode (original)" << std::endl;
        
        struct ggml_init_params params = {
            .mem_size   = weights_size,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        model_context_ = ggml_init(params);
        if (!model_context_) {
            throw GGUFLoadError("Failed to allocate model context.");
        }

        // Load tensors with data (original method)
        std::unordered_map<std::string, ggml_tensor *> tensors;
        loader_->load_all_tensors(model_context_, tensors);
        
        // Assign tensor pointers
        assign_tensor_pointers(tensors);
        
    } else {
        // ===== PATH B: BACKEND-BASED (New for Metal GPU) =====
        std::cout << "[Load Path] Using backend-based loading (Metal available)" << std::endl;
        
        // Step 1: Create context for tensor metadata only (no data allocation yet)
        size_t metadata_size = 128 * 1024 * 1024; // 128MB for tensor structs
        struct ggml_init_params params = {
            .mem_size   = metadata_size,
            .mem_buffer = NULL,
            .no_alloc   = true,  // Don't allocate tensor data in context
        };
        model_context_ = ggml_init(params);
        if (!model_context_) {
            throw GGUFLoadError("Failed to create ggml context.");
        }

        // Step 2: Load tensor metadata (creates ggml_tensor structs without data)
        std::unordered_map<std::string, ggml_tensor *> tensors;
        loader_->load_tensor_metadata(model_context_, tensors);
        std::cout << "Loaded metadata for " << tensors.size() << " tensors" << std::endl;

        // Step 3: Allocate backend buffer and associate all tensors with it
        // CRITICAL: ggml_backend_alloc_ctx_tensors allocates buffer AND sets tensor->buffer
        // On Apple Silicon with Metal, this uses unified memory that both CPU and GPU can access
        std::cout << "Allocating Metal backend buffer and associating tensors..." << std::endl;
        weights_buffer_ = ggml_backend_alloc_ctx_tensors(model_context_, backend_metal_);
        if (!weights_buffer_) {
            throw GGUFLoadError("Failed to allocate Metal backend buffer for tensors");
        }
        
        // Verify buffer allocation
        size_t buffer_size = ggml_backend_buffer_get_size(weights_buffer_);
        std::cout << "Metal buffer allocated: " << buffer_size / (1024 * 1024) << " MB" << std::endl;

        // Step 4: Copy tensor data from GGUF file into backend buffer
        // Now tensor->buffer is properly set, so ggml_backend_tensor_set() will work
        std::cout << "Copying tensor data to Metal backend buffer..." << std::endl;
        size_t tensors_copied = 0;
        size_t total_bytes_copied = 0;
        
        for (const auto& [name, tensor] : tensors) {
            // Verify tensor buffer was set by ggml_backend_alloc_ctx_tensors
            if (!tensor->buffer) {
                throw GGUFLoadError("Tensor buffer not set after allocation for: " + name);
            }
            bool is_vocab_tensor = (name == "token_embd.weight" || name == "output.weight");

            // For all other tensors, do a direct wholesale copy.
            const void* src_data = loader_->get_tensor_data(name);
            if (!src_data) {
                throw GGUFLoadError("Failed to get data for tensor: " + name);
            }
            size_t data_size = ggml_nbytes(tensor);
        
            // Copy data to backend buffer
            // On Metal with unified memory, this copies to memory accessible by both CPU and GPU
            ggml_backend_tensor_set(tensor, src_data, 0, data_size);
            tensors_copied++;
            total_bytes_copied += data_size;
            
            // Progress indicator every 50 tensors
            if (tensors_copied % 50 == 0) {
                std::cout << "  Progress: " << tensors_copied << "/" << tensors.size() 
                            << " tensors (" << (total_bytes_copied / (1024 * 1024)) << " MB)\r" 
                            << std::flush;
            }
        }
        std::cout << std::endl;
        std::cout << "Successfully copied " << tensors_copied << " tensors ("
                  << (total_bytes_copied / (1024 * 1024)) << " MB) to Metal backend buffer" << std::endl;
        
        // Assign tensor pointers
        assign_tensor_pointers(tensors);
    }

    std::cout << "All tensors loaded and assigned successfully." << std::endl;

    // Derive tokenizer config from standard GGUF metadata fields (tokenizer_type,
    // add_bos_token) so the tokenizer has no architecture string literals.
    std::cout << "Initializing tokenizer..." << std::endl;
    TokenizerConfig tok_cfg = tokenizer_config_from_gguf(metadata_);
    tokenizer_ = std::make_unique<Tokenizer>(&metadata_, tok_cfg);
    std::cout << "Tokenizer initialized." << std::endl;
}

void Model::assign_tensor_pointers(const std::unordered_map<std::string, ggml_tensor*>& tensors)
{
    try {
        token_embd_weight_ = tensors.at("token_embd.weight");
        output_norm_weight_ = tensors.at("output_norm.weight");

        auto it = tensors.find("output.weight");
        if (it != tensors.end()) {
            output_weight_ = it->second;
        }

        blocks_.resize(metadata_.block_count);
        for (uint32_t i = 0; i < metadata_.block_count; ++i) {
            std::string prefix = "blk." + std::to_string(i) + ".";

            // Shared: attention norm (pre-attention RMS norm)
            blocks_[i].attn_norm_weight = tensors.at(prefix + "attn_norm.weight");

            if (metadata_.architecture == "qwen35moe") {
                // Pre-FFN norm (called post_attention_norm in this family)
                blocks_[i].ffn_norm_weight = tensors.at(prefix + "post_attention_norm.weight");

                // MoE FFN tensors present in every layer
                auto require = [&](const std::string& key) -> ggml_tensor* {
                    auto it = tensors.find(key);
                    if (it == tensors.end()) {
                        throw GGUFLoadError(
                            "qwen35moe: missing tensor '" + key +
                            "': expected in block weights, got absent");
                    }
                    return it->second;
                };
                blocks_[i].moe_router_weight     = require(prefix + "ffn_gate_inp.weight");
                blocks_[i].moe_shexp_gate        = require(prefix + "ffn_gate_inp_shexp.weight");
                blocks_[i].moe_exp_gate_weight   = require(prefix + "ffn_gate_exps.weight");
                blocks_[i].moe_exp_up_weight     = require(prefix + "ffn_up_exps.weight");
                blocks_[i].moe_exp_down_weight   = require(prefix + "ffn_down_exps.weight");
                blocks_[i].moe_shexp_gate_w      = require(prefix + "ffn_gate_shexp.weight");
                blocks_[i].moe_shexp_up_weight   = require(prefix + "ffn_up_shexp.weight");
                blocks_[i].moe_shexp_down_weight = require(prefix + "ffn_down_shexp.weight");

                // Inline arithmetic: tensor binding runs at load time, before any recipe exists.
                // Typed configs (Qwen35MoEConfig) are constructed by recipes after this runs —
                // chicken/egg. Use the same one-line formula the recipe and validators use.
                const uint32_t fai = metadata_.raw_kv.get_uint32("qwen35moe.full_attention_interval");
                const bool is_full = (fai > 0) && ((i % fai) == (fai - 1));
                if (is_full) {
                    blocks_[i].attn_q_weight      = require(prefix + "attn_q.weight");
                    blocks_[i].attn_k_weight      = require(prefix + "attn_k.weight");
                    blocks_[i].attn_v_weight      = require(prefix + "attn_v.weight");
                    blocks_[i].attn_output_weight = require(prefix + "attn_output.weight");
                    blocks_[i].attn_q_norm_weight = require(prefix + "attn_q_norm.weight");
                    blocks_[i].attn_k_norm_weight = require(prefix + "attn_k_norm.weight");
                } else {
                    blocks_[i].ssm_a             = require(prefix + "ssm_a");
                    blocks_[i].ssm_conv1d_weight  = require(prefix + "ssm_conv1d.weight");
                    blocks_[i].ssm_dt_bias        = require(prefix + "ssm_dt.bias");
                    blocks_[i].ssm_alpha_weight   = require(prefix + "ssm_alpha.weight");
                    blocks_[i].ssm_beta_weight    = require(prefix + "ssm_beta.weight");
                    blocks_[i].attn_qkv_weight    = require(prefix + "attn_qkv.weight");
                    blocks_[i].attn_gate_weight   = require(prefix + "attn_gate.weight");
                    blocks_[i].ssm_norm_weight    = require(prefix + "ssm_norm.weight");
                    blocks_[i].ssm_out_weight     = require(prefix + "ssm_out.weight");
                }
                continue;
            }

            // FFN (shared by qwen2/qwen3/qwen35)
            blocks_[i].ffn_gate_weight = tensors.at(prefix + "ffn_gate.weight");
            blocks_[i].ffn_up_weight   = tensors.at(prefix + "ffn_up.weight");
            blocks_[i].ffn_down_weight = tensors.at(prefix + "ffn_down.weight");

            if (metadata_.architecture == "qwen35") {
                // qwen35: FFN norm is called "post_attention_norm"
                blocks_[i].ffn_norm_weight = tensors.at(prefix + "post_attention_norm.weight");

                const uint32_t fai = metadata_.raw_kv.get_uint32("qwen35.full_attention_interval");
                const bool is_full = (fai > 0) && ((i % fai) == (fai - 1));
                if (is_full) {
                    // Full softmax attention layer (layers 3,7,11,15,19,23)
                    blocks_[i].attn_q_weight      = tensors.at(prefix + "attn_q.weight");
                    blocks_[i].attn_k_weight      = tensors.at(prefix + "attn_k.weight");
                    blocks_[i].attn_v_weight      = tensors.at(prefix + "attn_v.weight");
                    blocks_[i].attn_output_weight = tensors.at(prefix + "attn_output.weight");
                    blocks_[i].attn_q_norm_weight = tensors.at(prefix + "attn_q_norm.weight");
                    blocks_[i].attn_k_norm_weight = tensors.at(prefix + "attn_k_norm.weight");
                } else {
                    // SSM (GatedDeltaNet) layer
                    blocks_[i].ssm_a              = tensors.at(prefix + "ssm_a");
                    blocks_[i].ssm_conv1d_weight  = tensors.at(prefix + "ssm_conv1d.weight");
                    blocks_[i].ssm_dt_bias        = tensors.at(prefix + "ssm_dt.bias");
                    blocks_[i].ssm_alpha_weight   = tensors.at(prefix + "ssm_alpha.weight");
                    blocks_[i].ssm_beta_weight    = tensors.at(prefix + "ssm_beta.weight");
                    blocks_[i].attn_qkv_weight    = tensors.at(prefix + "attn_qkv.weight");
                    blocks_[i].attn_gate_weight   = tensors.at(prefix + "attn_gate.weight");
                    blocks_[i].ssm_norm_weight    = tensors.at(prefix + "ssm_norm.weight");
                    blocks_[i].ssm_out_weight     = tensors.at(prefix + "ssm_out.weight");
                }

            } else if (metadata_.architecture == "qwen3") {
                blocks_[i].ffn_norm_weight    = tensors.at(prefix + "ffn_norm.weight");
                blocks_[i].attn_q_weight      = tensors.at(prefix + "attn_q.weight");
                blocks_[i].attn_k_weight      = tensors.at(prefix + "attn_k.weight");
                blocks_[i].attn_v_weight      = tensors.at(prefix + "attn_v.weight");
                blocks_[i].attn_output_weight = tensors.at(prefix + "attn_output.weight");
                blocks_[i].attn_q_norm_weight = tensors.at(prefix + "attn_q_norm.weight");
                blocks_[i].attn_k_norm_weight = tensors.at(prefix + "attn_k_norm.weight");

            } else if (metadata_.architecture == "qwen2") {
                blocks_[i].ffn_norm_weight    = tensors.at(prefix + "ffn_norm.weight");
                blocks_[i].attn_q_weight      = tensors.at(prefix + "attn_q.weight");
                blocks_[i].attn_k_weight      = tensors.at(prefix + "attn_k.weight");
                blocks_[i].attn_v_weight      = tensors.at(prefix + "attn_v.weight");
                blocks_[i].attn_output_weight = tensors.at(prefix + "attn_output.weight");
                blocks_[i].attn_q_bias        = tensors.at(prefix + "attn_q.bias");
                blocks_[i].attn_k_bias        = tensors.at(prefix + "attn_k.bias");
                blocks_[i].attn_v_bias        = tensors.at(prefix + "attn_v.bias");
            } else if (metadata_.architecture == "gemma") {
                // Gemma 1: GQA, no QK-norm, no biases, GeGLU FFN.
                // Norm form is (1+w) — applied at graph time, not at load time.
                blocks_[i].ffn_norm_weight    = tensors.at(prefix + "ffn_norm.weight");
                blocks_[i].attn_q_weight      = tensors.at(prefix + "attn_q.weight");
                blocks_[i].attn_k_weight      = tensors.at(prefix + "attn_k.weight");
                blocks_[i].attn_v_weight      = tensors.at(prefix + "attn_v.weight");
                blocks_[i].attn_output_weight = tensors.at(prefix + "attn_output.weight");
            } else if (metadata_.architecture == "gemma2") {
                // Gemma 2: same QKV / FFN tensor names as Gemma 1.
                // post_attention_norm / post_ffw_norm are fetched by the recipe
                // directly via ggml_get_tensor — they are G2-specific and not in
                // the generic TransformerBlock struct.
                blocks_[i].ffn_norm_weight    = tensors.at(prefix + "ffn_norm.weight");
                blocks_[i].attn_q_weight      = tensors.at(prefix + "attn_q.weight");
                blocks_[i].attn_k_weight      = tensors.at(prefix + "attn_k.weight");
                blocks_[i].attn_v_weight      = tensors.at(prefix + "attn_v.weight");
                blocks_[i].attn_output_weight = tensors.at(prefix + "attn_output.weight");
            }
        }
    }
    catch (const std::out_of_range &e) {
        throw GGUFLoadError("Error assigning tensor: " + std::string(e.what()));
    }
}

bool Model::validate_architecture() const
{
    if (!is_loaded_) {
        return false;
    }
    return metadata_.architecture == "qwen3" || metadata_.architecture == "qwen2" ||
           metadata_.architecture == "qwen35" || metadata_.architecture == "qwen35moe" ||
           metadata_.architecture == "gemma"  || metadata_.architecture == "gemma2";
}

Qwen3ModelSize Model::detect_model_size() const
{
    if (!is_loaded_) {
        return Qwen3ModelSize::QWEN3_0_6B;
    }

    if (metadata_.block_count == 28) {
        return Qwen3ModelSize::QWEN3_0_6B;
    } else if (metadata_.block_count == 36) {
        return Qwen3ModelSize::QWEN3_8B;
    } else if (metadata_.block_count == 64) {
        return Qwen3ModelSize::QWEN3_32B;
    }

    return Qwen3ModelSize::QWEN3_0_6B;
}

std::string Model::get_model_size_string() const
{
    switch (detect_model_size()) {
        case Qwen3ModelSize::QWEN3_0_6B: return "0.6B";
        case Qwen3ModelSize::QWEN3_8B: return "8B";
        case Qwen3ModelSize::QWEN3_32B: return "32B";
        default: return "Unknown";
    }
}

void Model::print_metadata() const
{
    if (!is_loaded_) {
        std::cout << "Model not loaded." << std::endl;
        return;
    }

    std::cout << "Model Metadata:" << std::endl;
    std::cout << "  Architecture: " << metadata_.architecture << std::endl;
    std::cout << "  Model Name: " << metadata_.model_name << std::endl;

    // general.basename / general.finetune identify base vs instruct variants.
    if (auto v = metadata_.raw_kv.get_string_opt("general.basename"); v)
        std::cout << "  Basename: " << *v << std::endl;
    if (auto v = metadata_.raw_kv.get_string_opt("general.finetune"); v)
        std::cout << "  Finetune: " << *v << std::endl;

    std::cout << "  Block Count: " << metadata_.block_count << std::endl;
    std::cout << "  Embedding Length: " << metadata_.embedding_length << std::endl;
    std::cout << "  Attention Heads: " << metadata_.attention_head_count << std::endl;
    std::cout << "  KV Heads: " << metadata_.attention_head_count_kv << std::endl;
    std::cout << "  Attn Key Length (n_embd_head): " << metadata_.attention_key_length << std::endl;
    std::cout << "  Rope Freq Base: " << metadata_.rope_freq_base << std::endl;
    std::cout << "  RMS Norm Epsilon: " << metadata_.rms_norm_eps << std::endl;
    std::cout << "  BOS token id: " << metadata_.bos_token_id
              << "  add_bos_token: " << (metadata_.add_bos_token ? "true" : "false") << std::endl;
    std::cout << "  EOS token id: " << metadata_.eos_token_id << std::endl;
    std::cout << "  Detected Model Size: " << get_model_size_string() << std::endl;
}