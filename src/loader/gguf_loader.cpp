#include "gguf_loader.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <cstring> // For memcpy
#include <string_view> 
#include <functional>  
#include <unordered_map>

// GGUF constants
static const uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
static const uint32_t GGUF_VERSION = 3;

// Metadata key constants
static constexpr std::string_view KEY_GENERAL_ARCHITECTURE = "general.architecture";
static constexpr std::string_view KEY_GENERAL_NAME = "general.name";

// Tokenizer metadata keys
static constexpr std::string_view KEY_TOKENIZER_MODEL = "tokenizer.ggml.model";
static constexpr std::string_view KEY_TOKENIZER_PRE = "tokenizer.ggml.pre";
static constexpr std::string_view KEY_TOKENIZER_TOKENS = "tokenizer.ggml.tokens";
static constexpr std::string_view KEY_TOKENIZER_TOKEN_TYPE = "tokenizer.ggml.token_type";
static constexpr std::string_view KEY_TOKENIZER_MERGES = "tokenizer.ggml.merges";
static constexpr std::string_view KEY_TOKENIZER_EOS_TOKEN_ID = "tokenizer.ggml.eos_token_id";
static constexpr std::string_view KEY_TOKENIZER_BOS_TOKEN_ID = "tokenizer.ggml.bos_token_id";
static constexpr std::string_view KEY_TOKENIZER_PADDING_TOKEN_ID = "tokenizer.ggml.padding_token_id";
static constexpr std::string_view KEY_TOKENIZER_ADD_BOS_TOKEN = "tokenizer.ggml.add_bos_token";

struct GGUFHeader
{
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

QwenGGUFLoader::QwenGGUFLoader() : is_loaded_(false), tensor_data_offset_(0) {}

QwenGGUFLoader::~QwenGGUFLoader() { cleanup_resources(); }

void QwenGGUFLoader::load_model(const std::string &path)
{
    model_path_ = path;
    std::cout << "Loading GGUF model: " << path << std::endl;

    file_mapper_ = std::make_unique<FileMapper>(path);
    if (!file_mapper_->is_open()) {
        throw GGUFLoadError("Failed to map file: " + model_path_);
    }

    size_t offset = 0;

    // 1. Read and validate header
    if (offset + sizeof(GGUFHeader) > file_mapper_->size()) {
        throw GGUFLoadError("File is too small for GGUF header");
    }
    const GGUFHeader* header = reinterpret_cast<const GGUFHeader*>(file_mapper_->data() + offset);
    offset += sizeof(GGUFHeader);

    if (header->magic != GGUF_MAGIC) {
        throw GGUFLoadError("Invalid GGUF magic number");
    }

    if (header->version > GGUF_VERSION) {
        throw GGUFLoadError("Unsupported GGUF version: " + std::to_string(header->version));
    }

    std::cout << "Valid GGUF header found. Tensor count: " << header->tensor_count
              << ", Metadata KV count: " << header->metadata_kv_count << std::endl;

    // 2. Parse metadata
    parse_and_validate_metadata(offset);

    // 3. Parse tensor inventory
    parse_tensor_inventory(offset, header->tensor_count);

    // 4. Calculate tensor data offset
    const long alignment = 32;
    tensor_data_offset_ = (offset + alignment - 1) / alignment * alignment;

    // 5. Validate tensor inventory
    validate_tensor_inventory();

    is_loaded_ = true;
}

void QwenGGUFLoader::extract_metadata(Qwen3Metadata &metadata) const
{
    if (!is_loaded_)
    {
        throw GGUFLoadError("Model is not loaded, cannot extract metadata.");
    }
    metadata = metadata_;
}

size_t QwenGGUFLoader::calculate_tensors_memory_size() const
{
    size_t total_bytes = 0;

    // This calculation must match how ggml allocates memory for a context:
    // For each tensor, it allocates a ggml_tensor struct and its data.
    // Both allocations are padded to GGML_MEM_ALIGN boundaries.
    const size_t ggml_tensor_struct_size = sizeof(ggml_tensor);

    for (const auto &[name, meta] : metadata_.tensor_inventory)
    {
        // 1. Align for ggml_tensor struct
        total_bytes = GGML_PAD(total_bytes, GGML_MEM_ALIGN);
        total_bytes += ggml_tensor_struct_size;

        // 2. Align for tensor data
        const size_t tensor_data_size = calculate_tensor_bytes(meta);
        total_bytes = GGML_PAD(total_bytes, GGML_MEM_ALIGN);
        total_bytes += tensor_data_size;
    }

    // Add a large scratch buffer for computation graph, KV cache, and intermediate tensors.
    // 512 MB is a generous buffer for a small model and typical sequence lengths.
    // For a production system, this should be calculated more precisely based on
    // batch size, sequence length, and KV cache size.
    const size_t scratch_buffer_size = 512 * 1024 * 1024;
    total_bytes += scratch_buffer_size;

    return total_bytes;
}

void QwenGGUFLoader::load_all_tensors(ggml_context *ctx, std::unordered_map<std::string, ggml_tensor *> &tensors)
{
    if (!file_mapper_) {
        throw GGUFLoadError("Model not loaded or mapped.");
    }

    for (const auto &[name, meta] : metadata_.tensor_inventory)
    {
        // Standard wholesale copy for non-vocabulary tensors
        std::vector<int64_t> shape(meta.shape.begin(), meta.shape.end());
        ggml_tensor *tensor = ggml_new_tensor(ctx, meta.type, shape.size(), shape.data());
        if (!tensor) {
            throw GGUFLoadError("Failed to create tensor: " + name);
        }
        ggml_set_name(tensor, name.c_str());

        const size_t tensor_size = ggml_nbytes(tensor);
        const size_t tensor_offset = tensor_data_offset_ + meta.offset;

        if (tensor_offset + tensor_size > file_mapper_->size()) {
            throw GGUFLoadError("Tensor data for \"" + name + "\" would read past end of file.");
        }

        memcpy(tensor->data, file_mapper_->data() + tensor_offset, tensor_size);
        tensors[name] = tensor;
    }
    std::cout << "Successfully loaded " << tensors.size() << " tensors." << std::endl;
}

const TensorMetadata& QwenGGUFLoader::get_tensor_metadata(const std::string& name) const
{
    auto it = metadata_.tensor_inventory.find(name);
    if (it == metadata_.tensor_inventory.end()) {
        throw GGUFLoadError("Tensor metadata not found for: " + name);
    }
    return it->second;
}




void QwenGGUFLoader::validate_tensor_shape(struct ggml_tensor *tensor,
                                           const std::vector<int64_t> &expected)
{
    if (!tensor) {
        throw GGUFLoadError("Tensor is null during shape validation.");
    }

    int actual_dims = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] > 0) actual_dims++;
    }

    if (actual_dims < (int)expected.size()) {
        throw GGUFLoadError("Tensor shape validation failed: tensor has fewer dimensions than expected. "
                            "Expected at least " + std::to_string(expected.size()) + ", got " + std::to_string(actual_dims));
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (tensor->ne[i] != expected[i]) {
            throw GGUFLoadError("Tensor shape validation failed for dimension " + std::to_string(i) + ". "
                                "Expected " + std::to_string(expected[i]) + ", got " + std::to_string(tensor->ne[i]));
        }
    }

    for (int i = expected.size(); i < actual_dims; ++i) {
        if (tensor->ne[i] != 1) {
            throw GGUFLoadError("Tensor shape validation failed: trailing dimension " + std::to_string(i) + " is not 1. "
                                "Expected 1, got " + std::to_string(tensor->ne[i]));
        }
    }
}

size_t QwenGGUFLoader::calculate_tensor_bytes(const TensorMetadata &meta) const
{
    size_t elements = 1;
    for (uint64_t dim : meta.shape)
    {
        elements *= dim;
    }
    return ggml_type_size(meta.type) * elements / ggml_blck_size(meta.type);
}

void QwenGGUFLoader::validate_architecture(const Qwen3Metadata &meta) const
{
    if (meta.architecture != "qwen3" && meta.architecture != "qwen2" &&
        meta.architecture != "qwen35" && meta.architecture != "qwen35moe")
    {
        throw GGUFLoadError("Invalid model architecture: '" + meta.architecture +
                            "', expected one of: 'qwen3', 'qwen2', 'qwen35', 'qwen35moe'");
    }
    std::cout << "Validated model architecture: " << meta.architecture << std::endl;
}

void QwenGGUFLoader::unload_model()
{
    cleanup_resources();
    is_loaded_ = false;
}

void QwenGGUFLoader::parse_and_validate_metadata(size_t& offset)
{
    const GGUFHeader* header = reinterpret_cast<const GGUFHeader*>(file_mapper_->data());

    // First pass: get architecture
    size_t initial_offset = offset;
    for (uint64_t i = 0; i < header->metadata_kv_count; ++i) {
        const std::string key = read_string_from_mem(initial_offset);
        const GGUFValueType type = read_value_from_mem<GGUFValueType>(initial_offset);
        if (key == KEY_GENERAL_ARCHITECTURE) {
            if (type != GGUFValueType::STRING) throw GGUFLoadError("general.architecture has unexpected type.");
            size_t arch_offset = initial_offset;
            metadata_.architecture = read_string_from_mem(arch_offset);
            break;
        }
        skip_gguf_value_from_mem(initial_offset, type);
    }

    if (metadata_.architecture.empty()) {
        throw GGUFLoadError("general.architecture not found in metadata");
    }
    validate_architecture(metadata_);
    const std::string prefix = metadata_.architecture + ".";

    // Second pass: parse all metadata using the correct prefix
    for (uint64_t i = 0; i < header->metadata_kv_count; ++i)
    {
        const std::string key = read_string_from_mem(offset);
        const GGUFValueType type = read_value_from_mem<GGUFValueType>(offset);

        if (key == KEY_GENERAL_NAME) {
            if (type != GGUFValueType::STRING) throw GGUFLoadError("general.name has unexpected type.");
            metadata_.model_name = read_string_from_mem(offset);
        } else if (key == prefix + "block_count") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.block_count = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "embedding_length") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.embedding_length = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "feed_forward_length") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.feed_forward_length = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "context_length") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.context_length = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "attention.head_count") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.attention_head_count = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "attention.head_count_kv") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.attention_head_count_kv = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "attention.key_length") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.attention_key_length = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "attention.value_length") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.attention_value_length = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "rope.freq_base") {
            if (type != GGUFValueType::FLOAT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.rope_freq_base = read_value_from_mem<float>(offset);
        } else if (key == prefix + "attention.layer_norm_rms_epsilon") {
            if (type != GGUFValueType::FLOAT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.rms_norm_eps = read_value_from_mem<float>(offset);
        } else if (key == KEY_TOKENIZER_MODEL) {
             if (type != GGUFValueType::STRING) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_MODEL) + " has unexpected type.");
            metadata_.tokenizer_type = read_string_from_mem(offset);
        } else if (key == KEY_TOKENIZER_PRE) {
            if (type != GGUFValueType::STRING) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_PRE) + " has unexpected type.");
            metadata_.tokenizer_pre = read_string_from_mem(offset);
        } else if (key == KEY_TOKENIZER_TOKENS) {
            if (type != GGUFValueType::ARRAY) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_TOKENS) + " has unexpected type.");
            const GGUFValueType array_type = read_value_from_mem<GGUFValueType>(offset);
            if (array_type != GGUFValueType::STRING) throw GGUFLoadError("Tokenizer tokens array has unexpected element type.");

            const uint64_t num_tokens = read_value_from_mem<uint64_t>(offset);
            metadata_.vocab_size = num_tokens;
            metadata_.id_to_token.resize(num_tokens);
            for (uint64_t i = 0; i < num_tokens; ++i) {
                metadata_.id_to_token[i] = read_string_from_mem(offset);
                is_valid_utf8(metadata_.id_to_token[i]);
            }
        } else if (key == KEY_TOKENIZER_TOKEN_TYPE) {
            if (type != GGUFValueType::ARRAY) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_TOKEN_TYPE) + " has unexpected type.");
            const GGUFValueType array_type = read_value_from_mem<GGUFValueType>(offset);
            if (array_type != GGUFValueType::INT32) throw GGUFLoadError("Tokenizer token_type array has unexpected element type.");

            const uint64_t num_tokens = read_value_from_mem<uint64_t>(offset);
            metadata_.token_types.resize(num_tokens);
            for (uint64_t i = 0; i < num_tokens; ++i) {
                metadata_.token_types[i] = static_cast<TokenType>(read_value_from_mem<int32_t>(offset));
            }
        } else if (key == KEY_TOKENIZER_MERGES) {
            if (type != GGUFValueType::ARRAY) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_MERGES) + " has unexpected type.");
            const GGUFValueType array_type = read_value_from_mem<GGUFValueType>(offset);
            if (array_type != GGUFValueType::STRING) throw GGUFLoadError("Tokenizer merges array has unexpected element type.");

            const uint64_t num_merges = read_value_from_mem<uint64_t>(offset);
            metadata_.merges.resize(num_merges);
            for (uint64_t i = 0; i < num_merges; ++i) {
                metadata_.merges[i] = read_string_from_mem(offset);
            }
        } else if (key == KEY_TOKENIZER_EOS_TOKEN_ID) {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_EOS_TOKEN_ID) + " has unexpected type.");
            metadata_.eos_token_id = read_value_from_mem<uint32_t>(offset);
        } else if (key == KEY_TOKENIZER_BOS_TOKEN_ID) {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_BOS_TOKEN_ID) + " has unexpected type.");
            metadata_.bos_token_id = read_value_from_mem<uint32_t>(offset);
        } else if (key == KEY_TOKENIZER_PADDING_TOKEN_ID) {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_PADDING_TOKEN_ID) + " has unexpected type.");
            metadata_.padding_token_id = read_value_from_mem<uint32_t>(offset);
        } else if (key == KEY_TOKENIZER_ADD_BOS_TOKEN) {
            if (type != GGUFValueType::BOOL) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_ADD_BOS_TOKEN) + " has unexpected type.");
            metadata_.add_bos_token = read_value_from_mem<bool>(offset);
        } else if (key == prefix + "ssm.conv_kernel") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.ssm_conv_kernel = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "ssm.state_size") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.ssm_state_size = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "ssm.group_count") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.ssm_group_count = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "ssm.time_step_rank") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.ssm_time_step_rank = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "ssm.inner_size") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.ssm_inner_size = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "full_attention_interval") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.full_attention_interval = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "rope.dimension_count") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.rope_dimension_count = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "expert_count") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.expert_count = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "expert_used_count") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.expert_used_count = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "expert_feed_forward_length") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.expert_feed_forward_length = read_value_from_mem<uint32_t>(offset);
        } else if (key == prefix + "expert_shared_feed_forward_length") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.expert_shared_feed_forward_length = read_value_from_mem<uint32_t>(offset);
        } else {
            skip_gguf_value_from_mem(offset, type);
        }
    }
}

void QwenGGUFLoader::cleanup_resources()
{
    file_mapper_.reset();
}

void QwenGGUFLoader::is_valid_utf8(const std::string &str) const
{
    int i = 0;
    while (i < (int)str.length()) {
        int len = 1;
        unsigned char c = str[i];
        if ((c & 0x80) == 0x00) { len = 1; } 
        else if ((c & 0xE0) == 0xC0) { len = 2; } 
        else if ((c & 0xF0) == 0xE0) { len = 3; } 
        else if ((c & 0xF8) == 0xF0) { len = 4; } 
        else { throw GGUFLoadError("Invalid UTF-8 sequence encountered."); } 

        if (i + len > (int)str.length()) throw GGUFLoadError("Invalid UTF-8 sequence: incomplete character at end of string.");

        for (int j = 1; j < len; ++j) {
            if ((str[i + j] & 0xC0) != 0x80) throw GGUFLoadError("Invalid UTF-8 sequence: continuation byte expected.");
        }
        i += len;
    }
}

void QwenGGUFLoader::parse_tensor_inventory(size_t& offset, uint64_t tensor_count)
{
    std::cout << "Parsing tensor inventory..." << std::endl;
    for (uint64_t i = 0; i < tensor_count; ++i)
    {
        TensorMetadata tensor_meta;
        tensor_meta.name = read_string_from_mem(offset);

        const uint32_t n_dims = read_value_from_mem<uint32_t>(offset);
        tensor_meta.shape.resize(n_dims);
        if (offset + n_dims * sizeof(uint64_t) > file_mapper_->size()) {
            throw GGUFLoadError("Attempt to read past end of file while parsing tensor shape for " + tensor_meta.name);
        }
        memcpy(tensor_meta.shape.data(), file_mapper_->data() + offset, n_dims * sizeof(uint64_t));
        offset += n_dims * sizeof(uint64_t);

        tensor_meta.type = read_value_from_mem<ggml_type>(offset);
        tensor_meta.offset = read_value_from_mem<uint64_t>(offset);

        metadata_.tensor_inventory[tensor_meta.name] = tensor_meta;
    }
    std::cout << "Successfully parsed " << metadata_.tensor_inventory.size()
              << " tensor metadata entries." << std::endl;
}

void validate_qwen35moe_inventory(const Qwen3Metadata& meta)
{
    const auto& inventory = meta.tensor_inventory;

    auto require = [&](const std::string& name) {
        if (inventory.find(name) == inventory.end()) {
            throw GGUFLoadError(
                "qwen35moe: missing tensor '" + name +
                "': expected in model weights, got absent");
        }
    };

    require("token_embd.weight");
    require("output_norm.weight");

    // MoE tensor names present in every block regardless of layer type
    static const std::vector<std::string> moe_tensors = {
        "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
        "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"
    };
    static const std::vector<std::string> attn_tensors = {
        "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"
    };
    static const std::vector<std::string> dn_tensors = {
        "ssm_a", "ssm_conv1d.weight", "ssm_dt.bias",
        "ssm_alpha.weight", "ssm_beta.weight",
        "attn_qkv.weight", "attn_gate.weight",
        "ssm_norm.weight", "ssm_out.weight"
    };

    for (uint32_t i = 0; i < meta.block_count; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        require(p + "attn_norm.weight");
        require(p + "post_attention_norm.weight");
        for (const auto& t : moe_tensors) require(p + t);

        if (meta.is_full_attention_layer(i)) {
            for (const auto& t : attn_tensors) require(p + t);
        } else {
            for (const auto& t : dn_tensors) require(p + t);
        }
    }
}

void QwenGGUFLoader::validate_tensor_inventory() const
{
    std::cout << "Validating tensor inventory..." << std::endl;
    const auto &inventory = metadata_.tensor_inventory;

    if (metadata_.architecture == "qwen35moe") {
        validate_qwen35moe_inventory(metadata_);
        std::cout << "Tensor inventory validation passed." << std::endl;
        return;
    }

    // Global tensors
    if (inventory.find("token_embd.weight") == inventory.end()) {
        throw GGUFLoadError("Validation failed: Missing 'token_embd.weight'");
    }
    if (inventory.find("output_norm.weight") == inventory.end()) {
        throw GGUFLoadError("Validation failed: Missing 'output_norm.weight'");
    }

    for (uint32_t i = 0; i < metadata_.block_count; ++i) {
        const std::string prefix = "blk." + std::to_string(i) + ".";

        if (metadata_.architecture == "qwen35") {
            // Shared tensors (both layer types)
            const std::vector<std::string> shared_tensors = {
                "attn_norm.weight", "post_attention_norm.weight",
                "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"
            };
            for (const auto& t : shared_tensors) {
                if (inventory.find(prefix + t) == inventory.end()) {
                    throw GGUFLoadError("Validation failed: Missing '" + prefix + t + "'");
                }
            }

            if (metadata_.is_full_attention_layer(i)) {
                // Full attention layer tensors
                const std::vector<std::string> attn_tensors = {
                    "attn_q.weight", "attn_k.weight", "attn_v.weight",
                    "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"
                };
                for (const auto& t : attn_tensors) {
                    if (inventory.find(prefix + t) == inventory.end()) {
                        throw GGUFLoadError("Validation failed: Missing '" + prefix + t + "' for attention layer " + std::to_string(i));
                    }
                }
            } else {
                // SSM (GatedDeltaNet) layer tensors
                const std::vector<std::string> ssm_tensors = {
                    "ssm_a", "ssm_conv1d.weight", "ssm_dt.bias",
                    "ssm_alpha.weight", "ssm_beta.weight",
                    "attn_qkv.weight", "attn_gate.weight",
                    "ssm_norm.weight", "ssm_out.weight"
                };
                for (const auto& t : ssm_tensors) {
                    if (inventory.find(prefix + t) == inventory.end()) {
                        throw GGUFLoadError("Validation failed: Missing '" + prefix + t + "' for SSM layer " + std::to_string(i));
                    }
                }
            }
        } else {
            // Original validation for qwen2/qwen3
            const std::vector<std::string> required_tensors = {
                "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
                "ffn_up.weight", "ffn_down.weight"
            };
            for (const auto& t_name : required_tensors) {
                const std::string full_name = prefix + t_name;
                if (inventory.find(full_name) == inventory.end()) {
                    throw GGUFLoadError("Validation failed: Missing tensor '" + full_name + "' for block " + std::to_string(i));
                }
            }
        }
    }

    std::cout << "Tensor inventory validation passed." << std::endl;
}

std::string QwenGGUFLoader::read_string_from_mem(size_t& offset)
{
    const uint64_t len = read_value_from_mem<uint64_t>(offset);
    if (offset + len > file_mapper_->size()) {
        throw GGUFLoadError("Read past end of file while reading string.");
    }
    std::string s(reinterpret_cast<const char*>(file_mapper_->data() + offset), len);
    offset += len;
    return s;
}

void QwenGGUFLoader::skip_gguf_value_from_mem(size_t& offset, GGUFValueType type)
{
    switch (type)
    {
        case GGUFValueType::UINT8: offset += sizeof(uint8_t); break;
        case GGUFValueType::INT8: offset += sizeof(int8_t); break;
        case GGUFValueType::UINT16: offset += sizeof(uint16_t); break;
        case GGUFValueType::INT16: offset += sizeof(int16_t); break;
        case GGUFValueType::UINT32: offset += sizeof(uint32_t); break;
        case GGUFValueType::INT32: offset += sizeof(int32_t); break;
        case GGUFValueType::FLOAT32: offset += sizeof(float); break;
        case GGUFValueType::BOOL: offset += sizeof(bool); break;
        case GGUFValueType::UINT64: offset += sizeof(uint64_t); break;
        case GGUFValueType::INT64: offset += sizeof(int64_t); break;
        case GGUFValueType::FLOAT64: offset += sizeof(double); break;
        case GGUFValueType::STRING: {
            const uint64_t len = read_value_from_mem<uint64_t>(offset);
            offset += len;
            break;
        }
        case GGUFValueType::ARRAY: {
            const GGUFValueType array_type = read_value_from_mem<GGUFValueType>(offset);
            const uint64_t array_len = read_value_from_mem<uint64_t>(offset);
            if (array_type == GGUFValueType::STRING) {
                for (uint64_t i = 0; i < array_len; ++i) {
                    skip_gguf_value_from_mem(offset, GGUFValueType::STRING);
                }
            } else {
                size_t element_size = 0;
                switch (array_type) {
                    case GGUFValueType::UINT8: element_size = sizeof(uint8_t); break;
                    case GGUFValueType::INT8: element_size = sizeof(int8_t); break;
                    case GGUFValueType::UINT16: element_size = sizeof(uint16_t); break;
                    case GGUFValueType::INT16: element_size = sizeof(int16_t); break;
                    case GGUFValueType::UINT32: element_size = sizeof(uint32_t); break;
                    case GGUFValueType::INT32: element_size = sizeof(int32_t); break;
                    case GGUFValueType::FLOAT32: element_size = sizeof(float); break;
                    case GGUFValueType::BOOL: element_size = sizeof(bool); break;
                    case GGUFValueType::UINT64: element_size = sizeof(uint64_t); break;
                    case GGUFValueType::INT64: element_size = sizeof(int64_t); break;
                    case GGUFValueType::FLOAT64: element_size = sizeof(double); break;
                    default:
                        throw GGUFLoadError("Unknown array type in GGUF metadata.");
                }
                offset += array_len * element_size;
            }
            break;
        }
    }
    if (offset > file_mapper_->size()) {
        throw GGUFLoadError("Read past end of file during skip.");
    }
}

std::unique_ptr<QwenGGUFLoader> create_gguf_loader() { return std::make_unique<QwenGGUFLoader>(); }

// Load tensor metadata only (no data copying)
void QwenGGUFLoader::load_tensor_metadata(ggml_context *ctx, std::unordered_map<std::string, ggml_tensor *> &tensors)
{
    if (!file_mapper_) {
        throw GGUFLoadError("Model not loaded or mapped.");
    }

    for (const auto &[name, meta] : metadata_.tensor_inventory)
    {
        std::vector<int64_t> shape(meta.shape.begin(), meta.shape.end());
        ggml_tensor *tensor = ggml_new_tensor(ctx, meta.type, shape.size(), shape.data());
        if (!tensor)
        {
            throw GGUFLoadError("Failed to create tensor: " + name);
        }
        ggml_set_name(tensor, name.c_str());

        // NOTE: tensor->data is NOT set here. 
        // Caller must use get_tensor_data() and ggml_backend_tensor_set()
        
        tensors[name] = tensor;
    }
    std::cout << "Successfully created " << tensors.size() << " tensor metadata entries." << std::endl;
}

// Get raw pointer to tensor data in mmap'd file
const void* QwenGGUFLoader::get_tensor_data(const std::string& name) const
{
    if (!is_loaded_) {
        throw GGUFLoadError("Model not loaded");
    }
    
    auto it = metadata_.tensor_inventory.find(name);
    if (it == metadata_.tensor_inventory.end()) {
        throw GGUFLoadError("Tensor not found: " + name);
    }
    
    const size_t tensor_offset = tensor_data_offset_ + it->second.offset;
    
    if (tensor_offset > file_mapper_->size()) {
        throw GGUFLoadError("Tensor data offset for \"" + name + "\" is beyond file size");
    }
    
    return file_mapper_->data() + tensor_offset;
}