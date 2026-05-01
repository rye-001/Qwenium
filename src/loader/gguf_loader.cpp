#include "gguf_loader.h"
#include "../models/model_registry.h"
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
static constexpr std::string_view KEY_TOKENIZER_UNK_TOKEN_ID = "tokenizer.ggml.unknown_token_id";
static constexpr std::string_view KEY_TOKENIZER_SCORES = "tokenizer.ggml.scores";

struct GGUFHeader
{
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

GGUFLoader::GGUFLoader() : is_loaded_(false), tensor_data_offset_(0) {}

GGUFLoader::~GGUFLoader() { cleanup_resources(); }

void GGUFLoader::load_model(const std::string &path)
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

void GGUFLoader::extract_metadata(ModelMetadata &metadata) const
{
    if (!is_loaded_)
    {
        throw GGUFLoadError("Model is not loaded, cannot extract metadata.");
    }
    metadata = metadata_;
}

size_t GGUFLoader::calculate_tensors_memory_size() const
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

void GGUFLoader::load_all_tensors(ggml_context *ctx, std::unordered_map<std::string, ggml_tensor *> &tensors)
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

const TensorMetadata& GGUFLoader::get_tensor_metadata(const std::string& name) const
{
    auto it = metadata_.tensor_inventory.find(name);
    if (it == metadata_.tensor_inventory.end()) {
        throw GGUFLoadError("Tensor metadata not found for: " + name);
    }
    return it->second;
}




void GGUFLoader::validate_tensor_shape(struct ggml_tensor *tensor,
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

size_t GGUFLoader::calculate_tensor_bytes(const TensorMetadata &meta) const
{
    size_t elements = 1;
    for (uint64_t dim : meta.shape)
    {
        elements *= dim;
    }
    return ggml_type_size(meta.type) * elements / ggml_blck_size(meta.type);
}

void GGUFLoader::validate_architecture(const ModelMetadata& meta) const
{
    if (is_architecture_registered(meta.architecture)) {
        std::cout << "Validated model architecture: " << meta.architecture << std::endl;
        return;
    }
    std::string allow_list;
    for (const auto& a : registered_architectures()) {
        if (!allow_list.empty()) allow_list += ", ";
        allow_list += "'" + a + "'";
    }
    throw GGUFLoadError(
        "validate_architecture: expected one of: " + allow_list +
        ", got '" + meta.architecture + "'");
}

void GGUFLoader::unload_model()
{
    cleanup_resources();
    is_loaded_ = false;
}

void GGUFLoader::parse_and_validate_metadata(size_t& offset)
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
            metadata_.raw_kv.set(key, metadata_.model_name);
        } else if (key == prefix + "block_count") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.block_count = read_value_from_mem<uint32_t>(offset);
            metadata_.raw_kv.set(key, metadata_.block_count);
        } else if (key == prefix + "embedding_length") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.embedding_length = read_value_from_mem<uint32_t>(offset);
            metadata_.raw_kv.set(key, metadata_.embedding_length);
        } else if (key == prefix + "feed_forward_length") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.feed_forward_length = read_value_from_mem<uint32_t>(offset);
            metadata_.raw_kv.set(key, metadata_.feed_forward_length);
        } else if (key == prefix + "context_length") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.context_length = read_value_from_mem<uint32_t>(offset);
            metadata_.raw_kv.set(key, metadata_.context_length);
        } else if (key == prefix + "attention.head_count") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.attention_head_count = read_value_from_mem<uint32_t>(offset);
            metadata_.raw_kv.set(key, metadata_.attention_head_count);
        } else if (key == prefix + "attention.head_count_kv") {
            // Gemma 4 stores this as a per-layer ARRAY of UINT32 (one entry
            // per block) because global vs sliding layers have different KV
            // head counts.  Existing architectures (Qwen, Gemma 1/2/3) keep
            // it as a scalar UINT32 — accept both.
            //
            // Array form: skip the bytes; leave the typed scalar member at
            // its default (0) and don't populate raw_kv.  The recipe is
            // responsible for deriving per-layer values (Gemma4Config does
            // this from blk.<il>.attn_k.weight shapes — no GGUFKVBag array
            // support required).  Storing only the first element here would
            // be a foot-gun: any consumer treating it as the uniform value
            // would silently get wrong results.
            if (type == GGUFValueType::UINT32) {
                metadata_.attention_head_count_kv = read_value_from_mem<uint32_t>(offset);
                metadata_.raw_kv.set(key, metadata_.attention_head_count_kv);
            } else if (type == GGUFValueType::ARRAY) {
                const GGUFValueType array_type = read_value_from_mem<GGUFValueType>(offset);
                if (array_type != GGUFValueType::UINT32 && array_type != GGUFValueType::INT32) {
                    throw GGUFLoadError(
                        key + ": expected ARRAY<UINT32> or UINT32, "
                              "got ARRAY of element-type " +
                        std::to_string(static_cast<int>(array_type)));
                }
                const uint64_t n = read_value_from_mem<uint64_t>(offset);
                for (uint64_t i = 0; i < n; ++i) {
                    (void)read_value_from_mem<uint32_t>(offset);
                }
            } else {
                throw GGUFLoadError(
                    key + ": expected UINT32 or ARRAY<UINT32>, got type " +
                    std::to_string(static_cast<int>(type)));
            }
        } else if (key == prefix + "attention.key_length") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.attention_key_length = read_value_from_mem<uint32_t>(offset);
            metadata_.raw_kv.set(key, metadata_.attention_key_length);
        } else if (key == prefix + "attention.value_length") {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.attention_value_length = read_value_from_mem<uint32_t>(offset);
            metadata_.raw_kv.set(key, metadata_.attention_value_length);
        } else if (key == prefix + "rope.freq_base") {
            if (type != GGUFValueType::FLOAT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.rope_freq_base = read_value_from_mem<float>(offset);
            metadata_.raw_kv.set(key, metadata_.rope_freq_base);
        } else if (key == prefix + "attention.layer_norm_rms_epsilon") {
            if (type != GGUFValueType::FLOAT32) throw GGUFLoadError(key + " has unexpected type.");
            metadata_.rms_norm_eps = read_value_from_mem<float>(offset);
            metadata_.raw_kv.set(key, metadata_.rms_norm_eps);
        } else if (key == KEY_TOKENIZER_MODEL) {
            if (type != GGUFValueType::STRING) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_MODEL) + " has unexpected type.");
            metadata_.tokenizer_type = read_string_from_mem(offset);
            metadata_.raw_kv.set(key, metadata_.tokenizer_type);
        } else if (key == KEY_TOKENIZER_PRE) {
            if (type != GGUFValueType::STRING) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_PRE) + " has unexpected type.");
            metadata_.tokenizer_pre = read_string_from_mem(offset);
            metadata_.raw_kv.set(key, metadata_.tokenizer_pre);
        } else if (key == KEY_TOKENIZER_TOKENS) {
            // Array — populates typed member only; arrays not stored in raw_kv.
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
            // Array — populates typed member only; arrays not stored in raw_kv.
            if (type != GGUFValueType::ARRAY) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_TOKEN_TYPE) + " has unexpected type.");
            const GGUFValueType array_type = read_value_from_mem<GGUFValueType>(offset);
            if (array_type != GGUFValueType::INT32) throw GGUFLoadError("Tokenizer token_type array has unexpected element type.");

            const uint64_t num_tokens = read_value_from_mem<uint64_t>(offset);
            metadata_.token_types.resize(num_tokens);
            for (uint64_t i = 0; i < num_tokens; ++i) {
                metadata_.token_types[i] = static_cast<TokenType>(read_value_from_mem<int32_t>(offset));
            }
        } else if (key == KEY_TOKENIZER_MERGES) {
            // Array — populates typed member only; arrays not stored in raw_kv.
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
            const uint32_t v = read_value_from_mem<uint32_t>(offset);
            metadata_.eos_token_id = static_cast<int32_t>(v);
            metadata_.raw_kv.set(key, v);
        } else if (key == KEY_TOKENIZER_BOS_TOKEN_ID) {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_BOS_TOKEN_ID) + " has unexpected type.");
            const uint32_t v = read_value_from_mem<uint32_t>(offset);
            metadata_.bos_token_id = static_cast<int32_t>(v);
            metadata_.raw_kv.set(key, v);
        } else if (key == KEY_TOKENIZER_PADDING_TOKEN_ID) {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_PADDING_TOKEN_ID) + " has unexpected type.");
            const uint32_t v = read_value_from_mem<uint32_t>(offset);
            metadata_.padding_token_id = static_cast<int32_t>(v);
            metadata_.raw_kv.set(key, v);
        } else if (key == KEY_TOKENIZER_ADD_BOS_TOKEN) {
            if (type != GGUFValueType::BOOL) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_ADD_BOS_TOKEN) + " has unexpected type.");
            metadata_.add_bos_token = read_value_from_mem<bool>(offset);
            metadata_.raw_kv.set(key, metadata_.add_bos_token);
        } else if (key == KEY_TOKENIZER_UNK_TOKEN_ID) {
            if (type != GGUFValueType::UINT32) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_UNK_TOKEN_ID) + " has unexpected type.");
            const uint32_t v = read_value_from_mem<uint32_t>(offset);
            metadata_.unknown_token_id = static_cast<int32_t>(v);
            metadata_.raw_kv.set(key, v);
        } else if (key == KEY_TOKENIZER_SCORES) {
            // Array — populates typed member only; arrays not stored in raw_kv.
            if (type != GGUFValueType::ARRAY) throw GGUFLoadError("Metadata key " + std::string(KEY_TOKENIZER_SCORES) + " has unexpected type.");
            const GGUFValueType array_type = read_value_from_mem<GGUFValueType>(offset);
            if (array_type != GGUFValueType::FLOAT32) throw GGUFLoadError("Tokenizer scores array must be FLOAT32.");
            const uint64_t n = read_value_from_mem<uint64_t>(offset);
            metadata_.scores.resize(n);
            for (uint64_t i = 0; i < n; ++i) {
                metadata_.scores[i] = read_value_from_mem<float>(offset);
            }
        } else {
            // Unknown key: store supported scalar types in raw_kv so recipe
            // from_metadata factories can read family-specific fields from there.
            // Array types are deliberately skipped — they belong on typed members.
            // Extend GGUFValue and add a branch here if a real consumer needs array support.
            switch (type) {
                case GGUFValueType::UINT32:
                    metadata_.raw_kv.set(key, read_value_from_mem<uint32_t>(offset));
                    break;
                case GGUFValueType::INT32:
                    metadata_.raw_kv.set(key, read_value_from_mem<int32_t>(offset));
                    break;
                case GGUFValueType::FLOAT32:
                    metadata_.raw_kv.set(key, read_value_from_mem<float>(offset));
                    break;
                case GGUFValueType::BOOL:
                    metadata_.raw_kv.set(key, read_value_from_mem<bool>(offset));
                    break;
                case GGUFValueType::STRING:
                    metadata_.raw_kv.set(key, read_string_from_mem(offset));
                    break;
                default:
                    // Narrower integer types, UINT64, INT64, FLOAT64, ARRAY — skip.
                    skip_gguf_value_from_mem(offset, type);
                    break;
            }
        }
    }
}

void GGUFLoader::cleanup_resources()
{
    file_mapper_.reset();
}

void GGUFLoader::is_valid_utf8(const std::string &str) const
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

void GGUFLoader::parse_tensor_inventory(size_t& offset, uint64_t tensor_count)
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

// Dispatch inventory validation via the model registry.
// The validator for each architecture lives in its recipe file.
// Re-wraps std::runtime_error as GGUFLoadError so callers see a consistent type.
void validate_inventory_for_architecture(const ModelMetadata& meta)
{
    if (!is_architecture_registered(meta.architecture)) {
        std::string allow_list;
        for (const auto& a : registered_architectures()) {
            if (!allow_list.empty()) allow_list += ", ";
            allow_list += "'" + a + "'";
        }
        throw GGUFLoadError(
            "validate_inventory_for_architecture: expected one of: " + allow_list +
            ", got '" + meta.architecture + "'");
    }
    try {
        lookup_inventory_validator(meta.architecture)(meta);
    } catch (const std::runtime_error& e) {
        throw GGUFLoadError(e.what());
    }
}

void GGUFLoader::validate_tensor_inventory() const
{
    std::cout << "Validating tensor inventory..." << std::endl;
    validate_inventory_for_architecture(metadata_);
    std::cout << "Tensor inventory validation passed." << std::endl;
}

std::string GGUFLoader::read_string_from_mem(size_t& offset)
{
    const uint64_t len = read_value_from_mem<uint64_t>(offset);
    if (offset + len > file_mapper_->size()) {
        throw GGUFLoadError("Read past end of file while reading string.");
    }
    std::string s(reinterpret_cast<const char*>(file_mapper_->data() + offset), len);
    offset += len;
    return s;
}

void GGUFLoader::skip_gguf_value_from_mem(size_t& offset, GGUFValueType type)
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

std::unique_ptr<GGUFLoader> create_gguf_loader() { return std::make_unique<GGUFLoader>(); }

// Load tensor metadata only (no data copying)
void GGUFLoader::load_tensor_metadata(ggml_context *ctx, std::unordered_map<std::string, ggml_tensor *> &tensors)
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
const void* GGUFLoader::get_tensor_data(const std::string& name) const
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