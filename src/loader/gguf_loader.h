#pragma once

#include "../qwen3-core/qwen3-model.h"
#include "ggml.h"
#include "../qwen3-core/platform.h"
#include <string>
#include <memory>
#include <cstddef>
#include <fstream>
#include <unordered_map>
#include <stdexcept>
#include <functional>
#include <string_view>
#include <vector>
#include <unordered_map>

class GGUFLoadError : public std::runtime_error {
public:
    explicit GGUFLoadError(const std::string& message)
        : std::runtime_error(message) {}
};

struct ggml_context;
struct ggml_tensor;

// GGUF value types
enum class GGUFValueType : uint32_t
{
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

class QwenGGUFLoader {
public:
    QwenGGUFLoader();
    ~QwenGGUFLoader();

    void load_model(const std::string& path);
    void extract_metadata(Qwen3Metadata& metadata) const;
    
    size_t calculate_tensors_memory_size() const;
    
    // Original method - backward compatible (copies data into context)
    void load_all_tensors(ggml_context* ctx, std::unordered_map<std::string, ggml_tensor*>& tensors);
    
    // NEW: Load tensor structs only (for backend usage)
    void load_tensor_metadata(ggml_context* ctx, std::unordered_map<std::string, ggml_tensor*>& tensors);
    
    // NEW: Get raw tensor data pointer (for backend copying)
    const void* get_tensor_data(const std::string& name) const;

    void validate_tensor_shape(struct ggml_tensor* tensor, const std::vector<int64_t>& expected_dims);

    void validate_architecture(const Qwen3Metadata& meta) const;
    void unload_model();
    
    bool is_loaded() const { return is_loaded_; }

    const TensorMetadata& get_tensor_metadata(const std::string& name) const;

private:
    std::string model_path_;
    std::unique_ptr<FileMapper> file_mapper_;
    bool is_loaded_;
    uint64_t tensor_data_offset_;
    Qwen3Metadata metadata_;

    void parse_and_validate_metadata(size_t& offset);
    void parse_tensor_inventory(size_t& offset, uint64_t tensor_count);
    std::string read_string_from_mem(size_t& offset);

    template<typename T>
    T read_value_from_mem(size_t& offset) {
        if (offset + sizeof(T) > file_mapper_->size()) {
            throw GGUFLoadError("Attempt to read past the end of the mapped file.");
        }
        T val;
        memcpy(&val, file_mapper_->data() + offset, sizeof(T));
        offset += sizeof(T);
        return val;
    }

    void skip_gguf_value_from_mem(size_t& offset, GGUFValueType type);

    size_t calculate_tensor_bytes(const TensorMetadata& meta) const;
    void validate_tensor_inventory() const;
    void is_valid_utf8(const std::string& str) const;

    using MetadataHandler = std::function<void(QwenGGUFLoader*, size_t&, GGUFValueType)>;
    static const std::unordered_map<std::string_view, MetadataHandler>& get_metadata_handlers();

    void cleanup_resources();
};

// Factory function for creating a loader
std::unique_ptr<QwenGGUFLoader> create_gguf_loader();

// Validates the tensor inventory of a loaded qwen35moe model. Exposed as a
// free function so unit tests can inject a crafted inventory to exercise the
// fail-loud error contract without needing a real GGUF file.
// Throws GGUFLoadError naming the missing tensor if any required key is absent.
void validate_qwen35moe_inventory(const Qwen3Metadata& meta);
