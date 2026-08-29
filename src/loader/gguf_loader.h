#pragma once
// gguf_loader.h — GGUF file -> ModelMetadata.
//
// Responsibility: mmap a GGUF file, parse its header/metadata/tensor directory,
//   and populate ModelMetadata — including the two fail-loud validators that
//   decide whether this build can run the file at all:
//     validate_architecture            — is general.architecture in the registry's
//                                        allow-list? (throws if not)
//     validate_inventory_for_architecture — does the file carry every tensor the
//                                        recipe needs, correctly shaped?
//   Both run during load_metadata, so a completed load IS acceptance. This is
//   the engine's only architecture/inventory gate; there is no second copy.
// State owned: the file mapping and the parsed directory. The mapping is handed
//   to Model and released once weights are copied to the backend (see model.h).
// Note: the big tokenizer arrays (vocab, merges, scores, token types) are
//   intercepted by key here and placed on ModelMetadata's typed members; only
//   scalar/small-array family keys reach the generic GGUFValue bag (gguf_value.h).
// Unit tests: tests/unit/test_loader.cpp, tests/unit/test_gguf_kv_bag.cpp

#include "engine/model.h"
#include "ggml.h"
#include "loader/platform.h"
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

class GGUFLoader {
public:
    GGUFLoader();
    ~GGUFLoader();

    // `validate_as_text_model` (default true): run text-model arch +
    // inventory validation against the registered architectures (Qwen /
    // Gemma family). Pass false for non-text-model GGUFs (e.g. the
    // Gemma 3 mmproj, which declares general.architecture="clip" and
    // ships a vision-tensor inventory neither validator knows about).
    // The binary GGUF parse + raw_kv extraction + tensor inventory
    // population are identical either way; only the two text-model
    // gates are skipped. See src/vision/vision_loader.cpp.
    void load_model(const std::string& path, bool validate_as_text_model = true);
    void extract_metadata(ModelMetadata& metadata) const;
    
    size_t calculate_tensors_memory_size() const;
    
    // Original method - backward compatible (copies data into context)
    void load_all_tensors(ggml_context* ctx, std::unordered_map<std::string, ggml_tensor*>& tensors);
    
    // NEW: Load tensor structs only (for backend usage)
    void load_tensor_metadata(ggml_context* ctx, std::unordered_map<std::string, ggml_tensor*>& tensors);
    
    // NEW: Get raw tensor data pointer (for backend copying)
    const void* get_tensor_data(const std::string& name) const;

    // Release the mmap of the GGUF once its tensor data has been copied to the
    // backend buffer. Until this is called the whole file stays resident
    // alongside the backend copy -- two full copies of the weights, which on a
    // 27B is ~13 GB of avoidable residency. Metadata is already extracted into
    // owning structures, so nothing else needs the mapping.
    //
    // After this, get_tensor_data / load_all_tensors / load_tensor_metadata
    // throw fail-loud rather than dereferencing a released mapping. Idempotent.
    void release_file_mapping();

    void validate_tensor_shape(struct ggml_tensor* tensor, const std::vector<int64_t>& expected_dims);

    void validate_architecture(const ModelMetadata& meta) const;
    void unload_model();
    
    bool is_loaded() const { return is_loaded_; }

    const TensorMetadata& get_tensor_metadata(const std::string& name) const;

private:
    std::string model_path_;
    std::unique_ptr<FileMapper> file_mapper_;
    bool is_loaded_;
    bool validate_as_text_model_ = true;  // set by load_model; gates arch + inventory validation
    uint64_t tensor_data_offset_;
    ModelMetadata metadata_;

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

    // Read an ARRAY value at `offset` into metadata_.raw_kv under `key`.
    // Element types GGUFValue models (uint32/int32/float/bool/string) are
    // stored; any other element type is skipped without storing. Advances
    // `offset` past the whole array either way. See the ARRAY case in
    // parse_metadata for why unmodelled element types are skipped, not refused.
    void read_array_into_kv(size_t& offset, const std::string& key);

    size_t calculate_tensor_bytes(const TensorMetadata& meta) const;
    void validate_tensor_inventory() const;
    void is_valid_utf8(const std::string& str) const;

    using MetadataHandler = std::function<void(GGUFLoader*, size_t&, GGUFValueType)>;
    static const std::unordered_map<std::string_view, MetadataHandler>& get_metadata_handlers();

    void cleanup_resources();
};

// Factory function for creating a loader
std::unique_ptr<GGUFLoader> create_gguf_loader();

// Dispatches inventory validation based on meta.architecture via the model
// registry.  Throws GGUFLoadError for an unregistered architecture, or when
// the registered validator rejects the inventory.
void validate_inventory_for_architecture(const ModelMetadata& meta);
