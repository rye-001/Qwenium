#pragma once
// gguf_loader.h — GGUF file -> ModelMetadata.
//
// Responsibility: mmap a GGUF file, parse its header/metadata/tensor directory,
//   and populate ModelMetadata — including the three fail-loud validators that
//   decide whether this build can run the file at all:
//     validate_tensor_type             — is every tensor's ggml type one THIS
//                                        build knows? (throws if not)
//     validate_architecture            — is general.architecture in the registry's
//                                        allow-list? (throws if not)
//     validate_inventory_for_architecture — does the file carry every tensor the
//                                        recipe needs, correctly shaped?
//   All three run during load_metadata, so a completed load IS acceptance. This is
//   the engine's only architecture/inventory gate; there is no second copy.
// State owned: the file mapping and the parsed directory. The mapping is handed
//   to Model and released once weights are copied to the backend (see model.h).
// Note: the big tokenizer arrays (vocab, merges, scores, token types) are
//   intercepted by key here and placed on ModelMetadata's typed members; only
//   scalar/small-array family keys reach the generic GGUFValue bag (gguf_value.h).
// Unit tests: tests/unit/test_gguf_loader.cpp, tests/unit/test_gguf_kv_bag.cpp

#include "engine/model.h"
#include "ggml.h"
#include "loader/platform.h"
#include <string>
#include <memory>
#include <cstddef>
#include <cstdint>
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
    
    // NEW: Load tensor structs only (for backend usage).
    //
    // `max_blocks` (default: every block, i.e. today's behavior byte-for-byte):
    // when finite, restricts the struct set to token_embd.weight plus
    // blk.{0..max_blocks-1}.* — no output_norm.weight, no output.weight, no
    // blk.{max_blocks..block_count-1}.*. This is the exact subset a forward
    // pass truncated at DecodePolicy::truncate_after_layer == max_blocks-1 can
    // ever read (architecture.md §6, causality: a layer cannot depend on one
    // above it), which is what makes the omission safe rather than a guess.
    // Used by --lens-verify-only (Model::load_tensors) to skip the SSD
    // read + backend copy for blocks the lens calibration will never tap.
    // Does NOT touch ModelMetadata::weights_hash — that is computed once in
    // GGUFLoader::load_model from the FULL parsed tensor inventory, before
    // this method (or its caller) ever runs, so a partial load and a full
    // load of the same file stamp the identical hash. Hashing only the
    // loaded subset would make a verify-only report incomparable with a
    // full-server one, which is exactly what config.weights exists to
    // prevent (docs/lens-format.md).
    //
    // `keep_output_head` (default false): on a partial load, also keep
    // output_norm.weight and output.weight — for a truncated server that reads
    // logits after its last loaded block (--lens-verdict, /v1/verdict). No
    // effect on a full load, which always keeps them.
    void load_tensor_metadata(ggml_context* ctx, std::unordered_map<std::string, ggml_tensor*>& tensors,
                               uint32_t max_blocks = UINT32_MAX, bool keep_output_head = false);
    
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

    // Base/size of the live GGUF mapping, for backing a backend buffer with the
    // mapped pages instead of copying into a fresh one (see Model::load_tensors,
    // mmap-weights path). Fail-loud once release_file_mapping() has run: a
    // buffer built over a released mapping would be a use-after-munmap.
    const void* mapped_base() const;
    size_t      mapped_size() const;

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

// Is `type_id` a ggml type THIS build knows? Throws GGUFLoadError if not.
//
// WHY THIS IS ITS OWN GATE. ggml_type_size() and ggml_blck_size() guard with
// plain assert(), which NDEBUG removes — so in a Release build an out-of-range
// type id silently indexes type_traits[] past its end and returns garbage
// instead of failing. That garbage reaches us as a tensor byte count: a
// 5.9 GB ternary file measured "567 MB" before this check existed, which is
// the size a ggml context would then have been given for it. ggml's own
// GGML_ASSERT in ggml_new_tensor_impl does abort, but only later, after we
// have already sized memory off a bogus number.
//
// Runs on EVERY GGUF, text model or mmproj, because an unknown type is fatal
// regardless of which tensor namespace the file uses.
//
// The live case: Prism ML's Ternary Bonsai 2 ships PQ2_0 (142) and PTQ1_0
// (143), fork types well past GGML_TYPE_COUNT. Those files also carry a
// prism.hadamard.* weight rotation that the runtime must mirror on
// activations, so admitting the type id alone would not be enough to run them
// correctly — see docs/architecture.md.
void validate_tensor_type(const std::string& tensor_name, uint32_t type_id);
