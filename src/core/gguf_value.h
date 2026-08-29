#pragma once

// gguf_value.h — generic GGUF scalar KV bag
//
// GGUFValue is a tagged union over the scalar types that appear in
// family-specific GGUF metadata keys, plus a vector alternative for each.
//
// Arrays were originally excluded ("add support when a real consumer needs
// it").  Three consumers arrived at once with the Qwen 3.5-family vision work
// (docs/plan-qwen35-vision-impl.md §3.8): `rope.dimension_sections` (INT32[4],
// M-RoPE section widths), `clip.vision.image_{mean,std}` (FLOAT32[3],
// normalization constants that are currently hardcoded), and
// `clip.vision.is_deepstack_layers` (BOOL[n_layer]).  All five element types
// are covered rather than only the three needed, because the alternative is
// discovering a fourth type later and special-casing again — which is what
// this replaces.
//
// The BIG arrays (tokenizer vocab, scores, merges, token types) never reach
// here: the loader intercepts them by key and puts them on ModelMetadata's
// typed members.  That split is deliberate and unchanged — a 250 K-entry vocab
// has no business in a variant.
//
// GGUFKVBag wraps an unordered_map and enforces all reads through typed
// accessors that follow the fail-loud error contract: "GGUFKVBag: key 'X'
// expected <type>, got <actual>" or "GGUFKVBag: key 'X' missing".
//
// This header is intentionally free of ggml/GGUF C-library includes so it
// can be used from any translation unit without dragging in the gguf headers.

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

// Tagged union over the five scalar GGUF types we actually encounter in
// family-specific metadata keys.
using GGUFValue = std::variant<uint32_t, int32_t, float, bool, std::string,
                               std::vector<uint32_t>, std::vector<int32_t>,
                               std::vector<float>, std::vector<bool>,
                               std::vector<std::string>>;

// Type-checked key-value bag populated by the loader during metadata parsing.
// The underlying map is not exposed — all reads go through the typed accessors.
class GGUFKVBag {
public:
    // ── Presence ──────────────────────────────────────────────────────────────
    bool contains(const std::string& key) const {
        return values_.count(key) != 0;
    }

    // ── Writer — the loader is the only legitimate caller ─────────────────────
    void set(const std::string& key, GGUFValue value) {
        values_[key] = std::move(value);
    }

    // ── Required accessors — throw on missing or wrong type ───────────────────
    // Error format: "GGUFKVBag: key '<key>' missing"
    //               "GGUFKVBag: key '<key>' expected <type>, got <actual>"
    uint32_t    get_uint32(const std::string& key) const { return extract<uint32_t>(key); }
    int32_t     get_int32 (const std::string& key) const { return extract<int32_t>(key); }
    float       get_float (const std::string& key) const { return extract<float>(key); }
    bool        get_bool  (const std::string& key) const { return extract<bool>(key); }
    std::string get_string(const std::string& key) const { return extract<std::string>(key); }

    // ── Optional accessors — nullopt if missing, throw if present but wrong type
    std::optional<uint32_t>    get_uint32_opt(const std::string& key) const { return extract_opt<uint32_t>(key); }
    std::optional<int32_t>     get_int32_opt (const std::string& key) const { return extract_opt<int32_t>(key); }
    std::optional<float>       get_float_opt (const std::string& key) const { return extract_opt<float>(key); }
    std::optional<bool>        get_bool_opt  (const std::string& key) const { return extract_opt<bool>(key); }
    std::optional<std::string> get_string_opt(const std::string& key) const { return extract_opt<std::string>(key); }

    // ── Array accessors ───────────────────────────────────────────────────────
    // Same contract as the scalars, one alternative per element type. Required
    // forms return a const reference (no copy); optional forms copy, which is
    // fine because every array that reaches this bag is metadata-sized — the
    // tokenizer's big arrays are intercepted by the loader before it.
    const std::vector<uint32_t>&    get_uint32_array(const std::string& key) const { return extract_ref<std::vector<uint32_t>>(key); }
    const std::vector<int32_t>&     get_int32_array (const std::string& key) const { return extract_ref<std::vector<int32_t>>(key); }
    const std::vector<float>&       get_float_array (const std::string& key) const { return extract_ref<std::vector<float>>(key); }
    const std::vector<bool>&        get_bool_array  (const std::string& key) const { return extract_ref<std::vector<bool>>(key); }
    const std::vector<std::string>& get_string_array(const std::string& key) const { return extract_ref<std::vector<std::string>>(key); }

    std::optional<std::vector<uint32_t>>    get_uint32_array_opt(const std::string& key) const { return extract_opt<std::vector<uint32_t>>(key); }
    std::optional<std::vector<int32_t>>     get_int32_array_opt (const std::string& key) const { return extract_opt<std::vector<int32_t>>(key); }
    std::optional<std::vector<float>>       get_float_array_opt (const std::string& key) const { return extract_opt<std::vector<float>>(key); }
    std::optional<std::vector<bool>>        get_bool_array_opt  (const std::string& key) const { return extract_opt<std::vector<bool>>(key); }
    std::optional<std::vector<std::string>> get_string_array_opt(const std::string& key) const { return extract_opt<std::vector<std::string>>(key); }

private:
    std::unordered_map<std::string, GGUFValue> values_;

    // Human-readable name for the type parameter T.
    template<typename T>
    static const char* tname() {
        if constexpr (std::is_same_v<T, uint32_t>)    return "uint32";
        else if constexpr (std::is_same_v<T, int32_t>)  return "int32";
        else if constexpr (std::is_same_v<T, float>)    return "float";
        else if constexpr (std::is_same_v<T, bool>)     return "bool";
        else if constexpr (std::is_same_v<T, std::string>) return "string";
        else if constexpr (std::is_same_v<T, std::vector<uint32_t>>)    return "uint32[]";
        else if constexpr (std::is_same_v<T, std::vector<int32_t>>)     return "int32[]";
        else if constexpr (std::is_same_v<T, std::vector<float>>)       return "float[]";
        else if constexpr (std::is_same_v<T, std::vector<bool>>)        return "bool[]";
        else if constexpr (std::is_same_v<T, std::vector<std::string>>) return "string[]";
        else return "unknown";
    }

    // Human-readable name for the active alternative of a GGUFValue.
    static std::string vname(const GGUFValue& v) {
        return std::visit([](const auto& x) -> std::string {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, uint32_t>)    return "uint32";
            else if constexpr (std::is_same_v<T, int32_t>)  return "int32";
            else if constexpr (std::is_same_v<T, float>)    return "float";
            else if constexpr (std::is_same_v<T, bool>)     return "bool";
            else if constexpr (std::is_same_v<T, std::string>) return "string";
            else if constexpr (std::is_same_v<T, std::vector<uint32_t>>)    return "uint32[]";
            else if constexpr (std::is_same_v<T, std::vector<int32_t>>)     return "int32[]";
            else if constexpr (std::is_same_v<T, std::vector<float>>)       return "float[]";
            else if constexpr (std::is_same_v<T, std::vector<bool>>)        return "bool[]";
            else if constexpr (std::is_same_v<T, std::vector<std::string>>) return "string[]";
            else return "unknown";
        }, v);
    }

    template<typename T>
    T extract(const std::string& key) const {
        auto it = values_.find(key);
        if (it == values_.end()) {
            throw std::runtime_error("GGUFKVBag: key '" + key + "' missing");
        }
        const T* p = std::get_if<T>(&it->second);
        if (!p) {
            throw std::runtime_error(
                "GGUFKVBag: key '" + key + "' expected " + tname<T>() +
                ", got " + vname(it->second));
        }
        return *p;
    }

    // Reference form — used by the array accessors so a read is not a copy.
    template<typename T>
    const T& extract_ref(const std::string& key) const {
        auto it = values_.find(key);
        if (it == values_.end()) {
            throw std::runtime_error("GGUFKVBag: key '" + key + "' missing");
        }
        const T* p = std::get_if<T>(&it->second);
        if (!p) {
            throw std::runtime_error(
                "GGUFKVBag: key '" + key + "' expected " + tname<T>() +
                ", got " + vname(it->second));
        }
        return *p;
    }

    template<typename T>
    std::optional<T> extract_opt(const std::string& key) const {
        auto it = values_.find(key);
        if (it == values_.end()) return std::nullopt;
        const T* p = std::get_if<T>(&it->second);
        if (!p) {
            throw std::runtime_error(
                "GGUFKVBag: key '" + key + "' expected " + tname<T>() +
                ", got " + vname(it->second));
        }
        return *p;
    }
};
