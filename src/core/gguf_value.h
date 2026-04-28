#pragma once

// gguf_value.h — generic GGUF scalar KV bag
//
// GGUFValue is a tagged union over the scalar types that appear in
// family-specific GGUF metadata keys.  Array types (tokenizer vocab, scores,
// merges) are deliberately excluded — they stay on ModelMetadata's typed
// members.  Add array support here only when a real consumer needs it.
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

// Tagged union over the five scalar GGUF types we actually encounter in
// family-specific metadata keys.
using GGUFValue = std::variant<uint32_t, int32_t, float, bool, std::string>;

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
