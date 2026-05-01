#include "model_registry.h"

#include "qwen3.h"
#include "qwen35.h"
#include "qwen36.h"
#include "gemma1.h"
#include "gemma2.h"
#include "gemma3.h"
#include "gemma4.h"

#include <map>
#include <mutex>
#include <stdexcept>

namespace {

struct ModelEntry {
    ForwardPassFactory              factory;
    InventoryValidator              validator;
    TokenizerConfig                 tokenizer_config;
    std::unique_ptr<ChatTemplate>   chat_template;
};

std::map<std::string, ModelEntry>& registry()
{
    static std::map<std::string, ModelEntry> r;
    return r;
}

std::mutex& registry_mu()
{
    static std::mutex m;
    return m;
}
} // namespace

void register_model(const std::string& architecture,
                    ForwardPassFactory  factory,
                    InventoryValidator  validator,
                    TokenizerConfig     tokenizer_config,
                    std::unique_ptr<ChatTemplate> chat_template)
{
    std::lock_guard<std::mutex> g(registry_mu());
    registry()[architecture] = ModelEntry{
        std::move(factory),
        std::move(validator),
        std::move(tokenizer_config),
        std::move(chat_template),
    };
}

void unregister_model(const std::string& architecture)
{
    std::lock_guard<std::mutex> g(registry_mu());
    registry().erase(architecture);
}

bool is_architecture_registered(const std::string& architecture)
{
    std::lock_guard<std::mutex> g(registry_mu());
    return registry().count(architecture) > 0;
}

std::vector<std::string> registered_architectures()
{
    std::lock_guard<std::mutex> g(registry_mu());
    std::vector<std::string> out;
    out.reserve(registry().size());
    for (const auto& [k, _] : registry()) out.push_back(k);
    return out;
}

InventoryValidator lookup_inventory_validator(const std::string& architecture)
{
    std::lock_guard<std::mutex> g(registry_mu());
    auto it = registry().find(architecture);
    if (it == registry().end()) {
        std::string registered;
        for (const auto& [k, _] : registry()) {
            if (!registered.empty()) registered += ", ";
            registered += "'" + k + "'";
        }
        throw std::runtime_error(
            "lookup_inventory_validator: expected one of: " + registered +
            ", got '" + architecture + "'");
    }
    return it->second.validator;
}

TokenizerConfig lookup_tokenizer_config(const std::string& architecture)
{
    std::lock_guard<std::mutex> g(registry_mu());
    auto it = registry().find(architecture);
    if (it == registry().end()) {
        std::string registered;
        for (const auto& [k, _] : registry()) {
            if (!registered.empty()) registered += ", ";
            registered += "'" + k + "'";
        }
        throw std::runtime_error(
            "lookup_tokenizer_config: expected one of: " + registered +
            ", got '" + architecture + "'");
    }
    return it->second.tokenizer_config;
}

const ChatTemplate* lookup_chat_template(const std::string& architecture)
{
    std::lock_guard<std::mutex> g(registry_mu());
    auto it = registry().find(architecture);
    if (it == registry().end()) {
        std::string registered;
        for (const auto& [k, _] : registry()) {
            if (!registered.empty()) registered += ", ";
            registered += "'" + k + "'";
        }
        throw std::runtime_error(
            "lookup_chat_template: expected one of: " + registered +
            ", got '" + architecture + "'");
    }
    return it->second.chat_template.get();
}

std::unique_ptr<ForwardPassBase> create_forward_pass(
    const Model&    model,
    const ModelMetadata* metadata,
    uint32_t             context_len,
    uint32_t             max_batch_size,
    int                  kv_quant_bits)
{
    ForwardPassFactory factory;
    {
        std::lock_guard<std::mutex> g(registry_mu());
        auto it = registry().find(metadata->architecture);
        if (it == registry().end()) {
            std::string registered;
            for (const auto& [k, _] : registry()) {
                if (!registered.empty()) registered += ", ";
                registered += "'" + k + "'";
            }
            throw std::runtime_error(
                "create_forward_pass: expected one of: " + registered +
                ", got '" + metadata->architecture + "'");
        }
        factory = it->second.factory;
    }
    return factory(model, metadata, context_len, max_batch_size, kv_quant_bits);
}

void register_builtin_models()
{
    auto qwen3_factory = [](const Model& m, const ModelMetadata* meta,
                            uint32_t ctx, uint32_t bs, int kvb) {
        return std::make_unique<Qwen3ForwardPass>(m, meta, ctx, bs, kvb);
    };
    register_model("qwen2", qwen3_factory, validate_qwen3_inventory,
                   qwen_tokenizer_config(), std::make_unique<QwenChatTemplate>());
    register_model("qwen3", qwen3_factory, validate_qwen3_inventory,
                   qwen_tokenizer_config(), std::make_unique<QwenChatTemplate>());

    register_model("qwen35",
        [](const Model& m, const ModelMetadata* meta, uint32_t ctx, uint32_t bs, int kvb) {
            return std::make_unique<Qwen35ForwardPass>(m, meta, ctx, bs, kvb);
        },
        validate_qwen35_inventory,
        qwen_tokenizer_config(), std::make_unique<QwenChatTemplate>());

    register_model("qwen35moe",
        [](const Model& m, const ModelMetadata* meta, uint32_t ctx, uint32_t bs, int kvb) {
            return std::make_unique<Qwen36ForwardPass>(m, meta, ctx, bs, kvb);
        },
        validate_qwen36_inventory,
        qwen_tokenizer_config(), std::make_unique<QwenChatTemplate>());

    register_model("gemma",
        [](const Model& m, const ModelMetadata* meta, uint32_t ctx, uint32_t bs, int kvb) {
            return std::make_unique<Gemma1ForwardPass>(m, meta, ctx, bs, kvb);
        },
        validate_gemma1_inventory,
        gemma1_tokenizer_config(), std::make_unique<GemmaChatTemplate>());

    register_model("gemma2",
        [](const Model& m, const ModelMetadata* meta, uint32_t ctx, uint32_t bs, int kvb) {
            return std::make_unique<Gemma2ForwardPass>(m, meta, ctx, bs, kvb);
        },
        validate_gemma2_inventory,
        gemma1_tokenizer_config(), std::make_unique<GemmaChatTemplate>());

    register_model("gemma3",
        [](const Model& m, const ModelMetadata* meta, uint32_t ctx, uint32_t bs, int kvb) {
            return std::make_unique<Gemma3ForwardPass>(m, meta, ctx, bs, kvb);
        },
        validate_gemma3_inventory,
        gemma1_tokenizer_config(), std::make_unique<GemmaChatTemplate>());

    // Gemma 4 (26B-A4B-It text-only).
    // Tokenizer config and chat template are wired now so loader-side
    // metadata dispatch can resolve "gemma4".  The forward-pass factory
    // produces a stub that throws on construction; PR G4.8 replaces it.
    // Reuses GemmaChatTemplate (basic <start_of_turn>/<end_of_turn>
    // rendering — the IT model's tool-calling Jinja is out of scope; see
    // docs/plan-gemma-impl.md → Phase G4 → "Out of scope for G4").
    register_model("gemma4",
        [](const Model& m, const ModelMetadata* meta, uint32_t ctx, uint32_t bs, int kvb) {
            return std::make_unique<Gemma4ForwardPass>(m, meta, ctx, bs, kvb);
        },
        validate_gemma4_inventory,
        gemma4_tokenizer_config(),
        // Gemma 4 IT uses native <|turn>...<turn|> markers — not the
        // <start_of_turn>/<end_of_turn> shape G1/2/3 share.  Using
        // GemmaChatTemplate here makes the IT model hallucinate
        // <tool_call|> because the prompt structure doesn't match its
        // training distribution.
        std::make_unique<Gemma4ChatTemplate>());
}
