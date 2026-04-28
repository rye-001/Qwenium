#pragma once
// model_registry.h — architecture → ForwardPass factory + tokenizer config
// + chat template dispatch.
//
// Each model recipe registers itself once at startup.  CLI dispatch becomes
// single-lookup calls into this registry — no architecture string literals
// outside src/models/.
//
// Fail-loud contract: looking up an unregistered architecture throws
// std::runtime_error naming the function, the expected list, and the actual
// architecture string.

#include "forward_pass_base.h"
#include "../loader/tokenizer_config.h"
#include "../loader/chat_template.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

using ForwardPassFactory = std::function<std::unique_ptr<ForwardPassBase>(
    const Model&        model,
    const ModelMetadata*     metadata,
    uint32_t                 context_len,
    uint32_t                 max_batch_size,
    int                      kv_quant_bits)>;

// Per-architecture tensor-inventory validator.
// Throws std::runtime_error (naming slot, expected, actual) on failure.
using InventoryValidator = std::function<void(const ModelMetadata&)>;

// Register a factory + inventory validator + tokenizer config + chat template
// for an architecture string.  Overwrites if already present.
//
// tokenizer_config and chat_template default to {} / nullptr so existing
// three-argument call sites (e.g. unit-test fake registrations) keep working.
void register_model(const std::string& architecture,
                    ForwardPassFactory  factory,
                    InventoryValidator  validator,
                    TokenizerConfig     tokenizer_config = {},
                    std::unique_ptr<ChatTemplate> chat_template = nullptr);

// Unregister a factory (used in tests to clean up after registering fakes).
void unregister_model(const std::string& architecture);

// Returns true iff `architecture` is registered.
bool is_architecture_registered(const std::string& architecture);

// List currently registered architecture strings (for diagnostics).
std::vector<std::string> registered_architectures();

// Return the registered InventoryValidator for `architecture`.
// Throws std::runtime_error if the architecture is not registered.
InventoryValidator lookup_inventory_validator(const std::string& architecture);

// Return the TokenizerConfig registered for `architecture`.
// Throws std::runtime_error if the architecture is not registered.
TokenizerConfig lookup_tokenizer_config(const std::string& architecture);

// Return a non-owning pointer to the ChatTemplate registered for
// `architecture`.  Returns nullptr if the architecture was registered without
// a template (e.g. fake test registrations).
// Throws std::runtime_error if the architecture is not registered at all.
const ChatTemplate* lookup_chat_template(const std::string& architecture);

// Create a forward pass for the architecture in `metadata`. Throws
// std::runtime_error if the architecture is not registered.
std::unique_ptr<ForwardPassBase> create_forward_pass(
    const Model&    model,
    const ModelMetadata* metadata,
    uint32_t             context_len,
    uint32_t             max_batch_size,
    int                  kv_quant_bits);

// Register all built-in model recipes (qwen2/3, qwen35, qwen35moe, gemma).
// Idempotent — safe to call multiple times. Call once from CLI startup.
void register_builtin_models();
