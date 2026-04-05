// test_parallel_grammar.cpp
// Same as test_parallel_order_management.cpp with grammar-constrained sampling.
//
// Usage:
//   Without grammar (identical to original):
//     QWEN_MODEL_PATH=./model.gguf ./test_parallel_grammar
//
//   With grammar:
//     QWEN_MODEL_PATH=./model.gguf GRAMMAR_FILE=./order-management.gbnf ./test_parallel_grammar
//
// When GRAMMAR_FILE is set:
//   - Loads the GBNF and creates one GrammarVocab instance per slot
//   - Masks logits at every decode step using get_valid_tokens(vocab)
//   - EOS is only unmasked when grammar is in accepting state
//   - Vocab is normalized: Qwen3's Ġ (U+0120) → ' ', Ċ (U+010A) → '\n'

#include "qwen3-core/qwen3-model.h"
#include "qwen3-core/gguf-loader.h"
#include "qwen3-core/forward-pass-factory.h"
#include "qwen3-core/tokenizer.h"
#include "sampling/sampling.h"
#include "sampling/vocab_utils.h"
#include "grammar_vocab.h"
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <map>
#include <regex>
#include <chrono>
#include <limits>
#include <algorithm>

static std::string get_model_path() {
    const char* path = std::getenv("QWEN_MODEL_PATH");
    return path ? std::string(path) : "";
}

// ============================================================
// SYSTEM PROMPT  (identical to original)
// ============================================================

const std::string SYSTEM_PROMPT = R"(You are a code generator for an Order Management System.
You output ONLY valid JavaScript. No explanation, no comments, no markdown.

=== ENTITIES ===

Customer: { id, name, tier, region, status }
  - tier: "standard" | "gold" | "platinum"
  - status: "active" | "suspended"

Order: { id, customerId, status, total, itemCount, createdAt }
  - status: "pending" | "confirmed" | "shipped" | "delivered" | "cancelled"

LineItem: { id, orderId, productId, sku, quantity, unitPrice }

Product: { id, sku, name, category, stock, price }

Shipment: { id, orderId, carrier, status, trackingNumber }
  - status: "preparing" | "in_transit" | "delivered" | "failed"

Payment: { id, orderId, status, amount, method }
  - status: "pending" | "completed" | "failed" | "refunded"

=== QUERY FUNCTIONS ===

getCustomers({ tier?, region?, status? }) → Customer[]
getOrders({ customerId?, status?, minTotal?, maxTotal? }) → Order[]
getLineItems({ orderId }) → LineItem[]
getProduct({ productId }) → Product
getProducts({ category?, inStock? }) → Product[]
getShipment({ orderId }) → Shipment
getPayments({ orderId?, status? }) → Payment[]

=== ACTION FUNCTIONS ===

updateOrderStatus({ orderId, status }) → void
cancelOrder({ orderId, reason }) → void
addOrderNote({ orderId, note }) → void
flagOrder({ orderId, reason }) → void

createShipment({ orderId, carrier }) → Shipment
updateShipment({ shipmentId, status, trackingNumber? }) → void

processPayment({ orderId, amount, method }) → Payment
refundPayment({ paymentId, reason }) → void

updateStock({ productId, quantity }) → void
assignToRep({ orderId, repId }) → void

=== ALLOWED SYNTAX ===

Variable declaration:
  const x = value;
  const x = await fn({ });

Loops:
  for (const x of array) { }

Conditionals:
  if (condition) { }
  if (condition) { } else { }
  if (condition) { } else if (condition) { } else { }

Parallel execution:
  await Promise.all(array.map(async (x) => { }));

Array operations:
  array.length
  array.filter(x => condition)
  array.map(x => value)
  array.find(x => condition)

Comparisons:
  ===  !==  >  <  >=  <=

Logic:
  &&  ||  !

=== RULES ===

1. Always use await with function calls
2. Always use const for variables
3. Function arguments must be strings, numbers, or variables
4. No console.log
5. No comments
6. No try/catch
7. No let or var
8. No string concatenation or templates
9. No arithmetic except comparisons
10. Output only code, nothing else

=== EXAMPLES ===

User: get all gold tier customers
Output:
const customers = await getCustomers({ tier: "gold" });

User: cancel all pending orders for suspended customers
Output:
const customers = await getCustomers({ status: "suspended" });
for (const customer of customers) {
  const orders = await getOrders({ customerId: customer.id, status: "pending" });
  for (const order of orders) {
    await cancelOrder({ orderId: order.id, reason: "Customer suspended" });
  }
}

User: for each order over $1000, if payment is pending, flag it for review
Output:
const orders = await getOrders({ minTotal: 1000 });
for (const order of orders) {
  const payments = await getPayments({ orderId: order.id, status: "pending" });
  if (payments.length > 0) {
    await flagOrder({ orderId: order.id, reason: "High value with pending payment" });
  }
}
)";

// ============================================================
// TEST CASE STRUCTURE  (identical to original)
// ============================================================

struct TestCase {
    std::string name;
    std::string prompt;
    std::vector<std::string> expected_functions;
    std::map<std::string, std::vector<std::string>> required_param_keys;
    std::map<std::string, std::map<std::string, std::string>> spot_check_values;
    std::vector<std::string> required_constructs;
    int max_tokens;
};

struct TestResult {
    std::string name;
    bool passed;
    std::vector<std::string> failures;
    std::string generated_code;
};

// ============================================================
// VALIDATION FUNCTIONS  (identical to original)
// ============================================================

bool contains_function_call(const std::string& code, const std::string& fn_name) {
    std::regex fn_regex(fn_name + "\\s*\\(");
    return std::regex_search(code, fn_regex);
}

bool contains_param_key(const std::string& code, const std::string& fn_name, const std::string& key) {
    std::regex pattern1(fn_name + "\\s*\\(\\s*\\{[^}]*" + key + "\\s*:");
    if (std::regex_search(code, pattern1)) return true;
    std::regex pattern2(fn_name + "\\s*\\(\\s*\\{[^}]*\\b" + key + "\\b[^:}]*\\}");
    return std::regex_search(code, pattern2);
}

bool contains_param_value(const std::string& code, const std::string& fn_name,
                          const std::string& key, const std::string& value) {
    std::regex pattern(fn_name + "\\s*\\(\\s*\\{[^}]*" + key + "\\s*:\\s*\"?" + value + "\"?");
    return std::regex_search(code, pattern);
}

bool contains_construct(const std::string& code, const std::string& construct) {
    if (construct == "for")
        return code.find("for (") != std::string::npos || code.find("for(") != std::string::npos;
    if (construct == "if")
        return code.find("if (") != std::string::npos || code.find("if(") != std::string::npos;
    if (construct == "await")
        return code.find("await ") != std::string::npos;
    if (construct == "Promise.all")
        return code.find("Promise.all") != std::string::npos;
    return code.find(construct) != std::string::npos;
}

std::string normalize_output(const std::string& output) {
    std::string normalized = output;

    // Qwen3 space marker Ġ (U+0120, UTF-8: C4 A0)
    size_t pos = 0;
    const std::string g_char = "\xC4\xA0";
    while ((pos = normalized.find(g_char, pos)) != std::string::npos) {
        normalized.replace(pos, g_char.length(), " ");
        pos += 1;
    }

    // Qwen3 newline marker Ċ (UTF-8: C4 8A)
    const std::string c_char = "\xC4\x8A";
    pos = 0;
    while ((pos = normalized.find(c_char, pos)) != std::string::npos) {
        normalized.replace(pos, c_char.length(), "\n");
        pos += 1;
    }

    for (const auto& tok : {"<|im_end|>", "<|endoftext|>", "</s>"}) {
        pos = normalized.find(tok);
        if (pos != std::string::npos) normalized = normalized.substr(0, pos);
    }

    return normalized;
}

TestResult validate_output(const TestCase& tc, const std::string& raw_output) {
    TestResult result;
    result.name   = tc.name;
    result.passed = true;

    std::string output = normalize_output(raw_output);
    result.generated_code = output;

    for (const auto& fn : tc.expected_functions)
        if (!contains_function_call(output, fn)) {
            result.passed = false;
            result.failures.push_back("Missing function call: " + fn);
        }

    for (const auto& [fn, keys] : tc.required_param_keys)
        for (const auto& key : keys)
            if (!contains_param_key(output, fn, key)) {
                result.passed = false;
                result.failures.push_back("Missing param key '" + key + "' in " + fn);
            }

    for (const auto& [fn, kv] : tc.spot_check_values)
        for (const auto& [key, val] : kv)
            if (!contains_param_value(output, fn, key, val)) {
                result.passed = false;
                result.failures.push_back("Missing value " + key + ": " + val + " in " + fn);
            }

    for (const auto& c : tc.required_constructs)
        if (!contains_construct(output, c)) {
            result.passed = false;
            result.failures.push_back("Missing construct: " + c);
        }

    return result;
}

// ============================================================
// TEST CASES  (identical to original)
// ============================================================

std::vector<TestCase> get_test_cases() {
    return {
        {
            "01_simple_query",
            "Get all products in the electronics category",
            {"getProducts"},
            {{"getProducts", {"category"}}},
            {{"getProducts", {{"category", "electronics"}}}},
            {"await"}, 256
        },
        {
            "02_loop_with_action",
            "Update all orders with status 'shipped' to 'delivered'",
            {"getOrders", "updateOrderStatus"},
            {{"getOrders", {"status"}}, {"updateOrderStatus", {"orderId", "status"}}},
            {{"getOrders", {{"status", "shipped"}}}, {"updateOrderStatus", {{"status", "delivered"}}}},
            {"await", "for"}, 512
        },
        {
            "03_nested_loop",
            "For each platinum customer, get their pending orders and add a note saying 'Priority customer'",
            {"getCustomers", "getOrders", "addOrderNote"},
            {{"getCustomers", {"tier"}}, {"getOrders", {"customerId", "status"}}, {"addOrderNote", {"orderId", "note"}}},
            {{"getCustomers", {{"tier", "platinum"}}}, {"getOrders", {{"status", "pending"}}}},
            {"await", "for"}, 512
        },
        {
            "04_conditional_multi_action",
            "Find orders over $500 with failed payments. Refund the payment and cancel the order with reason 'Payment failed'",
            {"getOrders", "getPayments", "refundPayment", "cancelOrder"},
            {{"getOrders", {"minTotal"}}, {"getPayments", {"orderId", "status"}},
             {"refundPayment", {"paymentId", "reason"}}, {"cancelOrder", {"orderId", "reason"}}},
            {{"getPayments", {{"status", "failed"}}}, {"getOrders", {{"minTotal", "500"}}}},
            {"await", "for", "if"}, 512
        },
        {
            "05_nested_condition",
            "For each order in 'confirmed' status, check if all products in its line items are in stock. If any product has stock < quantity ordered, flag the order with reason 'Insufficient stock'",
            {"getOrders", "getLineItems", "getProduct", "flagOrder"},
            {{"getOrders", {"status"}}, {"getLineItems", {"orderId"}},
             {"getProduct", {"productId"}}, {"flagOrder", {"orderId", "reason"}}},
            {{"getOrders", {{"status", "confirmed"}}}},
            {"await", "for", "if"}, 768
        },
        {
            "06_aggregation",
            "For customers in the EU region, get all completed orders and calculate total revenue. If total > $10000, flag all pending orders for that customer with reason 'High-value customer'",
            {"getCustomers", "getOrders", "flagOrder"},
            {{"getCustomers", {"region"}}, {"getOrders", {"customerId", "status"}},
             {"flagOrder", {"orderId", "reason"}}},
            {{"getCustomers", {{"region", "EU"}}}},
            {"await", "for", "if"}, 768
        },
        {
            "07_parallel_execution",
            "For the given order, update stock for all line items in parallel using Promise.all",
            {"getLineItems", "getProduct", "updateStock"},
            {{"getLineItems", {"orderId"}}, {"getProduct", {"productId"}},
             {"updateStock", {"productId", "quantity"}}},
            {},
            {"await", "Promise.all"}, 512
        },
        {
            "08_null_handling",
            "For each confirmed order, get its shipment. If no shipment exists, create one with carrier 'default'. Then update order status to 'shipped'",
            {"getOrders", "getShipment", "createShipment", "updateOrderStatus"},
            {{"getOrders", {"status"}}, {"getShipment", {"orderId"}},
             {"createShipment", {"orderId", "carrier"}}, {"updateOrderStatus", {"orderId", "status"}}},
            {{"getOrders", {{"status", "confirmed"}}}, {"createShipment", {{"carrier", "default"}}},
             {"updateOrderStatus", {{"status", "shipped"}}}},
            {"await", "for", "if"}, 512
        },
        {
            "09_multi_filter_chain",
            "Find all gold tier customers in EU region with pending orders over $200. For each matching order, process payment with method 'express', then update order to confirmed, then create shipment with carrier 'premium'",
            {"getCustomers", "getOrders", "processPayment", "updateOrderStatus", "createShipment"},
            {{"getCustomers", {"tier", "region"}}, {"getOrders", {"customerId", "status", "minTotal"}},
             {"processPayment", {"orderId", "method"}}, {"updateOrderStatus", {"orderId", "status"}},
             {"createShipment", {"orderId", "carrier"}}},
            {{"getCustomers", {{"tier", "gold"}, {"region", "EU"}}},
             {"getOrders", {{"status", "pending"}}}, {"processPayment", {{"method", "express"}}},
             {"createShipment", {{"carrier", "premium"}}}},
            {"await", "for"}, 512
        }
    };
}

// ============================================================
// SLOT STATE
// ============================================================

struct SlotState {
    uint32_t slot_id;
    int      test_idx;
    int      position;
    int      max_tokens;
    int      tokens_generated;
    std::vector<int32_t> all_tokens;
    std::string generated_text;
    bool    finished;
    int32_t last_token_id;

    // Grammar state — nullptr when grammar is disabled
    std::unique_ptr<qwen3::GrammarVocab> grammar;
};

// ============================================================
// GRAMMAR HELPERS
// ============================================================

// Normalize Qwen3 tokenizer markers so the grammar can match plain ASCII.
// Ġ (U+0120, C4 A0) → ' '   (space-prefixed token)
// Ċ (U+010A, C4 8A) → '\n'  (newline token)
static std::string normalize_token(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ) {
        if (i + 1 < s.size()
            && (unsigned char)s[i] == 0xC4
            && (unsigned char)s[i+1] == 0xA0) {
            out += ' ';  i += 2;
        } else if (i + 1 < s.size()
                   && (unsigned char)s[i] == 0xC4
                   && (unsigned char)s[i+1] == 0x8A) {
            out += '\n'; i += 2;
        } else {
            out += s[i++];
        }
    }
    return out;
}

// Mask logits: zero out anything not in the grammar's valid set.
// EOS is allowed only when the grammar is in accepting state.
static void mask_grammar_logits(
    std::vector<float>&           logits,
    qwen3::GrammarVocab*          grammar,
    const std::vector<std::string>& vocab,
    int32_t                       eos_id)
{
    auto valid = grammar->get_valid_tokens(vocab);
    std::set<int32_t> vs(valid.begin(), valid.end());
    if (grammar->is_accepting_state())
        vs.insert(eos_id);

    for (size_t i = 0; i < logits.size(); ++i)
        if (!vs.count((int32_t)i))
            logits[i] = -std::numeric_limits<float>::infinity();
}

// ============================================================
// PARALLEL MODEL RUNNER
// ============================================================

class ParallelModelRunner {
    std::shared_ptr<Qwen3Model>   model_;
    std::unique_ptr<ForwardPassBase> forward_pass_;
    std::unique_ptr<Tokenizer>    tokenizer_;
    std::string                   model_path_;
    std::vector<int32_t>          system_prompt_tokens_;
    uint32_t                      system_prompt_cache_pos_ = 0;
    uint32_t                      max_batch_size_;

    void precompute_system_prompt() {
        std::string sp = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n";
        system_prompt_tokens_ = tokenizer_->encode(sp);

        ggml_backend_sched_t scheduler = model_->get_scheduler();
        ggml_backend_sched_reset(scheduler);
        ggml_cgraph* gf = forward_pass_->build_prefill_graph(system_prompt_tokens_, 0);
        ggml_backend_sched_alloc_graph(scheduler, gf);
        forward_pass_->set_inputs(gf, system_prompt_tokens_, 0);
        ggml_backend_sched_graph_compute(scheduler, gf);
        forward_pass_->advance_cache(system_prompt_tokens_.size(), 0);
        system_prompt_cache_pos_ = forward_pass_->get_cache_pos(0);

        std::cout << "  System prompt tokens: " << system_prompt_tokens_.size() << "\n";
        std::cout << "  System prompt cache pos: " << system_prompt_cache_pos_ << "\n";
    }

public:
    explicit ParallelModelRunner(const std::string& path, uint32_t batch)
        : model_path_(path), max_batch_size_(batch) {}

    bool load() {
        try {
            model_ = std::make_shared<Qwen3Model>();
            model_->load_metadata(model_path_);
            model_->load_tensors();
            tokenizer_ = std::make_unique<Tokenizer>(&model_->get_metadata());
            forward_pass_ = create_forward_pass(*model_, &model_->get_metadata(), 2048, max_batch_size_);
            precompute_system_prompt();
            std::cout << "  Loaded: " << model_path_ << "\n";
            std::cout << "  Max batch size: " << max_batch_size_ << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "  Failed to load: " << e.what() << "\n";
            return false;
        }
    }

    std::vector<std::string> generate_parallel(const std::vector<TestCase>& test_cases) {
        if (!model_ || !forward_pass_ || !tokenizer_)
            throw std::runtime_error("Model not loaded");

        ggml_log_set([](ggml_log_level level, const char* text, void* user_data) {
            if (level == GGML_LOG_LEVEL_ERROR) fprintf(stderr, "%s", text);
            (void)user_data;
        }, nullptr);

        const size_t n_tests   = test_cases.size();
        const size_t vocab_size = model_->get_metadata().vocab_size;
        const int32_t eos_id   = model_->get_metadata().eos_token_id;

        if (n_tests > max_batch_size_)
            throw std::runtime_error("Too many tests for batch size");

        // ── Grammar setup ──────────────────────────────────────────────────
        std::string gbnf_str;
        bool grammar_enabled = false;

        const char* grammar_file = std::getenv("GRAMMAR_FILE");
        if (grammar_file) {
            std::ifstream f(grammar_file);
            if (!f.is_open()) {
                std::cerr << "  WARNING: Cannot open GRAMMAR_FILE=" << grammar_file << "\n";
            } else {
                gbnf_str = std::string(std::istreambuf_iterator<char>(f), {});
                grammar_enabled = true;
                std::cout << "  Grammar: " << grammar_file << " (" << gbnf_str.size() << " bytes)\n";
            }
        } else {
            std::cout << "  Grammar: disabled (set GRAMMAR_FILE to enable)\n";
        }

        // Build normalized vocab for grammar (only when needed)
        std::vector<std::string> grammar_vocab;
        if (grammar_enabled) {
            grammar_vocab.resize(vocab_size);
            for (size_t i = 0; i < vocab_size; ++i)
                grammar_vocab[i] = normalize_token(tokenizer_->decode((int32_t)i));
            std::cout << "  Grammar vocab built: " << vocab_size << " tokens\n";
        }

        qwen3::GreedySampler sampler;
        ggml_backend_sched_t scheduler = model_->get_scheduler();

        // ── Phase 1: Clone system-prompt prefix to all slots ───────────────
        std::cout << "\n  [Phase 1] Cloning prefix to " << n_tests << " slots...\n";
        for (size_t i = 1; i < n_tests; ++i)
            forward_pass_->clone_slot(0, i, system_prompt_cache_pos_);

        // ── Phase 2: Prefill user prompts + first token ────────────────────
        std::cout << "  [Phase 2] Prefilling user prompts...\n";

        std::vector<SlotState> slots(n_tests);

        for (size_t i = 0; i < n_tests; ++i) {
            const auto& tc = test_cases[i];

            std::string user_prompt =
                "<|im_start|>user\n" + tc.prompt + "<|im_end|>\n<|im_start|>assistant\n";
            std::vector<int32_t> user_tokens = tokenizer_->encode(user_prompt);

            slots[i].slot_id          = (uint32_t)i;
            slots[i].test_idx         = (int)i;
            slots[i].max_tokens       = tc.max_tokens;
            slots[i].tokens_generated = 0;
            slots[i].finished         = false;
            slots[i].generated_text   = "";
            slots[i].all_tokens       = system_prompt_tokens_;
            slots[i].all_tokens.insert(slots[i].all_tokens.end(),
                                       user_tokens.begin(), user_tokens.end());

            // Initialize per-slot grammar (independent instance per slot)
            if (grammar_enabled)
                slots[i].grammar = qwen3::GrammarVocab::parse_impl(gbnf_str);

            // Prefill user prompt
            ggml_backend_sched_reset(scheduler);
            ggml_cgraph* gf = forward_pass_->build_prefill_graph(
                user_tokens, system_prompt_cache_pos_, i);
            ggml_backend_sched_alloc_graph(scheduler, gf);
            forward_pass_->set_inputs(gf, user_tokens, system_prompt_cache_pos_);
            ggml_backend_sched_graph_compute(scheduler, gf);
            forward_pass_->advance_cache(user_tokens.size(), i);

            // Get first token
            std::vector<float> logits = forward_pass_->get_output_logits(gf);
            std::vector<float> last_logits(logits.end() - (int)vocab_size, logits.end());

            if (grammar_enabled && slots[i].grammar)
                mask_grammar_logits(last_logits, slots[i].grammar.get(), grammar_vocab, eos_id);

            int32_t first_token = sampler.sample(last_logits, slots[i].all_tokens);

            slots[i].last_token_id    = first_token;
            slots[i].all_tokens.push_back(first_token);
            slots[i].generated_text   = tokenizer_->decode(first_token);
            slots[i].tokens_generated = 1;
            slots[i].position         = forward_pass_->get_cache_pos(i);

            if (first_token == eos_id)
                slots[i].finished = true;

            if (grammar_enabled && slots[i].grammar && first_token != eos_id)
                slots[i].grammar->accept_token(first_token, grammar_vocab);

            std::cout << "    Slot " << i << ": prefilled " << user_tokens.size()
                      << " tokens, pos=" << slots[i].position << "\n";
        }

        // ── Phase 3: Batched decode loop ───────────────────────────────────
        std::cout << "  [Phase 3] Batched decoding...\n";

        const int max_iter = 768;
        for (int iter = 0; iter < max_iter; ++iter) {
            // Collect active slots
            std::vector<int32_t>  batch_tokens;
            std::vector<uint32_t> batch_slots;
            std::vector<int32_t>  batch_positions;
            std::vector<size_t>   active;

            for (size_t i = 0; i < n_tests; ++i) {
                if (!slots[i].finished) {
                    batch_tokens.push_back(slots[i].last_token_id);
                    batch_slots.push_back(slots[i].slot_id);
                    batch_positions.push_back(slots[i].position);
                    active.push_back(i);
                }
            }

            if (batch_tokens.empty()) {
                std::cout << "    All slots finished at iteration " << iter << "\n";
                break;
            }

            // Build and run batched decode
            ggml_backend_sched_reset(scheduler);
            ggml_cgraph* gf = forward_pass_->build_decoding_graph(
                batch_tokens, batch_slots, batch_positions);
            ggml_backend_sched_alloc_graph(scheduler, gf);
            forward_pass_->set_batched_inputs(gf, batch_tokens, batch_slots, batch_positions);
            ggml_backend_sched_graph_compute(scheduler, gf);

            // Sample for each active slot
            for (size_t b = 0; b < active.size(); ++b) {
                SlotState& slot = slots[active[b]];

                std::vector<float> slot_logits =
                    forward_pass_->get_output_logits_for_slot(gf, b);

                // Grammar masking
                if (grammar_enabled && slot.grammar)
                    mask_grammar_logits(slot_logits, slot.grammar.get(), grammar_vocab, eos_id);

                int32_t next = sampler.sample(slot_logits, slot.all_tokens);

                slot.last_token_id     = next;
                slot.all_tokens.push_back(next);
                slot.generated_text   += tokenizer_->decode(next);
                slot.tokens_generated++;
                slot.position++;

                forward_pass_->advance_cache(1, slot.slot_id);

                // Accept token in grammar
                if (grammar_enabled && slot.grammar && next != eos_id)
                    slot.grammar->accept_token(next, grammar_vocab);

                if (next == eos_id || slot.tokens_generated >= slot.max_tokens)
                    slot.finished = true;
            }

            if ((iter + 1) % 50 == 0) {
                int active_count = 0;
                for (const auto& s : slots) if (!s.finished) active_count++;
                std::cout << "    Iteration " << (iter+1)
                          << ", active slots: " << active_count << "\n";
            }
        }

        // Collect results
        std::vector<std::string> results(n_tests);
        for (size_t i = 0; i < n_tests; ++i)
            results[i] = slots[i].generated_text;
        return results;
    }
};

// ============================================================
// MAIN  (identical to original except for test binary name)
// ============================================================

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;

    std::cout << "========================================\n";
    std::cout << "PARALLEL Order Management DSL Tests (with optional grammar)\n";
    std::cout << "========================================\n";

    const std::string model_path = get_model_path();
    auto test_cases  = get_test_cases();
    uint32_t batch   = (uint32_t)test_cases.size();

    std::cout << "\n[Loading Model]\n";
    ParallelModelRunner runner(model_path, batch);
    if (!runner.load()) {
        std::cerr << "Failed to load model. Exiting.\n";
        return 1;
    }

    std::cout << "\n[Running Parallel Generation]\n";
    std::cout << "----------------------------------------\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::string> outputs;
    try {
        outputs = runner.generate_parallel(test_cases);
    } catch (const std::exception& e) {
        std::cerr << "Generation failed: " << e.what() << "\n";
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "\n[Generation completed in " << ms << " ms]\n";
    std::cout << "\n[Validating Results]\n";
    std::cout << "----------------------------------------\n";

    int passed = 0, failed = 0;
    std::vector<TestResult> results;

    for (size_t i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        std::cout << "\nTest: " << tc.name << "\n";

        TestResult r = validate_output(tc, outputs[i]);
        results.push_back(r);

        if (r.passed) { std::cout << "Result: PASSED ✓\n"; passed++; }
        else {
            std::cout << "Result: FAILED ✗\n";
            for (const auto& f : r.failures)
                std::cout << "  - " << f << "\n";
            failed++;
        }

        std::cout << "Generated:\n";
        std::string disp = r.generated_code.substr(0, 500);
        if (r.generated_code.size() > 500) disp += "...";
        std::cout << disp << "\n";
        std::cout << "----------------------------------------\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "Total:  " << test_cases.size() << "\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";
    std::cout << "Score:  " << (100 * passed / (int)test_cases.size()) << "%\n";
    std::cout << "Time:   " << ms << " ms\n";
    double throughput = test_cases.size() * 1000.0 / (double)ms;
    std::cout << "Throughput: " << throughput << " tests/sec\n";

    if (failed > 0) {
        std::cout << "\n[Failed Tests Detail]\n";
        for (const auto& r : results)
            if (!r.passed) {
                std::cout << "\n" << r.name << ":\n";
                for (const auto& f : r.failures)
                    std::cout << "  - " << f << "\n";
            }
    }

    std::cout << "\n[Performance Baseline]\n";
    const double min_throughput = 0.12;
    if (throughput < min_throughput) {
        std::cerr << "FAIL: Throughput regression. Expected >= " << min_throughput
                  << " tests/sec, got " << throughput << " tests/sec\n";
        return 1;
    }
    std::cout << "PASS: Throughput " << throughput << " >= " << min_throughput << " tests/sec\n";

    return failed > 0 ? 1 : 0;
}
