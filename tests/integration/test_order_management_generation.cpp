#include "qwen3-core/qwen3-model.h"
#include "qwen3-core/gguf-loader.h"
#include "qwen3-core/forward-pass.h"
#include "qwen3-core/tokenizer.h"
#include "sampling/sampling.h"
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <set>
#include <regex>

// ============================================================
// SYSTEM PROMPT
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
// TEST CASE STRUCTURE
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
// VALIDATION FUNCTIONS
// ============================================================

bool contains_function_call(const std::string& code, const std::string& fn_name) {
    std::regex fn_regex(fn_name + "\\s*\\(");
    return std::regex_search(code, fn_regex);
}

bool contains_param_key(const std::string& code, const std::string& fn_name, const std::string& key) {
    // Pattern 1: fnName({ ... key: ... }) - standard syntax
    std::regex pattern1(fn_name + "\\s*\\(\\s*\\{[^}]*" + key + "\\s*:");
    if (std::regex_search(code, pattern1)) {
        return true;
    }
    
    // Pattern 2: fnName({ key }) - ES6 shorthand syntax
    std::regex pattern2(fn_name + "\\s*\\(\\s*\\{[^}]*\\b" + key + "\\b[^:}]*\\}");
    return std::regex_search(code, pattern2);
}

bool contains_param_value(const std::string& code, const std::string& fn_name, 
                          const std::string& key, const std::string& value) {
    // Pattern: fnName({ ... key: "value" ... }) or fnName({ ... key: value ... })
    std::regex pattern(fn_name + "\\s*\\(\\s*\\{[^}]*" + key + "\\s*:\\s*\"?" + value + "\"?");
    return std::regex_search(code, pattern);
}

bool contains_construct(const std::string& code, const std::string& construct) {
    if (construct == "for") {
        return code.find("for (") != std::string::npos || 
               code.find("for(") != std::string::npos;
    } else if (construct == "if") {
        return code.find("if (") != std::string::npos || 
               code.find("if(") != std::string::npos;
    } else if (construct == "await") {
        return code.find("await ") != std::string::npos;
    } else if (construct == "Promise.all") {
        return code.find("Promise.all") != std::string::npos;
    }
    return code.find(construct) != std::string::npos;
}

std::string normalize_output(const std::string& output) {
    std::string normalized = output;
    
    // Replace Ġ (U+0120, UTF-8: C4 A0) with space - common in BPE tokenizers
    std::string g_char = "\xC4\xA0";  // UTF-8 encoding of Ġ
    size_t pos = 0;
    while ((pos = normalized.find(g_char, pos)) != std::string::npos) {
        normalized.replace(pos, g_char.length(), " ");
        pos += 1;
    }
    
    // Replace Ċ (U+010A, UTF-8: C4 8A) with newline - common in BPE tokenizers
    std::string c_char = "\xC4\x8A";  // UTF-8 encoding of Ċ
    pos = 0;
    while ((pos = normalized.find(c_char, pos)) != std::string::npos) {
        normalized.replace(pos, c_char.length(), "\n");
        pos += 1;
    }
    
    // Remove end tokens
    std::vector<std::string> end_tokens = {"<|im_end|>", "<|endoftext|>", "</s>"};
    for (const auto& token : end_tokens) {
        pos = normalized.find(token);
        if (pos != std::string::npos) {
            normalized = normalized.substr(0, pos);
        }
    }
    
    return normalized;
}

TestResult validate_output(const TestCase& tc, const std::string& raw_output) {
    TestResult result;
    result.name = tc.name;
    result.passed = true;
    
    // Normalize the output
    std::string output = normalize_output(raw_output);
    result.generated_code = output;

    // Check expected functions
    for (const auto& fn : tc.expected_functions) {
        if (!contains_function_call(output, fn)) {
            result.passed = false;
            result.failures.push_back("Missing function call: " + fn);
        }
    }

    // Check required param keys
    for (const auto& [fn, keys] : tc.required_param_keys) {
        for (const auto& key : keys) {
            if (!contains_param_key(output, fn, key)) {
                result.passed = false;
                result.failures.push_back("Missing param key '" + key + "' in " + fn);
            }
        }
    }

    // Spot check values
    for (const auto& [fn, kv_pairs] : tc.spot_check_values) {
        for (const auto& [key, value] : kv_pairs) {
            if (!contains_param_value(output, fn, key, value)) {
                result.passed = false;
                result.failures.push_back("Missing value " + key + ": " + value + " in " + fn);
            }
        }
    }

    // Check required constructs
    for (const auto& construct : tc.required_constructs) {
        if (!contains_construct(output, construct)) {
            result.passed = false;
            result.failures.push_back("Missing construct: " + construct);
        }
    }

    return result;
}

// ============================================================
// TEST CASES
// ============================================================

std::vector<TestCase> get_test_cases() {
    return {
        // Prompt 1: Simple query
        {
            "01_simple_query",
            "Get all products in the electronics category",
            {"getProducts"},
            {{"getProducts", {"category"}}},
            {{"getProducts", {{"category", "electronics"}}}},
            {"await"},
            256
        },

        // Prompt 2: Loop with action
        {
            "02_loop_with_action",
            "Update all orders with status 'shipped' to 'delivered'",
            {"getOrders", "updateOrderStatus"},
            {
                {"getOrders", {"status"}},
                {"updateOrderStatus", {"orderId", "status"}}
            },
            {
                {"getOrders", {{"status", "shipped"}}},
                {"updateOrderStatus", {{"status", "delivered"}}}
            },
            {"await", "for"},
            512
        },

        // Prompt 3: Nested loop
        {
            "03_nested_loop",
            "For each platinum customer, get their pending orders and add a note saying 'Priority customer'",
            {"getCustomers", "getOrders", "addOrderNote"},
            {
                {"getCustomers", {"tier"}},
                {"getOrders", {"customerId", "status"}},
                {"addOrderNote", {"orderId", "note"}}
            },
            {
                {"getCustomers", {{"tier", "platinum"}}},
                {"getOrders", {{"status", "pending"}}}
            },
            {"await", "for"},
            512
        },

        // Prompt 4: Conditional with multiple actions
        {
            "04_conditional_multi_action",
            "Find orders over $500 with failed payments. Refund the payment and cancel the order with reason 'Payment failed'",
            {"getOrders", "getPayments", "refundPayment", "cancelOrder"},
            {
                {"getOrders", {"minTotal"}},
                {"getPayments", {"orderId", "status"}},
                {"refundPayment", {"paymentId", "reason"}},
                {"cancelOrder", {"orderId", "reason"}}
            },
            {
                {"getPayments", {{"status", "failed"}}},
                {"getOrders", {{"minTotal", "500"}}}
            },
            {"await", "for", "if"},
            512
        },

        // Prompt 5: Complex nested condition
        {
            "05_nested_condition",
            "For each order in 'confirmed' status, check if all products in its line items are in stock. If any product has stock < quantity ordered, flag the order with reason 'Insufficient stock'",
            {"getOrders", "getLineItems", "getProduct", "flagOrder"},
            {
                {"getOrders", {"status"}},
                {"getLineItems", {"orderId"}},
                {"getProduct", {"productId"}},
                {"flagOrder", {"orderId", "reason"}}
            },
            {
                {"getOrders", {{"status", "confirmed"}}}
            },
            {"await", "for", "if"},
            768
        },

        // Prompt 6: Aggregation (note: may use let)
        {
            "06_aggregation",
            "For customers in the EU region, get all completed orders and calculate total revenue. If total > $10000, flag all pending orders for that customer with reason 'High-value customer'",
            {"getCustomers", "getOrders", "flagOrder"},
            {
                {"getCustomers", {"region"}},
                {"getOrders", {"customerId", "status"}},
                {"flagOrder", {"orderId", "reason"}}
            },
            {
                {"getCustomers", {{"region", "EU"}}}
            },
            {"await", "for", "if"},
            768
        },

        // Prompt 7: Parallel execution
        {
            "07_parallel_execution",
            "For the given order, update stock for all line items in parallel using Promise.all",
            {"getLineItems", "getProduct", "updateStock"},
            {
                {"getLineItems", {"orderId"}},
                {"getProduct", {"productId"}},
                {"updateStock", {"productId", "quantity"}}
            },
            {},
            {"await", "Promise.all"},
            512
        },

        // Prompt 8: Null handling
        {
            "08_null_handling",
            "For each confirmed order, get its shipment. If no shipment exists, create one with carrier 'default'. Then update order status to 'shipped'",
            {"getOrders", "getShipment", "createShipment", "updateOrderStatus"},
            {
                {"getOrders", {"status"}},
                {"getShipment", {"orderId"}},
                {"createShipment", {"orderId", "carrier"}},
                {"updateOrderStatus", {"orderId", "status"}}
            },
            {
                {"getOrders", {{"status", "confirmed"}}},
                {"createShipment", {{"carrier", "default"}}},
                {"updateOrderStatus", {{"status", "shipped"}}}
            },
            {"await", "for", "if"},
            512
        },

        // Prompt 9: Multi-filter chained actions
        {
            "09_multi_filter_chain",
            "Find all gold tier customers in EU region with pending orders over $200. For each matching order, process payment with method 'express', then update order to confirmed, then create shipment with carrier 'premium'",
            {"getCustomers", "getOrders", "processPayment", "updateOrderStatus", "createShipment"},
            {
                {"getCustomers", {"tier", "region"}},
                {"getOrders", {"customerId", "status", "minTotal"}},
                {"processPayment", {"orderId", "method"}},
                {"updateOrderStatus", {"orderId", "status"}},
                {"createShipment", {"orderId", "carrier"}}
            },
            {
                {"getCustomers", {{"tier", "gold"}, {"region", "EU"}}},
                {"getOrders", {{"status", "pending"}}},
                {"processPayment", {{"method", "express"}}},
                {"createShipment", {{"carrier", "premium"}}}
            },
            {"await", "for"},
            512
        }
    };
}

// ============================================================
// MODEL WRAPPER
// ============================================================

class ModelRunner {
    std::shared_ptr<Qwen3Model> model_;
    std::unique_ptr<Qwen3ForwardPass> forward_pass_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::string model_path_;
    std::vector<int32_t> system_prompt_tokens_;
    uint32_t system_prompt_cache_pos_ = 0;

    void precompute_system_prompt() {
        // Format system prompt
        std::string system_prompt = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n";
        system_prompt_tokens_ = tokenizer_->encode(system_prompt);

        ggml_backend_sched_t scheduler = model_->get_scheduler();

        // Prefill system prompt
        ggml_backend_sched_reset(scheduler);
        ggml_cgraph* gf_prefill = forward_pass_->build_prefill_graph(system_prompt_tokens_, 0, 0); // Slot 0
        ggml_backend_sched_alloc_graph(scheduler, gf_prefill);
        forward_pass_->set_inputs(gf_prefill, system_prompt_tokens_, 0);
        ggml_backend_sched_graph_compute(scheduler, gf_prefill);
        forward_pass_->advance_cache(system_prompt_tokens_.size(), 0); // Slot 0

        // Store rewind point
        system_prompt_cache_pos_ = forward_pass_->get_cache_pos(0); // Slot 0

        std::cout << "  System prompt tokens: " << system_prompt_tokens_.size() << std::endl;
        std::cout << "  System prompt cache pos: " << system_prompt_cache_pos_ << std::endl;
    }

public:
    explicit ModelRunner(const std::string& model_path) : model_path_(model_path) {}

    bool load() {
        try {
            model_ = std::make_shared<Qwen3Model>();
            model_->load_metadata(model_path_);
            model_->load_tensors();

            // Initialize tokenizer
            tokenizer_ = std::make_unique<Tokenizer>(&model_->get_metadata());

            // Initialize forward pass with large context and batch size 1
            forward_pass_ = std::make_unique<Qwen3ForwardPass>(*model_, &model_->get_metadata(), 8192, 1);

            // Precompute and cache system prompt
            precompute_system_prompt();

            std::cout << "  Loaded: " << model_path_ << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "  Failed to load " << model_path_ << ": " << e.what() << std::endl;
            return false;
        }
    }

    std::string generate(const std::string& prompt, int max_tokens) {
        if (!model_ || !forward_pass_ || !tokenizer_) {
            throw std::runtime_error("Model not loaded");
        }

        // Suppress non-error logs
        ggml_log_set([](ggml_log_level level, const char* text, void* user_data) {
            if (level == GGML_LOG_LEVEL_ERROR) {
                fprintf(stderr, "%s", text);
            }
            (void)user_data;
        }, nullptr);

        // Rewind cache to after system prompt
        forward_pass_->set_cache_pos(system_prompt_cache_pos_, 0); // Slot 0

        qwen3::GreedySampler sampler;
        ggml_backend_sched_t scheduler = model_->get_scheduler();

        // Format and tokenize user prompt only
        std::string user_prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<int32_t> user_tokens = tokenizer_->encode(user_prompt);

        // Track all tokens for sampling context
        std::vector<int32_t> all_tokens = system_prompt_tokens_;
        all_tokens.insert(all_tokens.end(), user_tokens.begin(), user_tokens.end());

        // Prefill user prompt starting at system prompt cache position
        ggml_backend_sched_reset(scheduler);
        ggml_cgraph* gf_prefill = forward_pass_->build_prefill_graph(user_tokens, system_prompt_cache_pos_, 0); // Slot 0
        ggml_backend_sched_alloc_graph(scheduler, gf_prefill);
        forward_pass_->set_inputs(gf_prefill, user_tokens, system_prompt_cache_pos_);
        ggml_backend_sched_graph_compute(scheduler, gf_prefill);
        forward_pass_->advance_cache(user_tokens.size(), 0); // Slot 0

        std::vector<float> logits = forward_pass_->get_output_logits(gf_prefill);
        size_t vocab_size = model_->get_metadata().vocab_size;
        std::vector<float> last_token_logits(logits.end() - vocab_size, logits.end());
        int next_token_id = sampler.sample(last_token_logits, all_tokens);
        all_tokens.push_back(next_token_id);

        // Decode
        std::string generated_text = tokenizer_->decode(next_token_id);
        for (int i = 1; i < max_tokens; ++i) {
            if (next_token_id == model_->get_metadata().eos_token_id) {
                break;
            }

            std::vector<int32_t> current_token_vec = {next_token_id};
            int current_pos = forward_pass_->get_cache_pos(0); // Slot 0

            ggml_backend_sched_reset(scheduler);
            std::vector<uint32_t> d_slots = {0};
            std::vector<int32_t> d_positions = {current_pos};
            ggml_cgraph* gf_decode = forward_pass_->build_decoding_graph(current_token_vec, d_slots, d_positions);
            ggml_backend_sched_alloc_graph(scheduler, gf_decode);
            forward_pass_->set_batched_inputs(gf_decode, current_token_vec, d_slots, d_positions);
            ggml_backend_sched_graph_compute(scheduler, gf_decode);
            forward_pass_->advance_cache(1, 0); // Slot 0

            logits = forward_pass_->get_output_logits(gf_decode);
            last_token_logits.assign(logits.begin(), logits.begin() + vocab_size);

            next_token_id = sampler.sample(last_token_logits, all_tokens);
            all_tokens.push_back(next_token_id);
            generated_text += tokenizer_->decode(next_token_id);
        }

        return generated_text;
    }
};

// ============================================================
// MAIN
// ============================================================

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Order Management DSL Generation Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // Model paths
    const std::string MODEL_QWEN2_14B = "./qwen2.5-coder-14b-instruct-q4_0.gguf";
    // const std::string MODEL_QWEN3_1_7B = "./Qwen3-1.7B-BF16.gguf";      // Placeholder
    // const std::string MODEL_QWEN3_0_6B = "./Qwen3-0.6B-Q8_0.gguf";      // Placeholder

    // Load model
    std::cout << "\n[Loading Model]" << std::endl;
    ModelRunner runner(MODEL_QWEN2_14B);
    if (!runner.load()) {
        std::cerr << "Failed to load model. Exiting." << std::endl;
        return 1;
    }

    // Get test cases
    auto test_cases = get_test_cases();

    // Run tests
    std::vector<TestResult> results;
    int passed = 0;
    int failed = 0;

    std::cout << "\n[Running Tests]" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    for (const auto& tc : test_cases) {
        std::cout << "\nTest: " << tc.name << std::endl;
        std::cout << "Prompt: " << tc.prompt.substr(0, 60) << "..." << std::endl;

        try {
            std::string output = runner.generate(tc.prompt, tc.max_tokens);
            TestResult result = validate_output(tc, output);
            results.push_back(result);

            if (result.passed) {
                std::cout << "Result: PASSED ✓" << std::endl;
                passed++;
            } else {
                std::cout << "Result: FAILED ✗" << std::endl;
                for (const auto& f : result.failures) {
                    std::cout << "  - " << f << std::endl;
                }
                failed++;
            }

            // Print generated code (normalized, truncated)
            std::cout << "Generated:" << std::endl;
            std::string display_code = result.generated_code.substr(0, 500);
            if (result.generated_code.length() > 500) display_code += "...";
            std::cout << display_code << std::endl;

        } catch (const std::exception& e) {
            std::cout << "Result: ERROR - " << e.what() << std::endl;
            failed++;
        }

        std::cout << "----------------------------------------" << std::endl;
    }

    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "SUMMARY" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total:  " << test_cases.size() << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Score:  " << (100 * passed / test_cases.size()) << "%" << std::endl;

    // Detailed failure report
    if (failed > 0) {
        std::cout << "\n[Failed Tests Detail]" << std::endl;
        for (const auto& r : results) {
            if (!r.passed) {
                std::cout << "\n" << r.name << ":" << std::endl;
                for (const auto& f : r.failures) {
                    std::cout << "  - " << f << std::endl;
                }
            }
        }
    }

    return failed > 0 ? 1 : 0;
}