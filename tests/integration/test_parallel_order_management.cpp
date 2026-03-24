#include "qwen3-core/qwen3-model.h"
#include "qwen3-core/gguf-loader.h"
#include "qwen3-core/forward-pass-factory.h"
#include "qwen3-core/tokenizer.h"
#include "qwen3-core/sampling.h"
#include "qwen3-core/vocab_utils.h"
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <set>
#include <regex>
#include <chrono>
#include <algorithm>

static std::string get_model_path() {
    const char* path = std::getenv("QWEN_MODEL_PATH");
    return path ? std::string(path) : "";
}

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
    std::regex pattern1(fn_name + "\\s*\\(\\s*\\{[^}]*" + key + "\\s*:");
    if (std::regex_search(code, pattern1)) {
        return true;
    }
    std::regex pattern2(fn_name + "\\s*\\(\\s*\\{[^}]*\\b" + key + "\\b[^:}]*\\}");
    return std::regex_search(code, pattern2);
}

bool contains_param_value(const std::string& code, const std::string& fn_name, 
                          const std::string& key, const std::string& value) {
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
    
    std::string g_char = "\xC4\xA0";
    size_t pos = 0;
    while ((pos = normalized.find(g_char, pos)) != std::string::npos) {
        normalized.replace(pos, g_char.length(), " ");
        pos += 1;
    }
    
    std::string c_char = "\xC4\x8A";
    pos = 0;
    while ((pos = normalized.find(c_char, pos)) != std::string::npos) {
        normalized.replace(pos, c_char.length(), "\n");
        pos += 1;
    }
    
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
    
    std::string output = normalize_output(raw_output);
    result.generated_code = output;

    for (const auto& fn : tc.expected_functions) {
        if (!contains_function_call(output, fn)) {
            result.passed = false;
            result.failures.push_back("Missing function call: " + fn);
        }
    }

    for (const auto& [fn, keys] : tc.required_param_keys) {
        for (const auto& key : keys) {
            if (!contains_param_key(output, fn, key)) {
                result.passed = false;
                result.failures.push_back("Missing param key '" + key + "' in " + fn);
            }
        }
    }

    for (const auto& [fn, kv_pairs] : tc.spot_check_values) {
        for (const auto& [key, value] : kv_pairs) {
            if (!contains_param_value(output, fn, key, value)) {
                result.passed = false;
                result.failures.push_back("Missing value " + key + ": " + value + " in " + fn);
            }
        }
    }

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
        {
            "01_simple_query",
            "Get all products in the electronics category",
            {"getProducts"},
            {{"getProducts", {"category"}}},
            {{"getProducts", {{"category", "electronics"}}}},
            {"await"},
            256
        },
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
// SLOT STATE FOR PARALLEL INFERENCE
// ============================================================

struct SlotState {
    uint32_t slot_id;
    int test_idx;
    int position;                           // Current position in sequence
    int max_tokens;                         // Max tokens to generate
    int tokens_generated;                   // Tokens generated so far
    std::vector<int32_t> all_tokens;        // Full token history for sampling
    std::string generated_text;             // Accumulated output
    bool finished;                          // Hit EOS or max tokens
    int last_token_id;                      // Last generated token
};

// ============================================================
// PARALLEL MODEL RUNNER
// ============================================================

class ParallelModelRunner {
    std::shared_ptr<Qwen3Model> model_;
    std::unique_ptr<ForwardPassBase> forward_pass_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::string model_path_;
    std::vector<int32_t> system_prompt_tokens_;
    uint32_t system_prompt_cache_pos_ = 0;
    uint32_t max_batch_size_;

    void precompute_system_prompt() {
        std::string system_prompt = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n";
        system_prompt_tokens_ = tokenizer_->encode(system_prompt);

        ggml_backend_sched_t scheduler = model_->get_scheduler();

        // Prefill system prompt into slot 0
        ggml_backend_sched_reset(scheduler);
        ggml_cgraph* gf_prefill = forward_pass_->build_prefill_graph(system_prompt_tokens_, 0);
        ggml_backend_sched_alloc_graph(scheduler, gf_prefill);
        forward_pass_->set_inputs(gf_prefill, system_prompt_tokens_, 0);
        ggml_backend_sched_graph_compute(scheduler, gf_prefill);
        forward_pass_->advance_cache(system_prompt_tokens_.size(), 0);

        system_prompt_cache_pos_ = forward_pass_->get_cache_pos(0);

        std::cout << "  System prompt tokens: " << system_prompt_tokens_.size() << std::endl;
        std::cout << "  System prompt cache pos: " << system_prompt_cache_pos_ << std::endl;
    }

public:
    explicit ParallelModelRunner(const std::string& model_path, uint32_t max_batch_size) 
        : model_path_(model_path), max_batch_size_(max_batch_size) {}

    bool load() {
        try {
            model_ = std::make_shared<Qwen3Model>();
            model_->load_metadata(model_path_);
            model_->load_tensors();

            tokenizer_ = std::make_unique<Tokenizer>(&model_->get_metadata());

            // Initialize forward pass with batch support
            forward_pass_ = create_forward_pass(
                *model_, &model_->get_metadata(), 2048, max_batch_size_);

            precompute_system_prompt();

            std::cout << "  Loaded: " << model_path_ << std::endl;
            std::cout << "  Max batch size: " << max_batch_size_ << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "  Failed to load " << model_path_ << ": " << e.what() << std::endl;
            return false;
        }
    }

    std::vector<std::string> generate_parallel(const std::vector<TestCase>& test_cases) {
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

        size_t n_tests = test_cases.size();
        if (n_tests > max_batch_size_) {
            throw std::runtime_error("Too many tests for batch size");
        }

        qwen3::GreedySampler sampler;

        // Load pruned vocabulary if specified
        // std::unordered_set<int32_t> pruned_vocab = qwen3::load_keep_list("./output/vocab_artifacts/keep_list.bin");
        // sampler.set_pruned_vocab(&pruned_vocab);
        // std::cout << "Loaded pruned vocabulary: " << pruned_vocab.size() << " tokens\n";

        ggml_backend_sched_t scheduler = model_->get_scheduler();
        int eos_token_id = model_->get_metadata().eos_token_id;
        size_t vocab_size = model_->get_metadata().vocab_size;

        // ============================================================
        // Phase 1: Clone prefix to all slots
        // ============================================================
        std::cout << "\n  [Phase 1] Cloning prefix to " << n_tests << " slots..." << std::endl;
        
        for (size_t i = 0; i < n_tests; i++) {
            // Clone from slot 0 to slot i
            // Slot 0 has the system prompt, we copy it to slots 1..N-1
            // Actually, we use slots 0..N-1, so clone slot 0 state to itself is a no-op
            // We'll use slot 0 as source, clone to slots 1..N-1
            if (i > 0) {
                forward_pass_->clone_slot(0, i, system_prompt_cache_pos_);
            }
        }

        // ============================================================
        // Phase 2: Prefill user prompts (sequential - different lengths)
        // ============================================================
        std::cout << "  [Phase 2] Prefilling user prompts..." << std::endl;

        std::vector<SlotState> slots(n_tests);
        
        for (size_t i = 0; i < n_tests; i++) {
            const auto& tc = test_cases[i];
            
            // Format user prompt
            std::string user_prompt = "<|im_start|>user\n" + tc.prompt + "<|im_end|>\n<|im_start|>assistant\n";
            std::vector<int32_t> user_tokens = tokenizer_->encode(user_prompt);

            // Initialize slot state
            slots[i].slot_id = i;
            slots[i].test_idx = i;
            slots[i].max_tokens = tc.max_tokens;
            slots[i].tokens_generated = 0;
            slots[i].finished = false;
            slots[i].generated_text = "";
            
            // Token history for sampling
            slots[i].all_tokens = system_prompt_tokens_;
            slots[i].all_tokens.insert(slots[i].all_tokens.end(), 
                                       user_tokens.begin(), user_tokens.end());

            // Prefill user prompt for this slot
            ggml_backend_sched_reset(scheduler);
            ggml_cgraph* gf_prefill = forward_pass_->build_prefill_graph(
                user_tokens, system_prompt_cache_pos_, i);
            ggml_backend_sched_alloc_graph(scheduler, gf_prefill);
            forward_pass_->set_inputs(gf_prefill, user_tokens, system_prompt_cache_pos_);
            ggml_backend_sched_graph_compute(scheduler, gf_prefill);
            forward_pass_->advance_cache(user_tokens.size(), i);

            // Get first token from prefill
            std::vector<float> logits = forward_pass_->get_output_logits(gf_prefill);
            std::vector<float> last_token_logits(logits.end() - vocab_size, logits.end());
            int next_token_id = sampler.sample(last_token_logits, slots[i].all_tokens);
            
            slots[i].last_token_id = next_token_id;
            slots[i].all_tokens.push_back(next_token_id);
            slots[i].generated_text = tokenizer_->decode(next_token_id);
            slots[i].tokens_generated = 1;
            slots[i].position = forward_pass_->get_cache_pos(i);

            if (next_token_id == eos_token_id) {
                slots[i].finished = true;
            }

            std::cout << "    Slot " << i << ": prefilled " << user_tokens.size() 
                      << " tokens, pos=" << slots[i].position << std::endl;
        }

        // ============================================================
        // Phase 3: Batched decode loop
        // ============================================================
        std::cout << "  [Phase 3] Batched decoding..." << std::endl;

        int max_iterations = 768;  // Safety limit
        int iteration = 0;

        while (iteration < max_iterations) {
            // Collect active slots
            std::vector<int32_t> batch_tokens;
            std::vector<uint32_t> batch_slots;
            std::vector<int32_t> batch_positions;
            std::vector<size_t> active_indices;

            for (size_t i = 0; i < n_tests; i++) {
                if (!slots[i].finished) {
                    batch_tokens.push_back(slots[i].last_token_id);
                    batch_slots.push_back(slots[i].slot_id);
                    batch_positions.push_back(slots[i].position);
                    active_indices.push_back(i);
                }
            }

            // Check if all done
            if (batch_tokens.empty()) {
                std::cout << "    All slots finished at iteration " << iteration << std::endl;
                break;
            }

            // Build and run batched decode
            ggml_backend_sched_reset(scheduler);
            ggml_cgraph* gf_decode = forward_pass_->build_decoding_graph(
                batch_tokens, batch_slots, batch_positions);
            ggml_backend_sched_alloc_graph(scheduler, gf_decode);
            forward_pass_->set_batched_inputs(gf_decode, batch_tokens, batch_slots, batch_positions);
            ggml_backend_sched_graph_compute(scheduler, gf_decode);

            // Sample for each active slot
            for (size_t b = 0; b < active_indices.size(); b++) {
                size_t idx = active_indices[b];
                SlotState& slot = slots[idx];

                std::vector<float> slot_logits = forward_pass_->get_output_logits_for_slot(gf_decode, b);
                int next_token_id = sampler.sample(slot_logits, slot.all_tokens);
                
                // Update slot state
                slot.last_token_id = next_token_id;
                slot.all_tokens.push_back(next_token_id);
                slot.generated_text += tokenizer_->decode(next_token_id);
                slot.tokens_generated++;
                slot.position++;

                // Advance cache for this slot
                forward_pass_->advance_cache(1, slot.slot_id);

                // Check termination
                if (next_token_id == eos_token_id || 
                    slot.tokens_generated >= slot.max_tokens) {
                    slot.finished = true;
                }
            }

            iteration++;

            // Progress update every 50 iterations
            if (iteration % 50 == 0) {
                int active_count = 0;
                for (const auto& s : slots) if (!s.finished) active_count++;
                std::cout << "    Iteration " << iteration 
                          << ", active slots: " << active_count << std::endl;
            }
        }

        // Collect results
        std::vector<std::string> results(n_tests);
        for (size_t i = 0; i < n_tests; i++) {
            results[i] = slots[i].generated_text;
        }

        return results;
    }
};

// ============================================================
// MAIN
// ============================================================

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "PARALLEL Order Management DSL Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    const std::string MODEL_QWEN = get_model_path();// "./qwen2.5-coder-14b-instruct-q4_0.gguf";

    auto test_cases = get_test_cases();
    uint32_t batch_size = test_cases.size();

    // Load model with batch support
    std::cout << "\n[Loading Model]" << std::endl;
    ParallelModelRunner runner(MODEL_QWEN, batch_size);
    if (!runner.load()) {
        std::cerr << "Failed to load model. Exiting." << std::endl;
        return 1;
    }

    // Run parallel generation
    std::cout << "\n[Running Parallel Generation]" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::string> outputs;
    try {
        outputs = runner.generate_parallel(test_cases);
    } catch (const std::exception& e) {
        std::cerr << "Generation failed: " << e.what() << std::endl;
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\n[Generation completed in " << duration.count() << " ms]" << std::endl;

    // Validate results
    std::cout << "\n[Validating Results]" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::vector<TestResult> results;
    int passed = 0;
    int failed = 0;

    for (size_t i = 0; i < test_cases.size(); i++) {
        const auto& tc = test_cases[i];
        const auto& output = outputs[i];

        std::cout << "\nTest: " << tc.name << std::endl;

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

        // Print generated code (truncated)
        std::cout << "Generated:" << std::endl;
        std::string display_code = result.generated_code.substr(0, 500);
        if (result.generated_code.length() > 500) display_code += "...";
        std::cout << display_code << std::endl;
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
    std::cout << "Time:   " << duration.count() << " ms" << std::endl;
    std::cout << "Throughput: " << (test_cases.size() * 1000.0 / duration.count()) 
              << " tests/sec" << std::endl;

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

    // ============================================================
    // PERFORMANCE BASELINE CHECK
    // ============================================================
    std::cout << "\n[Performance Baseline]" << std::endl;
    double throughput = (test_cases.size() * 1000.0 / duration.count());
    double min_throughput = 0.12; // Baseline is ~0.15, allow some variance. For b6697 it was 0.14. Now, 0.12.
    bool perf_pass = true;

    if (throughput < min_throughput) {
        std::cerr << "FAIL: Throughput regression. Expected >= " << min_throughput 
                  << " tests/sec, got " << throughput << " tests/sec" << std::endl;
        perf_pass = false;
    } else {
        std::cout << "PASS: Throughput " << throughput << " >= " << min_throughput << " tests/sec" << std::endl;
    }

    if (!perf_pass) {
        return 1;
    }

    return failed > 0 ? 1 : 0;
}
