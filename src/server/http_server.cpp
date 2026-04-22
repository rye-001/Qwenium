// http_server.cpp
// Minimal HTTP server for Qwen inference with OpenAI-compatible API
//
// Build dependencies:
//   - cpp-httplib (header-only): https://github.com/yhirose/cpp-httplib
//   - nlohmann/json (header-only): https://github.com/nlohmann/json
//
// Usage:
//   ./http_server [--port 8080] [--model path/to/model.gguf]

#include "inference_server.h"

// You'll need these headers - they're header-only libraries
// Place them in your include path or vendor them
#include "httplib.h"
#include "nlohmann/json.hpp"

// Your existing headers
#include "qwen3-core/qwen3-model.h"
#include "models/qwen3.h"
#include "models/qwen35.h"
#include "models/qwen36.h"
#include "loader/tokenizer.h"
#include "sampling/sampling.h"
#include "state/kv_cache_simple.h"

#include <iostream>
#include <thread>
#include <csignal>
#include <atomic>

using json = nlohmann::json;

// Global for signal handling
std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int signal) {
    std::cout << "\nShutdown requested (signal " << signal << ")" << std::endl;
    g_shutdown_requested = true;
}

// Helper to clean up token artifacts
std::string normalize_output(const std::string& output) {
    std::string normalized = output;
    
    // Replace Ä  (U+0120, UTF-8: C4 A0) with space - common in BPE tokenizers
    std::string g_char = "\xC4\xA0";  // UTF-8 encoding of Ä 
    size_t pos = 0;
    while ((pos = normalized.find(g_char, pos)) != std::string::npos) {
        normalized.replace(pos, g_char.length(), " ");
        pos += 1;
    }
    
    // Replace ÄŠ (U+010A, UTF-8: C4 8A) with newline - common in BPE tokenizers
    std::string c_char = "\xC4\x8A";  // UTF-8 encoding of ÄŠ
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

// =============================================================================
// Integration layer: Wire up InferenceServer to your Qwen implementation
// =============================================================================
class QwenServerIntegration {
    static constexpr int MAX_SLOTS = 10;
    static constexpr int TEMPLATE_SLOT = 0;  // Reserved for system prompt cache

public:
    QwenServerIntegration(const std::string& model_path, int max_ctx_per_slot = 2048)
        : max_ctx_per_slot_(max_ctx_per_slot) {
        
        std::cout << "Loading model from: " << model_path << std::endl;
        model_.load_metadata(model_path);
        model_.load_tensors();
        
        tokenizer_ = std::make_unique<Tokenizer>(&model_.get_metadata());
        sampler_ = std::make_unique<qwen3::GreedySampler>();
        
        // Initialize forward pass with MAX_SLOTS slots
        const auto& srv_meta = model_.get_metadata();
        if (srv_meta.architecture == "qwen35moe")
            forward_pass_ = std::make_unique<Qwen36ForwardPass>(
                model_, &srv_meta, max_ctx_per_slot_, MAX_SLOTS);
        else if (srv_meta.architecture == "qwen35")
            forward_pass_ = std::make_unique<Qwen35ForwardPass>(
                model_, &srv_meta, max_ctx_per_slot_, MAX_SLOTS);
        else
            forward_pass_ = std::make_unique<Qwen3ForwardPass>(
                model_, &srv_meta, max_ctx_per_slot_, MAX_SLOTS);
        
        scheduler_ = model_.get_scheduler();
        
        // Reserve memory for the maximum batch size to prevent reallocation errors
        reserve_max_batch();
        
        std::cout << "Model loaded successfully" << std::endl;
    }

    void reserve_max_batch() {
        std::cout << "Reserving memory for max batch size: " << MAX_SLOTS << " and max ctx: " << max_ctx_per_slot_ << std::endl;
        
        // 1. Reserve for max decode batch
        {
            std::vector<int32_t> tokens(MAX_SLOTS, 0);       // Dummy tokens
            std::vector<uint32_t> slot_ids(MAX_SLOTS);       // 0..MAX_SLOTS-1
            std::vector<int32_t> positions(MAX_SLOTS, 0);    // Dummy positions
            
            for (int i = 0; i < MAX_SLOTS; ++i) {
                slot_ids[i] = i;
            }

            ggml_backend_sched_reset(scheduler_);
            
            // Build the graph for the maximum possible workload
            ggml_cgraph* gf = forward_pass_->build_decoding_graph(tokens, slot_ids, positions);
            
            // Reserve memory in the scheduler
            if (!ggml_backend_sched_reserve(scheduler_, gf)) {
                std::cerr << "WARNING: Failed to reserve memory for max decode batch!" << std::endl;
            }
        }

        // 2. Reserve for max prefill batch
        // We reserve for the maximum number of tokens we might process in a single prefill step.
        // Assuming we might process up to max_ctx_per_slot_ tokens at once (full prompt).
        {
            // Limit reservation to a reasonable batch size if max_ctx is huge, 
            // but here we use max_ctx_per_slot_ as it's the theoretical max.
            // In practice, this should be the configured 'batch_size' for prompt processing.
            // Since we don't have a separate batch_size config here, we use max_ctx_per_slot_.
            int n_tokens = max_ctx_per_slot_;
            
            // Use a safe upper bound if max_ctx is very large to avoid OOM during reservation if not needed
            if (n_tokens > 2048) n_tokens = 2048; // Cap at 2048 for reservation safety if not specified otherwise

            std::vector<int32_t> tokens(n_tokens, 0);
            
            ggml_backend_sched_reset(scheduler_);
            ggml_cgraph* gf = forward_pass_->build_prefill_graph(tokens, 0, 0);
             if (!ggml_backend_sched_reserve(scheduler_, gf)) {
                std::cerr << "WARNING: Failed to reserve memory for max prefill batch!" << std::endl;
            }
        }
        
        std::cout << "Memory reservation complete." << std::endl;
    }

    // Cache a system prompt - returns cache length, or 0 on failure
    int cache_system_prompt(const std::string& system_prompt) {
        std::lock_guard<std::mutex> lock(model_mutex_);
        
        size_t hash = std::hash<std::string>{}(system_prompt);
        
        // Already cached?
        if (hash == cached_system_hash_ && system_prompt_cache_len_ > 0) {
            return system_prompt_cache_len_;
        }
        
        // Format and tokenize system prompt
        std::string formatted = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
        std::vector<int32_t> tokens = tokenizer_->encode(formatted);
        
        if (tokens.empty()) {
            return 0;
        }
        
        // Clear template slot
        forward_pass_->clear_slot(TEMPLATE_SLOT);
        
        // Prefill system prompt into template slot
        ggml_backend_sched_reset(scheduler_);
        ggml_cgraph* gf = forward_pass_->build_prefill_graph(tokens, 0, TEMPLATE_SLOT);
        ggml_backend_sched_alloc_graph(scheduler_, gf);
        forward_pass_->set_inputs(gf, tokens, 0);
        ggml_backend_sched_graph_compute(scheduler_, gf);
        forward_pass_->advance_cache(tokens.size(), TEMPLATE_SLOT);
        
        // Store cache info
        cached_system_hash_ = hash;
        system_prompt_cache_len_ = forward_pass_->get_cache_pos(TEMPLATE_SLOT);
        
        std::cout << "Cached system prompt: " << tokens.size() << " tokens, cache_pos=" 
                  << system_prompt_cache_len_ << std::endl;
        
        return system_prompt_cache_len_;
    }

    void configure_server(qwen3::InferenceServer& server) {
        server.set_tokenize([this](const std::string& text) {
            // Apply Qwen chat template for user message only
            // System prompt handled separately via caching
            std::string formatted = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
            return tokenizer_->encode(formatted);
        });

        server.set_detokenize([this](int token_id) {
            std::string text = tokenizer_->decode(token_id);
            return normalize_output(text);
        });

        server.set_prefill([this](int slot_id, const std::vector<int32_t>& tokens, int start_pos) {
            return run_prefill(slot_id, tokens, start_pos);
        });

        server.set_batched_decode([this](const std::vector<int32_t>& tokens,
                                         const std::vector<int>& slot_ids) {
            return run_batched_decode(tokens, slot_ids);
        });

        server.set_clear_slot([this](int slot_id) {
            forward_pass_->clear_slot(slot_id);
        });

        server.set_clone_slot([this](int from_slot, int to_slot) {
            std::lock_guard<std::mutex> lock(model_mutex_);
            if (system_prompt_cache_len_ > 0) {
                forward_pass_->clone_slot(from_slot, to_slot, system_prompt_cache_len_);
            }
        });

        server.set_get_eos_token([this]() {
            return model_.get_metadata().eos_token_id;
        });
    }
    
    // Check if system prompt matches cache, cache if new
    // Returns start_pos for prefill (0 if no cache, cache_len if cached)
    int prepare_system_prompt(const std::string& system_prompt) {
        if (system_prompt.empty()) {
            return 0;  // No caching
        }
        
        size_t hash = std::hash<std::string>{}(system_prompt);
        
        // Check if matches current cache
        if (hash == cached_system_hash_ && system_prompt_cache_len_ > 0) {
            return system_prompt_cache_len_;
        }
        
        // New system prompt - cache it
        return cache_system_prompt(system_prompt);
    }
    
    int get_system_prompt_cache_len() const {
        return system_prompt_cache_len_;
    }

private:
    int run_prefill(int slot_id, const std::vector<int32_t>& tokens, int start_pos) {
        std::lock_guard<std::mutex> lock(model_mutex_);
        
        ggml_backend_sched_reset(scheduler_);
        
        // Use build_prefill_graph with slot_id
        ggml_cgraph* gf = forward_pass_->build_prefill_graph(tokens, start_pos, slot_id);
        ggml_backend_sched_alloc_graph(scheduler_, gf);
        forward_pass_->set_inputs(gf, tokens, start_pos);
        ggml_backend_sched_graph_compute(scheduler_, gf);
        forward_pass_->advance_cache(tokens.size(), slot_id);

        // Sample first token
        std::vector<float> logits = forward_pass_->get_output_logits(gf);
        size_t vocab_size = model_.get_metadata().vocab_size;
        std::vector<float> last_token_logits(logits.end() - vocab_size, logits.end());
        
        std::vector<int32_t> context(tokens);
        return sampler_->sample(last_token_logits, context);
    }

    std::vector<int> run_batched_decode(const std::vector<int32_t>& tokens,
                                        const std::vector<int>& slot_ids) {
        std::lock_guard<std::mutex> lock(model_mutex_);
        
        ggml_backend_sched_reset(scheduler_);
        
        size_t n_batch = slot_ids.size();
        std::vector<uint32_t> slot_ids_u32;
        std::vector<int32_t> positions;
        slot_ids_u32.reserve(n_batch);
        positions.reserve(n_batch);

        for (int slot_id : slot_ids) {
            slot_ids_u32.push_back(static_cast<uint32_t>(slot_id));
            positions.push_back(forward_pass_->get_cache_pos(slot_id));
        }

        ggml_cgraph* gf = forward_pass_->build_decoding_graph(tokens, slot_ids_u32, positions);
        ggml_backend_sched_alloc_graph(scheduler_, gf);
        forward_pass_->set_batched_inputs(gf, tokens, slot_ids_u32, positions);
        ggml_backend_sched_graph_compute(scheduler_, gf);

        std::vector<int> next_tokens;
        next_tokens.reserve(n_batch);
        
        for (size_t i = 0; i < n_batch; ++i) {
            std::vector<float> slot_logits = forward_pass_->get_output_logits_for_slot(gf, i);
            next_tokens.push_back(sampler_->sample(slot_logits, {}));
            
            // Advance cache for each slot in the batch
            forward_pass_->advance_cache(1, slot_ids[i]);
        }
        
        return next_tokens;
    }

    Qwen3Model model_;
    std::unique_ptr<ForwardPassBase> forward_pass_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<qwen3::GreedySampler> sampler_;
    ggml_backend_sched_t scheduler_;
    std::mutex model_mutex_;  // Protects all model operations
    int max_ctx_per_slot_;
    
    // System prompt caching
    size_t cached_system_hash_ = 0;
    int system_prompt_cache_len_ = 0;
};

// =============================================================================
// HTTP Routes
// =============================================================================
void setup_routes(httplib::Server& http, qwen3::InferenceServer& inference, QwenServerIntegration& integration) {
    
    // Health check
    http.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        json response = {
            {"status", "ok"},
            {"active_slots", inference.stats().active_slots.load()},
            {"queue_depth", inference.stats().queue_depth.load()},
            {"requests_completed", inference.stats().requests_completed.load()},
            {"tokens_generated", inference.stats().tokens_generated.load()}
        };
        res.set_content(response.dump(), "application/json");
    });

    // OpenAI-compatible completions endpoint
    http.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::parse_error& e) {
            res.status = 400;
            res.set_content(json({{"error", "Invalid JSON"}}).dump(), "application/json");
            return;
        }

        // Extract parameters
        std::string prompt = body.value("prompt", "");
        if (prompt.empty()) {
            res.status = 400;
            res.set_content(json({{"error", "Missing 'prompt' field"}}).dump(), "application/json");
            return;
        }

        auto inf_req = std::make_shared<qwen3::InferenceRequest>();
        inf_req->prompt = prompt;
        inf_req->max_tokens = body.value("max_tokens", 256);
        inf_req->temperature = body.value("temperature", 0.0f);
        
        // Handle system prompt caching
        std::string system_prompt = body.value("system_prompt", "");
        if (!system_prompt.empty()) {
            inf_req->system_prompt = system_prompt;
            inf_req->start_pos = integration.prepare_system_prompt(system_prompt);
        }

        bool stream = body.value("stream", false);

        // Submit request
        if (!inference.submit(inf_req)) {
            res.status = 503;
            res.set_content(json({{"error", "Server overloaded, queue full"}}).dump(), "application/json");
            return;
        }

        if (stream) {
            // SSE streaming response
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_header("X-Accel-Buffering", "no");  // Disable nginx buffering

            res.set_chunked_content_provider(
                "text/event-stream",
                [inf_req, &inference](size_t /*offset*/, httplib::DataSink& sink) {
                    while (true) {
                        int token_id = inf_req->token_queue->pop_blocking();

                        if (token_id == qwen3::TokenQueue::QUEUE_END) {
                            // Send final event
                            std::string done_event = "data: [DONE]\n\n";
                            sink.write(done_event.c_str(), done_event.size());
                            sink.done();
                            return false;  // Stop
                        }

                        std::string token_text = inference.decode_token(token_id);
                        
                        json chunk = {
                            {"object", "text_completion"},
                            {"choices", {{
                                {"text", token_text},
                                {"index", 0},
                                {"finish_reason", nullptr}
                            }}}
                        };
                        
                        std::string sse = "data: " + chunk.dump() + "\n\n";
                        if (!sink.write(sse.c_str(), sse.size())) {
                            // Client disconnected
                            inf_req->cancelled = true;
                            return false;
                        }
                    }
                    return true;
                },
                [inf_req](bool success) {
                    if (!success) {
                        inf_req->cancelled = true;
                    }
                }
            );
        } else {
            // Non-streaming: collect all tokens
            std::string full_text;
            int token_count = 0;
            
            while (true) {
                int token_id = inf_req->token_queue->pop_blocking();
                if (token_id == qwen3::TokenQueue::QUEUE_END) break;
                full_text += inference.decode_token(token_id);
                token_count++;
            }

            json response = {
                {"object", "text_completion"},
                {"choices", {{
                    {"text", full_text},
                    {"index", 0},
                    {"finish_reason", "stop"}
                }}},
                {"usage", {
                    {"completion_tokens", token_count}
                }}
            };
            res.set_content(response.dump(), "application/json");
        }
    });

    // Models endpoint (for compatibility)
    http.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        json response = {
            {"object", "list"},
            {"data", {{
                {"id", "qwen-local"},
                {"object", "model"},
                {"owned_by", "local"}
            }}}
        };
        res.set_content(response.dump(), "application/json");
    });
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
    // Parse arguments
    int port = 8080;
    std::string model_path;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --port, -p PORT    Port to listen on (default: 8080)\n"
                      << "  --model, -m PATH   Path to model file\n"
                      << "  --help, -h         Show this help\n";
            return 0;
        }
    }

    // Setup signal handling
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        // Initialize model integration
        QwenServerIntegration integration(model_path);

        // Create inference server
        qwen3::InferenceServer::Config config;
        config.max_slots = 10;
        config.max_queue_depth = 100;
        config.request_timeout = std::chrono::seconds(120);

        qwen3::InferenceServer inference(config);
        integration.configure_server(inference);

        // Start inference thread
        std::thread inference_thread([&inference]() {
            inference.run();
        });

        // Setup HTTP server
        httplib::Server http;
        setup_routes(http, inference, integration);

        // Shutdown handler thread
        std::thread shutdown_thread([&http, &inference]() {
            while (!g_shutdown_requested) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::cout << "Stopping server..." << std::endl;
            inference.stop();
            http.stop();
        });

        std::cout << "Server starting on http://0.0.0.0:" << port << std::endl;
        std::cout << "Endpoints:" << std::endl;
        std::cout << "  GET  /health" << std::endl;
        std::cout << "  POST /v1/completions" << std::endl;
        std::cout << "  GET  /v1/models" << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;

        http.listen("0.0.0.0", port);

        // Cleanup
        inference.stop();
        if (inference_thread.joinable()) {
            inference_thread.join();
        }
        if (shutdown_thread.joinable()) {
            shutdown_thread.join();
        }

        std::cout << "Server stopped" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}