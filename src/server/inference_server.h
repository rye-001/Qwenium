#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <vector>
#include <set>
#include <atomic>
#include <chrono>
#include <string>
#include <functional>

namespace qwenium {

// Forward declarations - these come from your existing codebase
class Model;
class Qwen3ForwardPass;
class SimpleKVCache;
class Tokenizer;
class GreedySampler;

// =============================================================================
// TokenQueue: Per-request token delivery (HTTP handler blocks on this)
// =============================================================================
struct TokenQueue {
    static constexpr int QUEUE_END = -1;  // Sentinel for completion

    void push(int token_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        tokens_.push(token_id);
        cv_.notify_one();
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        cv_.notify_one();
    }

    // Returns token_id, or QUEUE_END when done
    int pop_blocking() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !tokens_.empty() || finished_; });
        if (tokens_.empty()) return QUEUE_END;
        int token = tokens_.front();
        tokens_.pop();
        return token;
    }

    // Non-blocking version, returns false if nothing available
    bool try_pop(int& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (tokens_.empty()) {
            if (finished_) {
                out = QUEUE_END;
                return true;
            }
            return false;
        }
        out = tokens_.front();
        tokens_.pop();
        return true;
    }

private:
    std::queue<int> tokens_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool finished_ = false;
};

// =============================================================================
// InferenceRequest: Submitted by HTTP handlers
// =============================================================================
struct InferenceRequest {
    std::string system_prompt;  // Optional: enables prefix caching if provided
    std::string prompt;
    int max_tokens = 256;
    float temperature = 0.0f;  // 0 = greedy
    int start_pos = 0;         // Set by server when system prompt is cached
    std::vector<std::string> stop;  // Optional: generation ends before first match
    std::shared_ptr<TokenQueue> token_queue = std::make_shared<TokenQueue>();
    std::atomic<bool> cancelled{false};

    // Outputs set by the inference thread. These are written before
    // token_queue->finish(), so the happens-before edge of the queue mutex makes
    // them safe to read once the HTTP handler observes QUEUE_END.
    int         prompt_tokens     = 0;   // Tokens in the (templated) prompt
    int         completion_tokens = 0;   // Tokens generated so far
    std::string output_text;             // Canonical completion (stop-truncated)
    std::string finish_reason;           // "stop" | "length" | "timeout" | "cancelled" | "error"
    std::string error_message;           // Non-empty => fail-loud: request rejected with a named reason

    // Timing info (set by server)
    std::chrono::steady_clock::time_point submitted_at;
    std::chrono::steady_clock::time_point started_at;
};

// =============================================================================
// RequestQueue: Thread-safe queue for incoming requests
// =============================================================================
struct RequestQueue {
    void push(std::shared_ptr<InferenceRequest> req) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(req));
        cv_.notify_one();
    }

    std::shared_ptr<InferenceRequest> pop_or_wait(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return nullptr;
        }
        auto req = std::move(queue_.front());
        queue_.pop();
        return req;
    }

    bool try_pop(std::shared_ptr<InferenceRequest>& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        out = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<std::shared_ptr<InferenceRequest>> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

// =============================================================================
// Slot: Tracks state for one active generation
// =============================================================================
struct Slot {
    int slot_id = -1;
    std::shared_ptr<InferenceRequest> request;
    std::vector<int32_t> context_tokens;  // Full context including prompt
    int last_token = 0;
    int tokens_generated = 0;
    bool active = false;

    void reset() {
        request.reset();
        context_tokens.clear();
        last_token = 0;
        tokens_generated = 0;
        active = false;
    }
};

// =============================================================================
// ServerStats: Observable metrics
// =============================================================================
struct ServerStats {
    std::atomic<uint64_t> requests_received{0};
    std::atomic<uint64_t> requests_completed{0};
    std::atomic<uint64_t> requests_cancelled{0};
    std::atomic<uint64_t> tokens_generated{0};
    std::atomic<uint64_t> active_slots{0};
    std::atomic<uint64_t> queue_depth{0};
};

// =============================================================================
// InferenceServer: Main server class (runs inference loop in dedicated thread)
// =============================================================================
class InferenceServer {
public:
    struct Config {
        int max_slots = 10;
        int max_queue_depth = 100;  // 0 = unlimited
        int system_prompt_len = 0;  // If using prefix caching
        int max_context = 0;        // Per-slot prompt-token ceiling; 0 = no limit (fail-loud guard)
        std::chrono::seconds request_timeout{60};
    };

    // Callback types for integration with your existing code
    using TokenizeFunc = std::function<std::vector<int32_t>(const std::string&)>;
    using DetokenizeFunc = std::function<std::string(int)>;
    using PrefillFunc = std::function<int(int slot_id, const std::vector<int32_t>& tokens, int start_pos)>;
    using BatchedDecodeFunc = std::function<std::vector<int>(const std::vector<int32_t>& tokens,
                                                              const std::vector<int>& slot_ids)>;
    using ClearSlotFunc = std::function<void(int slot_id)>;
    using CloneSlotFunc = std::function<void(int from_slot, int to_slot)>;
    using GetEosTokenFunc = std::function<int()>;

    InferenceServer(const Config& config) : config_(config), slots_(config.max_slots) {
        for (int i = 0; i < config.max_slots; ++i) {
            slots_[i].slot_id = i;
        }
    }

    // Wire up callbacks before calling start()
    void set_tokenize(TokenizeFunc fn) { tokenize_ = std::move(fn); }
    void set_detokenize(DetokenizeFunc fn) { detokenize_ = std::move(fn); }
    void set_prefill(PrefillFunc fn) { prefill_ = std::move(fn); }
    void set_batched_decode(BatchedDecodeFunc fn) { batched_decode_ = std::move(fn); }
    void set_clear_slot(ClearSlotFunc fn) { clear_slot_ = std::move(fn); }
    void set_clone_slot(CloneSlotFunc fn) { clone_slot_ = std::move(fn); }
    void set_get_eos_token(GetEosTokenFunc fn) { get_eos_token_ = std::move(fn); }

    // Submit a request (called from HTTP thread)
    bool submit(std::shared_ptr<InferenceRequest> req) {
        if (config_.max_queue_depth > 0 && request_queue_.size() >= (size_t)config_.max_queue_depth) {
            return false;  // Queue full, reject
        }
        req->submitted_at = std::chrono::steady_clock::now();
        request_queue_.push(req);
        stats_.requests_received++;
        stats_.queue_depth = request_queue_.size();
        return true;
    }

    // Decode a token (called from HTTP thread for streaming)
    std::string decode_token(int token_id) {
        return detokenize_(token_id);
    }

    // Get stats (thread-safe)
    const ServerStats& stats() const { return stats_; }

    // Start inference loop (blocks - run in dedicated thread)
    void run() {
        running_ = true;
        while (running_) {
            // 1. Assign pending requests to free slots
            assign_requests_to_slots();

            // 2. If nothing active, wait for requests
            if (active_slot_ids_.empty()) {
                auto req = request_queue_.pop_or_wait(std::chrono::milliseconds(100));
                if (req && !req->cancelled) {
                    assign_to_slot(req);
                }
                continue;
            }

            // 3. Batched decode step
            decode_step();
        }
    }

    void stop() {
        running_ = false;
    }

    bool is_running() const { return running_; }

private:
    void assign_requests_to_slots() {
        std::shared_ptr<InferenceRequest> req;
        while (active_slot_ids_.size() < (size_t)config_.max_slots && request_queue_.try_pop(req)) {
            stats_.queue_depth = request_queue_.size();
            if (!req->cancelled) {
                assign_to_slot(req);
            }
        }
    }

    int find_free_slot() {
        for (int i = 0; i < config_.max_slots; ++i) {
            if (!slots_[i].active) return i;
        }
        return -1;
    }

    void assign_to_slot(std::shared_ptr<InferenceRequest> req) {
        int slot_id = find_free_slot();
        if (slot_id < 0) return;

        req->started_at = std::chrono::steady_clock::now();

        // Tokenize prompt
        std::vector<int32_t> tokens = tokenize_(req->prompt);
        req->prompt_tokens = static_cast<int>(tokens.size());

        // Fail-loud guard: an oversized prompt would overflow the per-slot KV
        // cache (a downstream GGML_ASSERT abort). Reject it here with a named
        // error naming the slot, the expected ceiling, and the actual count.
        if (config_.max_context > 0 && req->prompt_tokens > config_.max_context) {
            req->error_message =
                "slot " + std::to_string(slot_id) +
                ": prompt too large; expected: <= " + std::to_string(config_.max_context) +
                " tokens, actual: " + std::to_string(req->prompt_tokens);
            req->finish_reason = "error";
            req->token_queue->finish();  // slot not consumed
            return;
        }

        // Clone system prompt cache if using prefix caching
        int start_pos = req->start_pos;
        if (clone_slot_ && start_pos > 0) {
            clone_slot_(0, slot_id);
        }

        // Run prefill and get first token
        int first_token = prefill_(slot_id, tokens, start_pos);

        // Setup slot state
        auto& slot = slots_[slot_id];
        slot.request = req;
        slot.context_tokens = std::move(tokens);
        slot.last_token = first_token;
        slot.tokens_generated = 1;
        slot.active = true;
        active_slot_ids_.insert(slot_id);
        stats_.active_slots = active_slot_ids_.size();

        // Deliver first token (may already be EOS / hit a stop string / max_tokens).
        if (deliver_token(slot, first_token)) {
            req->token_queue->finish();
            stats_.requests_completed++;
            if (clear_slot_) clear_slot_(slot_id);
            slots_[slot_id].reset();
            active_slot_ids_.erase(slot_id);
            stats_.active_slots = active_slot_ids_.size();
        }
    }

    // Record one generated token against a slot's request: deliver it to streaming
    // consumers, fold it into the canonical output_text, and decide whether the
    // request is now complete. Callers must set slot.tokens_generated to the count
    // including this token before calling. Returns true when generation is done;
    // the request's finish_reason is set on the boundary.
    bool deliver_token(Slot& slot, int token) {
        auto& req = *slot.request;

        // Always deliver the raw token id to streaming consumers.
        req.token_queue->push(token);
        stats_.tokens_generated++;

        if (token == get_eos_token_()) {
            req.finish_reason = "stop";
            return true;
        }

        req.completion_tokens = slot.tokens_generated;
        req.output_text += detokenize_(token);
        slot.context_tokens.push_back(token);
        slot.last_token = token;

        // Stop strings (item 2): end the output *before* the first match.
        for (const auto& s : req.stop) {
            if (s.empty()) continue;
            size_t pos = req.output_text.find(s);
            if (pos != std::string::npos) {
                req.output_text.erase(pos);
                req.finish_reason = "stop";
                return true;
            }
        }

        // max_tokens bound (item 3): clean termination, reason reported.
        if (slot.tokens_generated >= req.max_tokens) {
            req.finish_reason = "length";
            return true;
        }
        return false;
    }

    void decode_step() {
        // Gather tokens from all active slots
        std::vector<int32_t> batch_tokens;
        std::vector<int> batch_slot_ids;

        for (int slot_id : active_slot_ids_) {
            batch_tokens.push_back(slots_[slot_id].last_token);
            batch_slot_ids.push_back(slot_id);
        }

        // Single batched forward pass - returns next token for each slot
        std::vector<int> next_tokens = batched_decode_(batch_tokens, batch_slot_ids);

        // Process results
        std::vector<int> slots_to_remove;

        for (size_t i = 0; i < batch_slot_ids.size(); ++i) {
            int slot_id = batch_slot_ids[i];
            auto& slot = slots_[slot_id];
            int next_token = next_tokens[i];

            // Check for cancellation
            if (slot.request->cancelled) {
                slot.request->finish_reason = "cancelled";
                slots_to_remove.push_back(slot_id);
                slot.request->token_queue->finish();
                stats_.requests_cancelled++;
                continue;
            }

            // Check for timeout
            auto elapsed = std::chrono::steady_clock::now() - slot.request->started_at;
            if (elapsed > config_.request_timeout) {
                slot.request->finish_reason = "timeout";
                slots_to_remove.push_back(slot_id);
                slot.request->token_queue->finish();
                stats_.requests_cancelled++;
                continue;
            }

            // Deliver token, fold into output_text, decide completion (eos / stop / max_tokens)
            slot.tokens_generated++;
            if (deliver_token(slot, next_token)) {
                slots_to_remove.push_back(slot_id);
                slot.request->token_queue->finish();
                stats_.requests_completed++;
            }
        }

        // Cleanup completed slots
        for (int slot_id : slots_to_remove) {
            if (clear_slot_) {
                clear_slot_(slot_id);
            }
            slots_[slot_id].reset();
            active_slot_ids_.erase(slot_id);
        }
        stats_.active_slots = active_slot_ids_.size();
    }

    Config config_;
    RequestQueue request_queue_;
    std::vector<Slot> slots_;
    std::set<int> active_slot_ids_;
    std::atomic<bool> running_{false};
    ServerStats stats_;

    // Callbacks
    TokenizeFunc tokenize_;
    DetokenizeFunc detokenize_;
    PrefillFunc prefill_;
    BatchedDecodeFunc batched_decode_;
    ClearSlotFunc clear_slot_;
    CloneSlotFunc clone_slot_;
    GetEosTokenFunc get_eos_token_;
};

}  // namespace qwenium