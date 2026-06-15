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
#include <cstdint>
#include <stdexcept>

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
    std::string prompt;
    int max_tokens = 256;
    // Sampling controls (per request → built into a per-slot sampler at assign
    // time). temperature 0 => greedy (deterministic argmax, byte-identical to the
    // pre-sampling server). temperature > 0 => TemperatureSampler with top_k/top_p
    // nucleus sampling; `seed` >= 0 makes that draw stream reproducible (OpenAI
    // `seed`), < 0 leaves it random_device-seeded.
    float temperature = 0.0f;  // 0 = greedy
    float top_p = 0.95f;       // nucleus cutoff (temperature path only)
    int   top_k = 40;          // top-k cutoff (temperature path only)
    long long seed = -1;       // >=0 => reproducible; <0 => non-deterministic
    // Optional GBNF grammar for constrained (structured) output. Empty => no
    // constraint. When set (text requests only), the slot's sampler masks logits
    // to grammar-valid tokens each step and generation ends when the grammar
    // reaches an accepting state. A parse failure is a fail-loud request error.
    std::string grammar;
    std::vector<std::string> stop;  // Optional: generation ends before first match
    bool skip_template = false;  // true => `prompt` is ALREADY fully templated
                                 // (e.g. a rendered <|im_start|> chat conversation
                                 // from /v1/chat/completions); tokenize it raw
                                 // instead of wrapping it as a single user turn.
                                 // Default false keeps /v1/completions unchanged.

    // Decoded image payloads (raw PNG/JPEG/… bytes). Empty for every text-only request, so
    // the text path is byte-identical. When non-empty, `prompt` has already had
    // the projector's image marker rendered into the turn, and the server routes
    // this request through the multimodal prefill callback (which tokenizes,
    // expands the marker into the soft-token span, encodes the image, splices the
    // soft tokens, and prefills). Only /v1/chat/completions populates this;
    // /v1/completions never does.
    std::vector<std::vector<uint8_t>> image_bytes;

    // Optional content-keyed prefix-cache hint (text path, server §1 decision A
    // follow-on). When non-empty, this is the ALREADY-TEMPLATED leading prefix
    // of `prompt` (the system-turn block) whose KV is a recurring, cacheable
    // unit — the route fills it from the system message so a recurring system
    // prompt skips its prefill on a cache HIT. Empty => no prefix caching (the
    // request prefills its whole prompt fresh, unchanged). Only consulted when
    // the server has a text prefix cache wired AND this is a text request
    // (image requests use the multimodal path instead). Transparent: a HIT is
    // byte-identical to a full re-prefill, version-gated fail-loud.
    std::string cacheable_prefix_text;

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
        int max_context = 0;        // Per-slot prompt-token ceiling; 0 = no limit (fail-loud guard)
        std::chrono::seconds request_timeout{60};
    };

    // Callback types for integration with your existing code
    using TokenizeFunc = std::function<std::vector<int32_t>(const std::string&)>;
    using DetokenizeFunc = std::function<std::string(int)>;
    using PrefillFunc = std::function<int(int slot_id, const std::vector<int32_t>& tokens, int start_pos)>;
    // Multimodal prefill: the image variant of PrefillFunc. The integration owns
    // the whole image-prefill flow — it tokenizes req.prompt (already image-marker
    // rendered), expands the marker into the soft-token span, encodes req.image_bytes
    // through the vision encoder, splices the soft tokens into the residual stream,
    // and prefills slot_id at start_pos. Returns the sampled first token and writes
    // the FULL expanded token stream (incl. the soft-token placeholders) back via
    // `out_tokens`, so the slot's context length / prompt_tokens / KV positions are
    // correct. Fail-loud: throws (named param) on a malformed image, a missing
    // capability, or an over-ceiling prompt; the server catches and surfaces it as
    // a named request error. Optional — when unset, image requests are rejected.
    using MultimodalPrefillFunc = std::function<int(
        int slot_id, const InferenceRequest& req, int start_pos,
        std::vector<int32_t>& out_tokens)>;
    // Cached text prefill: the prefix-cache variant of PrefillFunc (same shape
    // as MultimodalPrefillFunc — request-aware, writes the full token stream
    // back). The integration tokenizes req.prompt, splits off the cacheable
    // system-prefix (req.cacheable_prefix_text), restores/stores its KV via the
    // prefix library, and prefills only the variable suffix. Optional — used
    // only for a text request whose cacheable_prefix_text is non-empty when a
    // text prefix cache is wired; otherwise the plain PrefillFunc runs.
    using CachedTextPrefillFunc = MultimodalPrefillFunc;
    // Configure the per-slot control state (sampler, and later grammar) for a
    // freshly assigned request, BEFORE its prefill samples the first token. The
    // integration owns the slot→sampler mapping; the engine just signals "slot
    // slot_id now serves this request." Optional — unset leaves whatever default
    // the integration set up.
    using PrepareSlotFunc = std::function<void(int slot_id, const InferenceRequest& req)>;
    // True iff slot slot_id has reached its own self-contained completion (today:
    // a constrained-output grammar in an accepting state). Checked after each
    // delivered token, alongside the stop-token / stop-string / max_tokens
    // checks. Generic: the engine does not know WHY the slot is done. Optional —
    // unset means only the token/string/length checks apply.
    using SlotCompleteFunc = std::function<bool(int slot_id)>;
    using BatchedDecodeFunc = std::function<std::vector<int>(const std::vector<int32_t>& tokens,
                                                              const std::vector<int>& slot_ids)>;
    using ClearSlotFunc = std::function<void(int slot_id)>;
    // True iff `token` ends generation. The model has a SET of stop tokens
    // (primary EOS plus family end-of-turn markers, e.g. Gemma 4 IT's <turn|>),
    // not one — checking only the primary EOS lets a model emit its turn-ender,
    // have it ignored, and run past the turn boundary to the max_tokens cap.
    using IsStopTokenFunc = std::function<bool(int)>;

    InferenceServer(const Config& config) : config_(config), slots_(config.max_slots) {
        for (int i = 0; i < config.max_slots; ++i) {
            slots_[i].slot_id = i;
        }
    }

    // Wire up callbacks before calling start()
    void set_tokenize(TokenizeFunc fn) { tokenize_ = std::move(fn); }
    // Raw (no chat template) tokenizer for requests with skip_template=true.
    void set_raw_tokenize(TokenizeFunc fn) { raw_tokenize_ = std::move(fn); }
    void set_detokenize(DetokenizeFunc fn) { detokenize_ = std::move(fn); }
    void set_prefill(PrefillFunc fn) { prefill_ = std::move(fn); }
    // Optional: per-slot sampler/grammar setup, invoked once per request at
    // assignment (before prefill). Leave unset for a fixed default sampler.
    void set_prepare_slot(PrepareSlotFunc fn) { prepare_slot_ = std::move(fn); }
    // Optional: per-slot self-completion check (e.g. grammar accepting state).
    void set_slot_complete(SlotCompleteFunc fn) { slot_complete_ = std::move(fn); }
    // Optional: enables image input. Leave unset for a text-only server.
    void set_multimodal_prefill(MultimodalPrefillFunc fn) { multimodal_prefill_ = std::move(fn); }
    // Optional: enables content-keyed text prefix caching. Leave unset to always
    // prefill text prompts whole (the stateless default).
    void set_cached_text_prefill(CachedTextPrefillFunc fn) { cached_text_prefill_ = std::move(fn); }
    void set_batched_decode(BatchedDecodeFunc fn) { batched_decode_ = std::move(fn); }
    void set_clear_slot(ClearSlotFunc fn) { clear_slot_ = std::move(fn); }
    void set_is_stop_token(IsStopTokenFunc fn) { is_stop_token_ = std::move(fn); }

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

        // Configure this slot's sampler + grammar from the request BEFORE prefill
        // samples the first token, so the whole generation — first token included
        // — honors the request's temperature/top_p/seed and grammar. Fail-loud: a
        // bad GBNF grammar ends the request with a named error and no slot used.
        if (prepare_slot_) {
            try {
                prepare_slot_(slot_id, *req);
            } catch (const std::exception& e) {
                req->error_message = e.what();
                req->finish_reason = "error";
                req->token_queue->finish();  // slot not consumed
                return;
            }
        }

        std::vector<int32_t> tokens;
        int first_token = 0;

        if (!req->image_bytes.empty()) {
            // ── Image request: the integration owns the whole image-prefill flow
            // (tokenize → expand marker → encode → splice → prefill). It writes the
            // expanded token stream (with the soft-token span) back into `tokens`,
            // so slot bookkeeping below is unchanged. Decode after this is identical
            // text decode over the spliced KV cache.
            if (!multimodal_prefill_) {
                req->error_message =
                    "slot " + std::to_string(slot_id) +
                    ": parameter 'image': expected a vision-capable server "
                    "(started with --mmproj), actual: no multimodal prefill "
                    "configured (text-only server)";
                req->finish_reason = "error";
                req->token_queue->finish();  // slot not consumed
                return;
            }
            try {
                first_token = multimodal_prefill_(slot_id, *req, /*start_pos=*/0, tokens);
            } catch (const std::exception& e) {
                // Fail-loud: malformed image, capability mismatch, or over-ceiling
                // prompt. The callback guards before touching the slot KV, so the
                // slot is not consumed.
                req->error_message = e.what();
                req->finish_reason = "error";
                req->token_queue->finish();
                return;
            }
            req->prompt_tokens = static_cast<int>(tokens.size());
        } else if (cached_text_prefill_ && !req->cacheable_prefix_text.empty()) {
            // ── Text prefix-cache request: the integration owns the whole flow
            // (tokenize → split off the cacheable system-prefix → restore/store
            // its KV → prefill only the variable suffix). Like the image branch,
            // it does its own tokenize + oversize guard and writes the full token
            // stream back into `tokens`. Decode after this is identical. The
            // result is byte-identical to the plain prefill below (transparent);
            // a foreign/stale cached blob is refused fail-loud, never re-used.
            try {
                first_token = cached_text_prefill_(slot_id, *req, /*start_pos=*/0, tokens);
            } catch (const std::exception& e) {
                req->error_message = e.what();
                req->finish_reason = "error";
                req->token_queue->finish();  // slot not consumed
                return;
            }
            req->prompt_tokens = static_cast<int>(tokens.size());
        } else {
            // Tokenize prompt. skip_template requests (chat-completions, already
            // fully <|im_start|>-rendered) bypass the single-user-turn wrap.
            tokens = (req->skip_template && raw_tokenize_)
                ? raw_tokenize_(req->prompt)
                : tokenize_(req->prompt);
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

            // Stateless: every request prefills its full prompt fresh at
            // position 0 (no system-prompt prefix cache).
            first_token = prefill_(slot_id, tokens, /*start_pos=*/0);
        }

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

        if (is_stop_token_(token)) {
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

        // Grammar completion (item 4): the slot's grammar has reached an
        // accepting state after this token — the constrained output is complete,
        // so stop (mirrors the CLI's is_accepting_state() break). The grammar
        // was already advanced by accept_token in the prefill/decode callback,
        // so this reflects post-token state. No-op when no grammar is attached.
        if (slot_complete_ && slot_complete_(slot.slot_id)) {
            req.finish_reason = "stop";
            return true;
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
    TokenizeFunc raw_tokenize_;   // no-template variant (skip_template requests)
    DetokenizeFunc detokenize_;
    PrefillFunc prefill_;
    PrepareSlotFunc prepare_slot_;  // optional; per-slot sampler/grammar setup
    SlotCompleteFunc slot_complete_;  // optional; grammar accepting-state stop
    MultimodalPrefillFunc multimodal_prefill_;  // optional; image requests only
    CachedTextPrefillFunc cached_text_prefill_;  // optional; text prefix-cache path
    BatchedDecodeFunc batched_decode_;
    ClearSlotFunc clear_slot_;
    IsStopTokenFunc is_stop_token_;
};

}  // namespace qwenium