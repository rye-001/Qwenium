#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <vector>
#include <set>
#include <unordered_map>
#include <atomic>
#include <random>
#include <future>
#include <chrono>
#include <string>
#include <fstream>
#include <functional>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace qinf {

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

    // Conversation continuation handle (--conversational). INPUT: the client's
    // conversation_id, or empty for a new/stateless request. OUTPUT: on a CREATE
    // the engine writes the minted id here and the route returns it. A non-empty
    // id means "continue this conversation" — the engine appends the rendered
    // delta to the conversation's retained KV instead of prefilling whole. An
    // unknown/closed id is fail-loud (resend full history). Text path only.
    std::string conversation_id;

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
    // The stop token this turn ended on (a clean turn closer for conversational
    // continuation), or -1 if it ended on length / stop-string / grammar.
    int end_token = -1;
    // True when `last_token` was drafted/verified by a PRIOR speculative step
    // but never delivered — the engine's analogue of the CLI printing
    // next_token_id at the top of each loop iteration before feeding it
    // onward (cli/complete.cpp). decode_step_speculative() delivers it (via
    // deliver_token, same as every other path) before feeding it back through
    // the model. False for every non-speculative token: the batched path and
    // a slot's first token (activate_slot) always deliver immediately, so
    // last_token is already part of context_tokens by the time it's read.
    bool last_token_pending = false;
    // --token-log bookkeeping. prompt_n marks how much of context_tokens was
    // prompt at slot start (context_tokens may grow past it); generated_ids
    // accumulates exactly what this request emitted.
    size_t prompt_n = 0;
    std::vector<int32_t> generated_ids;

    void reset() {
        request.reset();
        context_tokens.clear();
        last_token = 0;
        tokens_generated = 0;
        active = false;
        end_token = -1;
        last_token_pending = false;
        prompt_n = 0;
        generated_ids.clear();
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
        // Opt-in transparent warm-KV prefix reuse for multi-turn chat
        // (--chat-prefix-cache). When on, a finished slot's KV is RETAINED (not
        // cleared) and the next text request that starts with exactly that token
        // stream prefills only its suffix — token-identical to a cold prefill.
        // Default off: stateless, every prompt prefilled whole. See
        // docs/plan-chat-prefix-cache.md.
        bool chat_prefix_cache = false;
        // Opt-in warm conversational server (--conversational). When on, a
        // request may carry a conversation_id: the server retains that
        // conversation's KV + token log across turns and appends only the new
        // rendered turn (chat.cpp-grade reasoning retention — warm != cold by
        // construction). Default off: §1 stateless stays the default and a
        // conversation_id is rejected fail-loud. See
        // docs/plan-warm-conversational-server.md.
        bool conversational = false;
        // Token-id request log (--token-log <path>). Empty = disabled, and
        // disabled is the DEFAULT: token ids are losslessly reversible to the
        // prompt text, so this is opt-in, not on by accident.
        //
        // Ids, not text, and that is the whole point: re-tokenizing logged text
        // yields a DIFFERENT sequence than the model actually saw (chat-template
        // rendering, special tokens, thinking scaffold), and every offline
        // analysis this log exists to serve -- draft hit rate / accepted length
        // for a speculative source, strict-prefix reuse for --chat-prefix-cache,
        // how much of max_tokens the thought channel spends -- depends on exact
        // token boundaries. One JSON object per completed request, appended.
        //
        // n_generated COUNTS THE STOP TOKEN; the OpenAI usage.completion_tokens
        // in the response does not. The log is deliberately the wider of the two
        // -- a replay needs the turn terminator -- so the two numbers differ by
        // one on a "stop" finish. Do not treat them as the same quantity.
        std::string token_log_path;
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
    // Current KV append position for a slot (= bos_count + tokens already
    // materialized). The warm chat-prefix path appends a turn's suffix here
    // rather than deriving the position from its own BOS-free token count, so a
    // model that prepends BOS at pos 0 (Gemma -it) stays consistent. Engine
    // truth; required only when chat_prefix_cache is on.
    using GetCachePosFunc = std::function<int(int slot_id)>;
    // True iff `token` ends generation. The model has a SET of stop tokens
    // (primary EOS plus family end-of-turn markers, e.g. Gemma 4 IT's <turn|>),
    // not one — checking only the primary EOS lets a model emit its turn-ender,
    // have it ignored, and run past the turn boundary to the max_tokens cap.
    using IsStopTokenFunc = std::function<bool(int)>;

    // ── Speculative decoding (--speculative, optional) ───────────────────────
    // The server's batch axis is SLOTS (one forward pass advances every active
    // slot); speculation's batch axis is DRAFT POSITIONS. They multiply, and
    // that multiplication is not affordable here (Gemma 4-12B: 4 verify lanes
    // = 1.41x a single step, 8 = 2.64x, 16 = 2.98x — cost grows faster than
    // acceptance with width; the Qwen hybrids fail-loud-abort a decode graph
    // above 10 slots on node count, and 5 slots x 4 drafts = 20 lanes would
    // walk straight into that). So: speculation engages ONLY when exactly one
    // slot is active; two or more falls back to the unchanged batched path.
    // See docs/architecture.md §6.

    // What one speculative decode step produced. `delivered_tokens` are
    // ALREADY model-verified (accepted draft tokens, plus the model's own eos
    // id appended when generation ended on it) — the engine delivers each via
    // deliver_token(), in order, stopping at the first one that completes the
    // slot (mirrors the CLI's per-accepted-token stop check, cli/complete.cpp).
    // `has_next`/`next_token`: when generation continues without terminating,
    // the trailing token is the model's own next-position prediction — already
    // computed by the same verify pass, but NOT YET fed through the model or
    // delivered. It becomes the slot's pending last_token for the FOLLOWING
    // step (Slot::last_token_pending), exactly mirroring the CLI deferring
    // next_token_id to the top of its next loop iteration.
    struct SpeculativeStepResult {
        std::vector<int32_t> delivered_tokens;
        bool    has_next   = false;
        int32_t next_token = -1;
    };
    // Run one speculative step for the single active slot: feed `last_token`
    // through the model, draft + verify a continuation in one batched pass,
    // and return every token it produced. `prompt_tokens` / `generated_tokens`
    // mirror the CLI's own two vectors (draft sources read them; `generated_tokens`
    // already includes `last_token` — the engine delivers it before calling in).
    // Reuses SpeculativeDecoder::try_speculative_step (sampling/speculative.h)
    // end to end — the same verify/rewind/checkpoint machinery the CLI uses,
    // not a second implementation.
    using SpeculativeDecodeFunc = std::function<SpeculativeStepResult(
        int slot_id, int32_t last_token,
        const std::vector<int32_t>& prompt_tokens,
        const std::vector<int32_t>& generated_tokens)>;
    // False => this slot must not speculate this step (today: a per-request
    // GBNF grammar is active — matches the CLI, which disables speculative
    // decoding outright when a grammar is set; draft tokens are never checked
    // against the grammar). Optional — unset means every slot is eligible.
    using SpeculativeEligibleFunc = std::function<bool(int slot_id)>;

    InferenceServer(const Config& config)
        : config_(config), slots_(config.max_slots),
          slot_resident_tokens_(config.max_slots) {
        slot_conversation_.resize(config.max_slots);
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
    // Optional: enables single-slot speculative decoding (--speculative). Unset
    // (the default) leaves decode_step() exactly as it was — the byte-identical
    // gate depends on that. Set together; speculative_eligible_ alone with no
    // decode fn is a no-op (decode_step() only branches on the decode fn).
    void set_speculative_decode(SpeculativeDecodeFunc fn) { speculative_decode_ = std::move(fn); }
    void set_speculative_eligible(SpeculativeEligibleFunc fn) { speculative_eligible_ = std::move(fn); }
    void set_clear_slot(ClearSlotFunc fn) { clear_slot_ = std::move(fn); }
    // Required when Config::chat_prefix_cache is on; harmless otherwise.
    void set_get_cache_pos(GetCachePosFunc fn) { get_cache_pos_ = std::move(fn); }
    void set_is_stop_token(IsStopTokenFunc fn) { is_stop_token_ = std::move(fn); }

    // Result of warm slot selection: the chosen slot and how many of its
    // retained tokens are reused (lcp). lcp > 0 ⇒ warm hit (append the suffix);
    // lcp == 0 ⇒ cold (prefill the whole prompt). slot_id == -1 ⇒ no free slot.
    struct WarmPick { int slot_id; size_t lcp; };

    // Prefix-aware slot routing (pure; unit-tested in test_inference_server.cpp).
    // Among the FREE slots (active[i] == 0), pick the one whose retained token
    // stream is the LONGEST strict prefix of `tokens` (resident non-empty,
    // shorter than tokens, and a byte-prefix of it). That slot's KV is reused as
    // a warm prefix. If none qualifies, return the first free slot with lcp 0
    // (cold). Strict-prefix-append ONLY: a shorter common prefix (divergence) is
    // never reused — rewinding would corrupt overwrite-semantics recurrent state
    // on hybrid recipes (CLAUDE.md KV-vs-recurrent invariant).
    static WarmPick warm_prefix_pick(
            const std::vector<std::vector<int32_t>>& residents,
            const std::vector<char>& active,
            const std::vector<int32_t>& tokens) {
        int best = -1, any_free = -1;
        size_t best_lcp = 0;
        for (size_t i = 0; i < residents.size(); ++i) {
            if (active[i]) continue;
            if (any_free < 0) any_free = static_cast<int>(i);
            const auto& res = residents[i];
            if (!res.empty() && res.size() < tokens.size() &&
                std::equal(res.begin(), res.end(), tokens.begin()) &&
                res.size() > best_lcp) {
                best = static_cast<int>(i);
                best_lcp = res.size();
            }
        }
        if (best >= 0) return {best, best_lcp};
        return {any_free, 0};
    }

    // ── Warm conversational server (--conversational) ────────────────────────
    // The reserved conversation_id a client sends to START a conversation; the
    // server replaces it with a real minted id in the response. An absent/empty
    // id => stateless (invariant 1: no handle, zero state). A real id => continue.
    static constexpr const char* kNewConversation = "new";

    // Routing decision for a request carrying a conversation_id — the
    // explicit-handle analog of warm_prefix_pick. See
    // docs/plan-warm-conversational-server.md.
    enum class ConvRoute {
        Stateless,          // no create/continue intent → existing stateless path
        Create,             // create sentinel → mint id, cold-prefill full at pos 0
        ContinueWarm,       // known id, KV resident → append delta at get_cache_pos
        ContinueRebuild,    // known id, KV evicted → prefill log++delta at pos 0
        FailUnknown,        // continue id not in registry → resend full history
        FailNotContinuable, // id known, last turn left no clean closer → resend full
        FailModeOff,        // create/continue asked but --conversational is off
    };

    // Classify a conversational request from registry state. Pure & unit-tested.
    //   mode_on         — Config::conversational
    //   wants_create    — request carries the kNewConversation create sentinel
    //   has_continue_id — request carries a real (non-sentinel) conversation_id
    //   id_known        — that id is in the registry
    //   kv_resident     — the conversation's slot still holds its KV (warm, idle)
    //   continuable     — the conversation's last turn ended on a clean closer
    // kv_resident / continuable are only meaningful when id_known.
    static ConvRoute classify_conversation_request(
            bool mode_on, bool wants_create, bool has_continue_id,
            bool id_known, bool kv_resident, bool continuable) {
        const bool wants_conv = wants_create || has_continue_id;
        if (!mode_on) return wants_conv ? ConvRoute::FailModeOff : ConvRoute::Stateless;
        if (wants_create) return ConvRoute::Create;
        if (!has_continue_id) return ConvRoute::Stateless;
        if (!id_known) return ConvRoute::FailUnknown;
        if (!continuable) return ConvRoute::FailNotContinuable;
        return kv_resident ? ConvRoute::ContinueWarm : ConvRoute::ContinueRebuild;
    }

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

    // Clear conversation state (called from HTTP threads). The registry + KV are
    // owned by the inference thread, so the op is marshaled onto it; the returned
    // future resolves to the number cleared (clear_conversation: 1, or -1 if the
    // id is unknown; flush: the count). See process_control_ops().
    std::future<int> clear_conversation(const std::string& id) {
        return enqueue_control(
            ControlOp{ControlOp::Type::ClearConversation, id, std::promise<int>()});
    }
    std::future<int> clear_all_conversations() {
        return enqueue_control(
            ControlOp{ControlOp::Type::ClearAll, std::string(), std::promise<int>()});
    }

    // Get stats (thread-safe)
    const ServerStats& stats() const { return stats_; }

    // Start inference loop (blocks - run in dedicated thread)
    // Backstop: this runs as a bare std::thread, so an escaping exception is
    // std::terminate — the whole server dies with no diagnosis. Individual
    // failure modes are contained at their own call sites (prefill callbacks,
    // batched decode); this catches anything that got past them, reports it, and
    // stops the loop cleanly instead of aborting the process.
    void run() {
        try {
            run_loop();
        } catch (const std::exception& e) {
            std::cerr << "[inference] fatal: " << e.what()
                      << "\n[inference] the inference loop has stopped; "
                         "restart the server" << std::endl;
            running_ = false;
        }
    }

    void run_loop() {
        running_ = true;
        while (running_) {
            // 0. Apply any queued clear/flush control ops (from HTTP threads).
            process_control_ops();

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

    // A plain text request eligible for warm chat-prefix reuse: the cache is on,
    // it carries no image (image KV splices placeholder ids — not token-prefix
    // reusable), and it is not routed to the disk system-prefix cache. The same
    // predicate gates slot routing (assign) and KV retention (release), so the
    // two never disagree about which slots are warm-managed.
    bool is_warm_text_request(const InferenceRequest& req) const {
        return config_.chat_prefix_cache && req.image_bytes.empty() &&
               !(cached_text_prefill_ && !req.cacheable_prefix_text.empty());
    }

    // On completion: retain this slot's KV as a warm prefix (warm text request)
    // or clear it (everything else — the stateless default). Retain records the
    // exact BOS-free token stream the slot materialized so the next turn can
    // reuse the longest matching prefix; the KV is left live and its position
    // pointer stays put. Reads slot.context_tokens — call BEFORE slot.reset().
    void release_slot_kv(int slot_id) {
        Slot& slot = slots_[slot_id];
        // Conversational slot (--conversational): snapshot the turn into the
        // conversation log + closer and KEEP the KV live + bound (warm for the
        // next turn) instead of the transparent retain/clear below.
        if (!slot_conversation_[slot_id].empty()) {
            release_conversation_turn(slot_id);
            return;
        }
        if (slot.request && is_warm_text_request(*slot.request)) {
            // Retain EXACTLY the tokens the KV materialized. context_tokens holds
            // the full prompt + every generated token, but the FINAL sampled
            // token is never fed back into the KV (generation stops on it), and a
            // leading BOS the prefill prepends is not in context_tokens. So the
            // materialized BOS-free length is get_cache_pos - bos; truncate to it
            // to keep the invariant get_cache_pos == bos + resident.size(). A
            // resident longer than the KV would make the next turn's suffix
            // prefill start one position late (a skipped token = corruption).
            const int mat = get_cache_pos_
                ? get_cache_pos_(slot_id) : static_cast<int>(slot.context_tokens.size());
            const int bos = kv_bos_count_ >= 0 ? kv_bos_count_ : 0;
            int keep = mat - bos;
            if (keep < 0) keep = 0;
            if (keep > static_cast<int>(slot.context_tokens.size()))
                keep = static_cast<int>(slot.context_tokens.size());
            slot_resident_tokens_[slot_id].assign(
                slot.context_tokens.begin(), slot.context_tokens.begin() + keep);
        } else if (clear_slot_) {
            clear_slot_(slot_id);
        }
    }

    // A pos-0 prefill (image / disk-cache / cold text) needs a clean slot. If a
    // previous warm turn retained KV here, drop it so the new prefill starts at
    // position 0 with no stale prefix underneath.
    void clear_retained_if_any(int slot_id) {
        // A conversation may own this slot's KV; unbind it (its log survives in
        // the registry → it falls back to ContinueRebuild) before the pos-0
        // prefill steals the slot.
        const bool bound = !slot_conversation_[slot_id].empty();
        if (bound) {
            auto it = conversations_.find(slot_conversation_[slot_id]);
            if (it != conversations_.end()) it->second.slot = -1;
            slot_conversation_[slot_id].clear();
        }
        if (bound || !slot_resident_tokens_[slot_id].empty()) {
            if (clear_slot_) clear_slot_(slot_id);
            slot_resident_tokens_[slot_id].clear();
        }
    }

    // Install a freshly-prefilled request into its slot and deliver the first
    // token; on immediate completion (EOS / stop / max_tokens == 1) release the
    // slot. `tokens` is the FULL prompt stream the slot's KV now materializes
    // (warm or cold), so resident bookkeeping and usage counts are correct.
    // Shared tail of every prefill branch in assign_to_slot.
    void activate_slot(int slot_id, std::shared_ptr<InferenceRequest> req,
                       std::vector<int32_t>&& tokens, int first_token) {
        auto& slot = slots_[slot_id];
        slot.request = req;
        slot.context_tokens = std::move(tokens);
        slot.prompt_n = slot.context_tokens.size();
        slot.last_token = first_token;
        slot.tokens_generated = 1;
        slot.active = true;
        active_slot_ids_.insert(slot_id);
        stats_.active_slots = active_slot_ids_.size();

        if (deliver_token(slot, first_token)) {
            log_slot_end(slot, "first-token");
            req->token_queue->finish();
            stats_.requests_completed++;
            release_slot_kv(slot_id);
            slots_[slot_id].reset();
            active_slot_ids_.erase(slot_id);
            stats_.active_slots = active_slot_ids_.size();
        }
    }

    // Snapshot the active flag of every slot (for warm_prefix_pick, which is
    // decoupled from Slot so it stays pure and unit-testable).
    std::vector<char> slot_active_flags() const {
        std::vector<char> a(slots_.size());
        for (size_t i = 0; i < slots_.size(); ++i) a[i] = slots_[i].active ? 1 : 0;
        return a;
    }

    // ── Warm conversational server: execution (--conversational) ─────────────
    // Fail a conversational request fail-loud: a named error, queue finished,
    // slot NOT consumed. Void so callers can `return fail_request(...)`.
    void fail_request(const std::shared_ptr<InferenceRequest>& req,
                      const std::string& message) {
        req->error_message = message;
        req->finish_reason = "error";
        req->token_queue->finish();
    }

    // Opaque conversation id, "conv_"-prefixed so it never collides with the
    // kNewConversation create sentinel. Minted on the inference thread only.
    std::string mint_conversation_id() {
        return "conv_" + std::to_string(conv_rng_()) + "_" +
               std::to_string(conv_rng_());
    }

    // Snapshot a completed conversational turn into its log + closer, KEEPING the
    // slot's KV live and bound (warm for the next turn). Same off-by-one as
    // release_slot_kv (resident == get_cache_pos - bos). Reads context_tokens, so
    // it runs BEFORE slot.reset().
    void release_conversation_turn(int slot_id) {
        Slot& slot = slots_[slot_id];
        auto it = conversations_.find(slot_conversation_[slot_id]);
        if (it == conversations_.end()) return;  // shouldn't happen
        Conversation& c = it->second;
        const int mat = get_cache_pos_ ? get_cache_pos_(slot_id)
                                       : static_cast<int>(slot.context_tokens.size());
        const int bos = kv_bos_count_ >= 0 ? kv_bos_count_ : 0;
        int keep = mat - bos;
        if (keep < 0) keep = 0;
        if (keep > static_cast<int>(slot.context_tokens.size()))
            keep = static_cast<int>(slot.context_tokens.size());
        c.log.assign(slot.context_tokens.begin(),
                     slot.context_tokens.begin() + keep);
        c.last_closer = slot.end_token;  // clean closer (stop token) or -1
        c.slot = slot_id;                // stays bound; KV not cleared
        c.last_used = std::chrono::steady_clock::now();
    }

    // A clear/flush op marshaled from an HTTP thread onto the inference thread;
    // its result resolves the caller's future (clear: 1 / -1 unknown; flush: count).
    struct ControlOp {
        enum class Type { ClearConversation, ClearAll } type;
        std::string id;
        std::promise<int> result;
    };

    // Enqueue a control op (HTTP thread) and return the future its result lands on.
    std::future<int> enqueue_control(ControlOp op) {
        auto f = op.result.get_future();
        { std::lock_guard<std::mutex> lock(control_mutex_); control_queue_.push(std::move(op)); }
        return f;
    }

    // Drain queued clear ops (inference thread, top of the run loop). Safe to
    // touch the registry + KV here — this thread solely owns them.
    void process_control_ops() {
        std::queue<ControlOp> ops;
        { std::lock_guard<std::mutex> lock(control_mutex_); std::swap(ops, control_queue_); }
        while (!ops.empty()) {
            ControlOp op = std::move(ops.front()); ops.pop();
            const int r = (op.type == ControlOp::Type::ClearAll)
                ? do_clear_all_conversations() : do_clear_conversation(op.id);
            op.result.set_value(r);
        }
    }

    // Drop one conversation: its registry entry, its slot binding, and (if the
    // slot is idle) its KV. If a turn is mid-flight on the slot, only unbind —
    // that generation's release becomes a no-op and the KV is cleared when the
    // slot is next reused. Returns 1, or -1 if the id is unknown.
    int do_clear_conversation(const std::string& id) {
        auto it = conversations_.find(id);
        if (it == conversations_.end()) return -1;
        const int slot = it->second.slot;
        if (slot >= 0) {
            if (!slots_[slot].active && clear_slot_) clear_slot_(slot);
            slot_conversation_[slot].clear();
        }
        conversations_.erase(it);
        return 1;
    }

    // Flush every conversation; returns the count cleared.
    int do_clear_all_conversations() {
        const int n = static_cast<int>(conversations_.size());
        for (int i = 0; i < config_.max_slots; ++i) {
            if (slot_conversation_[i].empty()) continue;
            if (!slots_[i].active && clear_slot_) clear_slot_(i);
            slot_conversation_[i].clear();
        }
        conversations_.clear();
        return n;
    }

    // Choose a slot for a Create / Rebuild (pos-0) prefill. Prefer a truly idle,
    // clean slot; else sacrifice an idle transparent resident; else evict the LRU
    // idle conversation (its log survives → ContinueRebuild later). The chosen
    // slot is unbound here; the caller's pos-0 prefill clears its KV. Returns -1
    // only if every slot is active (the run loop assigns only when one is idle).
    int pick_slot_for_conversation() {
        int clean = -1, transparent = -1, lru_conv = -1;
        std::chrono::steady_clock::time_point lru_time =
            std::chrono::steady_clock::time_point::max();
        for (int i = 0; i < config_.max_slots; ++i) {
            if (slots_[i].active) continue;
            const bool bound = !slot_conversation_[i].empty();
            if (!bound && slot_resident_tokens_[i].empty()) { clean = i; break; }
            if (!bound) { if (transparent < 0) transparent = i; continue; }
            auto it = conversations_.find(slot_conversation_[i]);
            const auto t = (it != conversations_.end())
                ? it->second.last_used
                : std::chrono::steady_clock::time_point::min();
            if (t < lru_time) { lru_time = t; lru_conv = i; }
        }
        const int chosen = clean >= 0 ? clean
                         : (transparent >= 0 ? transparent : lru_conv);
        if (chosen < 0) return -1;
        if (!slot_conversation_[chosen].empty()) {  // evict the prior owner
            auto it = conversations_.find(slot_conversation_[chosen]);
            if (it != conversations_.end()) it->second.slot = -1;
            slot_conversation_[chosen].clear();
        }
        slot_resident_tokens_[chosen].clear();
        return chosen;
    }

    // The conversational counterpart of assign_to_slot's text path: route a
    // request carrying a conversation_id through the registry (create / continue-
    // warm / rebuild / fail). Reuses prefill_ (append at get_cache_pos for warm;
    // pos 0 for create/rebuild) and activate_slot. Text only — caller gates on no
    // image. See docs/plan-warm-conversational-server.md.
    void assign_conversational_slot(std::shared_ptr<InferenceRequest> req) {
        const std::string cid = req->conversation_id;
        const bool wants_create = (cid == kNewConversation);
        const bool has_continue_id = !cid.empty() && !wants_create;

        Conversation* conv = nullptr;
        bool id_known = false, kv_resident = false, continuable = false;
        if (has_continue_id) {
            auto it = conversations_.find(cid);
            if (it != conversations_.end()) {
                conv = &it->second;
                id_known = true;
                kv_resident = (conv->slot >= 0 && !slots_[conv->slot].active);
                continuable = (conv->last_closer >= 0);
            }
        }

        const ConvRoute route = classify_conversation_request(
            config_.conversational, wants_create, has_continue_id,
            id_known, kv_resident, continuable);
        switch (route) {
            case ConvRoute::FailModeOff:
                return fail_request(req, "field 'conversation_id': conversational "
                    "mode is not enabled on this server (start it with "
                    "--conversational)");
            case ConvRoute::FailUnknown:
                return fail_request(req, "field 'conversation_id': unknown "
                    "conversation '" + cid + "'; resend the full history with no "
                    "conversation_id to start a new one");
            case ConvRoute::FailNotContinuable:
                return fail_request(req, "field 'conversation_id': conversation '" +
                    cid + "' did not end on a turn boundary (last turn hit a length "
                    "or stop limit); resend the full history to continue");
            case ConvRoute::Stateless:
                return;  // unreachable: callers delegate only non-empty ids
            case ConvRoute::Create:
            case ConvRoute::ContinueWarm:
            case ConvRoute::ContinueRebuild:
                break;
        }
        const bool warm = (route == ConvRoute::ContinueWarm);

        // Render → this turn's tokens (chat-completions is pre-templated).
        std::vector<int32_t> turn_tokens = (req->skip_template && raw_tokenize_)
            ? raw_tokenize_(req->prompt) : tokenize_(req->prompt);

        const int slot_id = warm ? conv->slot : pick_slot_for_conversation();
        if (slot_id < 0) return;  // every slot active (run loop shouldn't allow it)

        req->started_at = std::chrono::steady_clock::now();
        if (prepare_slot_) {
            try { prepare_slot_(slot_id, *req); }
            catch (const std::exception& e) { return fail_request(req, e.what()); }
        }

        // Build the prefill stream (fed now) and the full BOS-free stream the KV
        // will hold. Warm appends [closer]+turn at the engine position; create and
        // rebuild prefill the whole thing at pos 0.
        std::vector<int32_t> prefill_tokens, tokens;
        int prefill_pos = 0;
        if (warm) {
            if (!get_cache_pos_)
                return fail_request(req, "slot " + std::to_string(slot_id) +
                    ": '--conversational': expected a get_cache_pos callback, "
                    "actual: none (server wiring error)");
            if (conv->last_closer >= 0) prefill_tokens.push_back(conv->last_closer);
            prefill_tokens.insert(prefill_tokens.end(),
                                  turn_tokens.begin(), turn_tokens.end());
            prefill_pos = get_cache_pos_(slot_id);
            tokens = conv->log;
            tokens.insert(tokens.end(), prefill_tokens.begin(), prefill_tokens.end());
        } else {  // Create or ContinueRebuild
            if (conv) {  // Rebuild: replay the log + its closer, then this turn
                tokens = conv->log;
                if (conv->last_closer >= 0) tokens.push_back(conv->last_closer);
            }
            tokens.insert(tokens.end(), turn_tokens.begin(), turn_tokens.end());
            prefill_tokens = tokens;
            prefill_pos = 0;
        }

        req->prompt_tokens = static_cast<int>(tokens.size());
        if (config_.max_context > 0 && req->prompt_tokens > config_.max_context)
            return fail_request(req, "slot " + std::to_string(slot_id) +
                ": prompt too large; expected: <= " +
                std::to_string(config_.max_context) + " tokens, actual: " +
                std::to_string(req->prompt_tokens));

        if (prefill_pos == 0 && clear_slot_) clear_slot_(slot_id);  // reset KV+recurrent
        const int first_token = prefill_(slot_id, prefill_tokens, prefill_pos);
        if (prefill_pos == 0 && get_cache_pos_ && kv_bos_count_ < 0)
            kv_bos_count_ = get_cache_pos_(slot_id) -
                            static_cast<int>(prefill_tokens.size());

        // Bind the slot to the conversation (create mints + returns the id).
        const std::string id = wants_create ? mint_conversation_id() : cid;
        if (wants_create) req->conversation_id = id;  // OUTPUT: returned to client
        Conversation& centry = conversations_[id];
        centry.slot = slot_id;
        centry.last_used = std::chrono::steady_clock::now();
        slot_conversation_[slot_id] = id;

        activate_slot(slot_id, req, std::move(tokens), first_token);
    }

    void assign_to_slot(std::shared_ptr<InferenceRequest> req) {
        // Warm conversational server: a request carrying a conversation_id routes
        // through the registry (create/continue/fail), NOT the transparent
        // chat-prefix path. An empty id => the stateless/transparent path below
        // (invariant 1). Image requests ignore conversation_id in v1 (text only).
        if (req->image_bytes.empty() && !req->conversation_id.empty()) {
            assign_conversational_slot(std::move(req));
            return;
        }

        // Warm chat-prefix eligibility (a plain text request, cache on). Drives
        // both prefix-aware slot routing and KV retention. A warm request is
        // tokenized BEFORE slot choice so it can route to the slot whose
        // retained KV is the longest prefix of it.
        const bool warm_text = is_warm_text_request(*req);
        std::vector<int32_t> warm_tokens;
        size_t warm_lcp = 0;
        int slot_id;
        if (warm_text) {
            warm_tokens = (req->skip_template && raw_tokenize_)
                ? raw_tokenize_(req->prompt) : tokenize_(req->prompt);
            const WarmPick pick = warm_prefix_pick(
                slot_resident_tokens_, slot_active_flags(), warm_tokens);
            slot_id = pick.slot_id;
            warm_lcp = pick.lcp;
        } else {
            slot_id = find_free_slot();
        }
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
            // A previous warm turn may have retained KV on this slot; an image
            // prefill writes at pos 0, so drop it first (chat-cache off ⇒ no-op).
            clear_retained_if_any(slot_id);
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
            clear_retained_if_any(slot_id);  // pos-0 prefill needs a clean slot
            try {
                first_token = cached_text_prefill_(slot_id, *req, /*start_pos=*/0, tokens);
            } catch (const std::exception& e) {
                req->error_message = e.what();
                req->finish_reason = "error";
                req->token_queue->finish();  // slot not consumed
                return;
            }
            req->prompt_tokens = static_cast<int>(tokens.size());
        } else if (warm_text) {
            // ── Warm chat-prefix request (--chat-prefix-cache). Already
            // tokenized above for routing. On a HIT the matching slot's KV
            // already holds tokens[0,warm_lcp); we append only the suffix at the
            // engine's current position (which already counts any leading BOS),
            // so the result is token-identical to a cold prefill. On a MISS we
            // drop any stale retained KV and prefill the whole prompt at 0. The
            // FULL token stream is recorded as the slot's resident prefix on
            // completion (release_slot_kv), so the next turn can reuse it.
            tokens = std::move(warm_tokens);
            req->prompt_tokens = static_cast<int>(tokens.size());
            if (config_.max_context > 0 && req->prompt_tokens > config_.max_context) {
                req->error_message =
                    "slot " + std::to_string(slot_id) +
                    ": prompt too large; expected: <= " + std::to_string(config_.max_context) +
                    " tokens, actual: " + std::to_string(req->prompt_tokens);
                req->finish_reason = "error";
                req->token_queue->finish();  // slot not consumed
                return;
            }
            if (warm_lcp > 0) {
                if (!get_cache_pos_) {
                    req->error_message =
                        "slot " + std::to_string(slot_id) +
                        ": '--chat-prefix-cache': expected a get_cache_pos "
                        "callback, actual: none (server wiring error)";
                    req->finish_reason = "error";
                    req->token_queue->finish();  // slot not consumed
                    return;
                }
                const int pos = get_cache_pos_(slot_id);
                const std::vector<int32_t> suffix(
                    tokens.begin() + static_cast<long>(warm_lcp), tokens.end());
                first_token = prefill_(slot_id, suffix, pos);
                std::cout << "[chat-cache] HIT slot=" << slot_id << ": reused "
                          << warm_lcp << " tokens, prefilled " << suffix.size()
                          << " suffix at pos " << pos << std::endl;
            } else {
                clear_retained_if_any(slot_id);  // drop stale KV; prefill at 0
                first_token = prefill_(slot_id, tokens, /*start_pos=*/0);
                // Observe the model's leading-BOS count once: after a pos-0
                // prefill of N tokens the KV is at bos + N (the callback prepends
                // BOS for Gemma -it). Needed to retain exactly the materialized KV.
                if (get_cache_pos_ && kv_bos_count_ < 0)
                    kv_bos_count_ = get_cache_pos_(slot_id) - static_cast<int>(tokens.size());
                std::cout << "[chat-cache] MISS slot=" << slot_id << ": prefilled "
                          << tokens.size() << " tokens cold" << std::endl;
            }
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
            // position 0 (no system-prompt prefix cache). Drop any retained KV /
            // conversation binding first (no-op unless a cache/mode left some) so
            // the pos-0 prefill — and hybrid recurrent state — starts clean.
            clear_retained_if_any(slot_id);
            // Wrapped like every sibling prefill above (multimodal :1004,
            // cached-text :1025). It was NOT, and that was a real defect: this is
            // the ordinary /v1/completions path, and prefill_ genuinely throws
            // here — the oversize-prompt guard and any engine fail-loud both
            // surface as exceptions. Unwrapped, such a throw escaped
            // assign_to_slot, so the request was never activated, its TokenQueue
            // was never finish()ed, and the HTTP thread blocked forever in
            // pop_blocking: the client hung instead of getting an error. Found
            // 2026-09-02 while testing a slot-hygiene assertion on this path.
            try {
                first_token = prefill_(slot_id, tokens, /*start_pos=*/0);
            } catch (const std::exception& e) {
                // Leave the SLOT recoverable, not just the server. A prefill can
                // fail part-way with KV already written, and nothing else clears
                // it: the slot is never activated, so no release fires, and every
                // later request routed here would fail on the same debris —
                // one failure would poison the slot for the process lifetime.
                if (clear_slot_) clear_slot_(slot_id);
                req->error_message = e.what();
                req->finish_reason = "error";
                req->token_queue->finish();  // slot not consumed
                return;
            }
        }

        activate_slot(slot_id, req, std::move(tokens), first_token);
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
        if (!config_.token_log_path.empty()) slot.generated_ids.push_back(token);

        if (is_stop_token_(token)) {
            req.finish_reason = "stop";
            slot.end_token = token;  // clean turn closer for conversational continue
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

    // DEBUG: one line at every generation-end boundary, naming the reason and
    // the numbers that tell them apart — elapsed≈timeout ⇒ timed out;
    // gen==max_tokens ⇒ length; reason="stop" mid-sentence ⇒ an EOS/stop-seq
    // hit. If the server dies WITHOUT printing this first, the decode path threw.
    void log_slot_end(const Slot& slot, const char* where) {
        const auto& req = *slot.request;
        const double elapsed_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - req.started_at).count();
        std::cout << "[server-stop] slot=" << slot.slot_id
                  << " reason=" << req.finish_reason
                  << " where=" << where
                  << " gen=" << slot.tokens_generated
                  << " max_tokens=" << req.max_tokens
                  << " elapsed=" << elapsed_s << "s"
                  << " timeout=" << config_.request_timeout.count() << "s"
                  << std::endl;
        write_token_log(slot);
    }

    // One JSON object per completed request, appended to config_.token_log_path.
    // Called from log_slot_end, which every completion path already converges on,
    // so no exit route can silently skip it. Fail-loud on an unopenable path:
    // a logging flag that silently does nothing is worse than no flag.
    void write_token_log(const Slot& slot) {
        if (config_.token_log_path.empty()) return;
        std::ofstream out(config_.token_log_path, std::ios::app);
        if (!out) {
            throw std::runtime_error(
                "write_token_log: token_log_path expected an appendable file, actual '"
                + config_.token_log_path + "' (slot " + std::to_string(slot.slot_id) + ")");
        }
        const auto& req = *slot.request;
        const size_t pn = std::min(slot.prompt_n, slot.context_tokens.size());
        out << "{\"slot\":" << slot.slot_id
            << ",\"finish_reason\":\"" << req.finish_reason << "\""
            << ",\"n_prompt\":" << pn
            << ",\"n_generated\":" << slot.generated_ids.size()
            << ",\"max_tokens\":" << req.max_tokens
            << ",\"grammar\":" << (req.grammar.empty() ? "false" : "true")
            << ",\"prompt\":[";
        for (size_t i = 0; i < pn; ++i) out << (i ? "," : "") << slot.context_tokens[i];
        out << "],\"generated\":[";
        for (size_t i = 0; i < slot.generated_ids.size(); ++i)
            out << (i ? "," : "") << slot.generated_ids[i];
        out << "]}\n";
    }

    void decode_step() {
        // Speculative decoding (--speculative): engage ONLY when exactly one
        // slot is active. speculative_decode_ unset (the default) leaves this
        // whole branch dead — decode_step() falls straight through to the
        // unchanged batched path below, which is what the byte-identical
        // gate (speculation OFF) depends on. See the SpeculativeDecodeFunc
        // comment above for why >1 slot is never attempted.
        if (speculative_decode_) {
            if (active_slot_ids_.size() == 1) {
                const int slot_id = *active_slot_ids_.begin();
                const bool eligible =
                    !speculative_eligible_ || speculative_eligible_(slot_id);
                if (eligible) {
                    spec_fallback_logged_ = false;  // re-arm the edge-triggered log
                    decode_step_speculative(slot_id);
                    return;
                }
                log_speculative_fallback(
                    "grammar active on slot " + std::to_string(slot_id));
            } else {
                log_speculative_fallback(
                    std::to_string(active_slot_ids_.size()) +
                    " active slots (>1) — speculation's draft-position batch "
                    "axis does not compose with the slot batch axis");
            }
        }

        // Gather tokens from all active slots
        std::vector<int32_t> batch_tokens;
        std::vector<int> batch_slot_ids;

        for (int slot_id : active_slot_ids_) {
            batch_tokens.push_back(slots_[slot_id].last_token);
            batch_slot_ids.push_back(slot_id);
        }

        // Single batched forward pass - returns next token for each slot.
        //
        // A throw here is a backend compute failure (engine/graph_compute.h) —
        // in practice a Metal command-buffer failure, usually GPU OOM. It is not
        // per-request bad input: the pass computed nothing, so EVERY slot in this
        // batch is affected and none of them has a next token. Fail them all with
        // the named error rather than let the exception escape run(), which is
        // launched as a bare std::thread and would std::terminate the server.
        //
        // Metal latches this state, so subsequent batches will fail the same way.
        // That is deliberate and honest: every client gets a clear error until the
        // process is restarted, instead of one wrong answer followed by more.
        std::vector<int> next_tokens;
        try {
            next_tokens = batched_decode_(batch_tokens, batch_slot_ids);
        } catch (const std::exception& e) {
            for (int slot_id : batch_slot_ids) {
                auto& slot = slots_[slot_id];
                if (slot.request) {
                    slot.request->error_message = e.what();
                    slot.request->finish_reason = "error";
                    slot.request->token_queue->finish();
                }
                // Same teardown the normal completion path uses.
                release_slot_kv(slot_id);
                slots_[slot_id].reset();
                active_slot_ids_.erase(slot_id);
            }
            stats_.active_slots = active_slot_ids_.size();
            return;
        }

        // Process results
        std::vector<int> slots_to_remove;

        for (size_t i = 0; i < batch_slot_ids.size(); ++i) {
            int slot_id = batch_slot_ids[i];
            auto& slot = slots_[slot_id];
            int next_token = next_tokens[i];

            // Check for cancellation
            if (slot.request->cancelled) {
                slot.request->finish_reason = "cancelled";
                log_slot_end(slot, "cancelled");
                slots_to_remove.push_back(slot_id);
                slot.request->token_queue->finish();
                stats_.requests_cancelled++;
                continue;
            }

            // Check for timeout
            auto elapsed = std::chrono::steady_clock::now() - slot.request->started_at;
            if (elapsed > config_.request_timeout) {
                slot.request->finish_reason = "timeout";
                log_slot_end(slot, "timeout");
                slots_to_remove.push_back(slot_id);
                slot.request->token_queue->finish();
                stats_.requests_cancelled++;
                continue;
            }

            // Deliver token, fold into output_text, decide completion (eos / stop / max_tokens)
            slot.tokens_generated++;
            if (deliver_token(slot, next_token)) {
                log_slot_end(slot, "deliver");
                slots_to_remove.push_back(slot_id);
                slot.request->token_queue->finish();
                stats_.requests_completed++;
            }
        }

        // Cleanup completed slots: retain the KV as a warm prefix (chat-cache,
        // warm text request) or clear it (default). Reads slot state, so it runs
        // BEFORE reset().
        for (int slot_id : slots_to_remove) {
            release_slot_kv(slot_id);
            slots_[slot_id].reset();
            active_slot_ids_.erase(slot_id);
        }
        stats_.active_slots = active_slot_ids_.size();
    }

    // Edge-triggered operator log: prints once when the engine ENTERS a
    // fallback stretch (was speculating, now isn't — or never got to start),
    // and stays silent while it continues, however many steps that takes.
    // decode_step() clears the latch the moment speculation resumes, so a
    // later re-entry logs again. This is a policy choice (batch axes don't
    // compose past one slot, or a grammar request), not an error: the
    // request is served correctly by the batched path either way.
    void log_speculative_fallback(const std::string& reason) {
        if (spec_fallback_logged_) return;
        std::cout << "[speculative] fallback to batched decode: " << reason
                  << std::endl;
        spec_fallback_logged_ = true;
    }

    // One speculative decode step for the single active slot (decode_step()
    // has already confirmed eligibility). Delivers 0..K+1 tokens through the
    // same deliver_token() every other path uses, so termination (stop token,
    // stop string, grammar, max_tokens) is decided identically regardless of
    // how many tokens a step produces.
    void decode_step_speculative(int slot_id) {
        Slot& slot = slots_[slot_id];
        auto teardown = [this, slot_id]() {
            release_slot_kv(slot_id);
            slots_[slot_id].reset();
            active_slot_ids_.erase(slot_id);
            stats_.active_slots = active_slot_ids_.size();
        };

        // A token drafted/verified by the PRIOR speculative step but not yet
        // delivered must be delivered now — see Slot::last_token_pending.
        // Cleared unconditionally: whether or not it turns out to be a stop
        // token, this step is done deciding its fate.
        if (slot.last_token_pending) {
            slot.last_token_pending = false;
            slot.tokens_generated++;
            if (deliver_token(slot, slot.last_token)) {
                log_slot_end(slot, "deliver-speculative");
                slot.request->token_queue->finish();
                teardown();
                stats_.requests_completed++;
                return;
            }
        }

        // Draft sources read the same two vectors the CLI passes them
        // (sampling/draft_source.h DraftContext): the original prompt, and
        // everything generated since — INCLUDING the token just delivered
        // above, which is what this step is about to feed through the model
        // and draft a continuation from.
        std::vector<int32_t> prompt_tokens(
            slot.context_tokens.begin(),
            slot.context_tokens.begin() + slot.request->prompt_tokens);
        std::vector<int32_t> generated_tokens(
            slot.context_tokens.begin() + slot.request->prompt_tokens,
            slot.context_tokens.end());

        SpeculativeStepResult result;
        try {
            result = speculative_decode_(slot_id, slot.last_token,
                                         prompt_tokens, generated_tokens);
        } catch (const std::exception& e) {
            // Same fail-loud contract as the batched path's compute failure:
            // this slot's request gets a named error, the slot is released.
            slot.request->error_message = e.what();
            slot.request->finish_reason = "error";
            slot.request->token_queue->finish();
            teardown();
            return;
        }

        // Cancellation / timeout: checked once per step, same contract as
        // the batched path. A step here can now cover several tokens (up to
        // the draft width), so a cancelled/timed-out request may discard
        // that many already-computed-but-undelivered tokens instead of at
        // most one — the compute already happened either way (the batched
        // path also discards one already-computed token on the same check);
        // this bounds the discard at the draft width, not an unbounded
        // amount. Detection itself is not delayed: still checked every step.
        if (slot.request->cancelled) {
            slot.request->finish_reason = "cancelled";
            log_slot_end(slot, "cancelled");
            slot.request->token_queue->finish();
            stats_.requests_cancelled++;
            teardown();
            return;
        }
        auto elapsed = std::chrono::steady_clock::now() - slot.request->started_at;
        if (elapsed > config_.request_timeout) {
            slot.request->finish_reason = "timeout";
            log_slot_end(slot, "timeout");
            slot.request->token_queue->finish();
            stats_.requests_cancelled++;
            teardown();
            return;
        }

        for (int32_t tok : result.delivered_tokens) {
            slot.tokens_generated++;
            if (deliver_token(slot, tok)) {
                log_slot_end(slot, "deliver-speculative");
                slot.request->token_queue->finish();
                teardown();
                stats_.requests_completed++;
                return;
            }
        }

        if (result.has_next) {
            slot.last_token = result.next_token;
            slot.last_token_pending = true;
        }
        // !has_next with no delivered token having completed the slot would
        // mean the integration returned a result that neither continues nor
        // terminates — a wiring bug in the SpeculativeDecodeFunc, not a
        // request-level condition. try_speculative_step's contract (an
        // attempted draft always yields either an accepted prefix ending on
        // eos, or a bonus token to continue with) rules this out; nothing
        // else to do here.
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
    GetCachePosFunc get_cache_pos_;   // engine KV append position (chat prefix cache)
    IsStopTokenFunc is_stop_token_;
    SpeculativeDecodeFunc speculative_decode_;      // optional; --speculative
    SpeculativeEligibleFunc speculative_eligible_;  // optional; false => grammar active
    // Edge-triggered latch for log_speculative_fallback: true once the
    // operator has been told about the CURRENT fallback stretch, reset the
    // moment speculation resumes. Inference-thread-only, like everything else
    // decode_step() touches.
    bool spec_fallback_logged_ = false;

    // Warm chat-prefix cache (Config::chat_prefix_cache). Per slot: the exact
    // BOS-free token stream the slot's KV currently materializes ([0,pos) minus
    // any leading BOS), or empty if the slot's KV is clear. Retained across
    // requests so the next turn of a conversation reuses the longest matching
    // prefix. Never consulted while a slot is active.
    std::vector<std::vector<int32_t>> slot_resident_tokens_;
    // Leading-BOS count the prefill prepends at pos 0 (0 for Qwen, 1 for Gemma
    // -it). Model-global, so captured once at the first cold prefill:
    // get_cache_pos - prefilled_token_count. Used to retain EXACTLY the
    // materialized KV (resident) so a turn's suffix appends without a position
    // skip. -1 = not yet observed.
    int kv_bos_count_ = -1;

    // ── Warm conversational server state (Config::conversational) ────────────
    // Separate from slot_resident_tokens_ (the transparent handle-free path):
    // different lifecycle (explicit handle, chat.cpp reasoning-retention,
    // warm != cold), so NOT unified (CLAUDE.md parameterize-or-split). A
    // conversation is an opaque minted id owning a token log + a slot binding.
    struct Conversation {
        // BOS-free materialized token stream of every COMPLETED turn so far
        // (scaffolds retained), EXCLUDING the last turn's closer (see
        // last_closer). prefill(log, 0) rebuilds the conversation's KV exactly.
        std::vector<int32_t> log;
        // Slot currently holding this conversation's warm KV, or -1 if evicted
        // (the next turn rebuilds from log). A warm hit appends the delta here.
        int slot = -1;
        // The stop token the last turn ended on (the assistant turn's
        // end-of-turn marker), re-fed as the FIRST delta token next turn so the
        // KV closes the prior turn before the new one. -1 = the last turn had no
        // clean closer (length / stop-string) → not delta-continuable, the next
        // continue fails loud and the client resends full history.
        int last_closer = -1;
        std::chrono::steady_clock::time_point last_used;
    };
    std::unordered_map<std::string, Conversation> conversations_;
    // Which conversation owns each slot's retained KV (empty = none /
    // transparent-managed). Lets a slot-stealing request evict the prior owner.
    std::vector<std::string> slot_conversation_;
    // RNG for opaque conversation ids (inference-thread only).
    std::mt19937_64 conv_rng_{std::random_device{}()};

    // Control-op queue (ControlOp defined above) marshaling clear/flush from HTTP
    // threads onto the inference thread, which solely owns the registry + KV.
    // Drained in process_control_ops() at the top of the run loop.
    std::queue<ControlOp> control_queue_;
    std::mutex control_mutex_;
};

}  // namespace qinf