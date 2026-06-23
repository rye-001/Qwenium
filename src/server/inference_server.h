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
#include <iostream>
#include <algorithm>

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
        // Opt-in transparent warm-KV prefix reuse for multi-turn chat
        // (--chat-prefix-cache). When on, a finished slot's KV is RETAINED (not
        // cleared) and the next text request that starts with exactly that token
        // stream prefills only its suffix — token-identical to a cold prefill.
        // Default off: stateless, every prompt prefilled whole. See
        // docs/plan-chat-prefix-cache.md.
        bool chat_prefix_cache = false;
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

    InferenceServer(const Config& config)
        : config_(config), slots_(config.max_slots),
          slot_resident_tokens_(config.max_slots) {
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
        if (!slot_resident_tokens_[slot_id].empty()) {
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

    void assign_to_slot(std::shared_ptr<InferenceRequest> req) {
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
            // position 0 (no system-prompt prefix cache).
            first_token = prefill_(slot_id, tokens, /*start_pos=*/0);
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
};

}  // namespace qwenium