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
#include "core/model.h"
#include "models/model_registry.h"
#include "loader/tokenizer.h"
#include "loader/chat_template.h"
#include "loader/channel_filter.h"
#include "sampling/sampling.h"
#include "sampling/grammar_vocab.h"  // GBNF constrained output (per-slot grammar)

// Vision (image input on /v1/chat/completions). The whole image pipeline lives
// in ServerVision (server_vision.{h,cpp}) and is inert unless the server is
// started with --mmproj: a text-only server never constructs one, and a
// text-only request never touches it. http_server.cpp keeps only the data-URI
// decode + capability gate it needs to route an image request.
#include "image_data_uri.h"
#include "server_vision.h"

// Attention Lens extraction (--attention-lens): the dedicated /v1/extract
// endpoint. Document + complete key vocabulary → audited key-value JSON on the
// attention trust layer. Inert unless the flag is set; the OpenAI endpoints are
// untouched (docs/plan-qemmi-lens.md P2/A2).
#include "server_lens.h"

// Text prefix cache (server §1 decision A follow-on): the shipped, transparent,
// content-keyed L2 PrefixLibrary wired into the TEXT prefill path — a recurring
// system-prompt block skips its prefill on a HIT (mirrors the vision V2 move).
#include "core/prefix_library.h"
#include "core/slot_snapshot.h"

#include <iostream>
#include <thread>
#include <csignal>
#include <atomic>
#include <ctime>
#include <cstdint>
#include <optional>
#include <algorithm>
#include <cstdlib>

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
    
    // Remove end-of-turn / EOS markers. Qwen (<|im_end|>) and Gemma
    // (<end_of_turn>, <eos>) markers are both listed — they are mutually
    // exclusive in practice (a model emits only its own), so the union is
    // safe cross-family and keeps this free function model-agnostic. (F6)
    std::vector<std::string> end_tokens = {
        "<|im_end|>", "<|endoftext|>", "</s>",
        "<end_of_turn>", "<eos>",
        // Gemma 4 IT's native end-of-turn marker. It IS a stop token (so the
        // slot halts before folding it into output_text), but the raw id is
        // still pushed to the SSE stream before the stop fires — strip it so the
        // streamed delta doesn't leak the literal marker.
        "<turn|>"};
    for (const auto& token : end_tokens) {
        pos = normalized.find(token);
        if (pos != std::string::npos) {
            normalized = normalized.substr(0, pos);
        }
    }

    // NB: Gemma 4 channel framing (the <|channel>/<channel|> thought channel) is
    // NOT stripped here. This function runs PER TOKEN (it backs detokenize_), and
    // the ChannelFilter is a state machine that spans tokens — running it
    // per-token resets its state every call, so the <|channel> marker is consumed
    // but the following "thought" name + content leak through on the next token.
    // Channel stripping is therefore applied at the RESPONSE BOUNDARY instead:
    // one-shot ChannelFilter::strip over the full assembled output_text for
    // non-streaming responses, and a single stateful ChannelFilter::feed() per
    // request for SSE streaming (mirroring the CLI). The BPE char fixups and
    // end-marker removal above ARE per-token-safe and stay here.
    return normalized;
}

// =============================================================================
// Integration layer: Wire up InferenceServer to your Qwen implementation
// =============================================================================
class QweniumServerIntegration {
    static constexpr int MAX_SLOTS = 10;

public:
    QweniumServerIntegration(const std::string& model_path, int max_ctx_per_slot = 2048,
                          int max_slots = MAX_SLOTS,
                          const std::string& mmproj_path = "",
                          const std::string& image_embed_cache_dir = "",
                          const std::string& image_prefix_cache_dir = "",
                          const std::string& prefix_cache_dir = "",
                          bool kv_f16 = false)
        : max_ctx_per_slot_(max_ctx_per_slot),
          max_slots_(max_slots < 1 ? 1 : (max_slots > MAX_SLOTS ? MAX_SLOTS : max_slots)),
          kv_type_(kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32) {

        std::cout << "Loading model from: " << model_path << std::endl;
        // Register architectures BEFORE load_metadata: the GGUF loader validates
        // the model's architecture against the registered allow-list, so the
        // registry must be populated first or every load fails with an empty
        // "expected one of:" list.
        register_builtin_models();
        // --mmproj signals the vision pipeline: allow a checkpoint carrying
        // image-placeholder tokens instead of refusing it as multimodal-only
        // (mirrors the CLI; harmless for a text export, which still loads as text).
        model_.load_metadata(model_path, /*allow_multimodal=*/!mmproj_path.empty());
        model_.load_tensors();

        // F6: build the tokenizer with the architecture's registered
        // TokenizerConfig (NOT a config-less default). For Gemma this carries the
        // SPM normalizer, byte-fallback, BOS, and the <start_of_turn>/<end_of_turn>
        // specials; without it the server byte-explodes Gemma's control tokens
        // into garbage. Mirrors core/model.cpp's tokenizer init.
        const std::string& arch = model_.get_metadata().architecture;
        tokenizer_ = std::make_unique<Tokenizer>(&model_.get_metadata(),
                                                 lookup_tokenizer_config(arch));
        // One sampler + (optional) grammar per slot — control state is per-slot,
        // like KV. Default each sampler to greedy; prepare_slot() rebuilds a
        // slot's sampler/grammar from its request when that request is assigned.
        slot_samplers_.resize(max_slots_);
        for (auto& s : slot_samplers_)
            s = std::make_unique<qwenium::GreedySampler>();
        slot_grammars_.resize(max_slots_);  // null unless a request sets a grammar

        // Cache the vocabulary once (id -> token string). Passed as token_strs to
        // sample() so a slot's grammar can mask logits, and used to build each
        // grammar slot's TokenTrie. ~vocab_size strings; fetched a single time.
        vocab_ = tokenizer_->get_vocabulary();

        // F6: render prompts with the architecture's registered ChatTemplate
        // (Qwen <|im_start|> vs Gemma <start_of_turn> vs Gemma-4's own markers),
        // not a hardwired Qwen string.
        chat_template_ = lookup_chat_template(arch);
        if (!chat_template_)
            throw std::runtime_error(
                "QweniumServerIntegration: no chat template registered for arch '" +
                arch + "'");
        std::cout << "Tokenizer + chat template wired for arch=" << arch << std::endl;

        // Initialize forward pass with max_slots_ slots (KV cache is sized
        // ctx × slots × kv_type_, so fewer slots frees context headroom — the
        // right trade for one-request-at-a-time delegation). --kv-f16 halves
        // that term; recurrent state is unaffected.
        const auto& srv_meta = model_.get_metadata();
        std::cout << "KV cache type: "
                  << (kv_type_ == GGML_TYPE_F16 ? "F16 (--kv-f16)" : "F32 (default)")
                  << std::endl;
        forward_pass_ = create_forward_pass(
            model_, &srv_meta, max_ctx_per_slot_, max_slots_, kv_type_);
        
        scheduler_ = model_.get_scheduler();
        
        // Reserve memory for the maximum batch size to prevent reallocation errors
        reserve_max_batch();

        // Vision projector (image input). Stays null for a text-only server; a
        // successful construction is what `vision_enabled()` reports. Fail-loud
        // ctor (no image hook / missing markers) propagates out of here.
        if (!mmproj_path.empty()) {
            vision_ = std::make_unique<ServerVision>(
                model_, *forward_pass_, *tokenizer_, model_mutex_,
                max_ctx_per_slot_, mmproj_path, image_embed_cache_dir,
                image_prefix_cache_dir);
        }

        // Text prefix cache (--prefix-cache): opt-in, version-gated, transparent.
        // Requires a recipe that exposes its KV cache(s); refuse loudly at setup
        // if not (opt-in explicit, the F9 rule). Null => no text prefix caching.
        if (!prefix_cache_dir.empty()) {
            if (forward_pass_->snapshot_kv_caches().empty())
                throw std::runtime_error(
                    "QweniumServerIntegration: parameter '--prefix-cache': expected a "
                    "recipe that exposes its KV cache(s) (snapshot_kv_caches "
                    "non-empty), actual: a recipe without L2 snapshot support (" +
                    model_label() + ")");
            text_prefix_lib_ = std::make_unique<PrefixLibrary>(
                prefix_cache_dir,
                qinf::snapshot::make_snapshot_header(
                    model_.get_metadata(), forward_pass_->snapshot_kv_caches()));
            std::cout << "Text: prefix cache '" << prefix_cache_dir
                      << "' (skip recurring system-prompt prefill)" << std::endl;
        }

        std::cout << "Model loaded successfully" << std::endl;
    }

    void reserve_max_batch() {
        std::cout << "Reserving memory for max batch size: " << max_slots_ << " and max ctx: " << max_ctx_per_slot_ << std::endl;

        // 1. Reserve for max decode batch — at MAXIMUM KV depth.
        //
        // build_decoding_graph sizes its growing per-step inputs (gather_indices
        // and the per-window attention masks) from get_cache_pos(slot), i.e. the
        // CURRENT KV depth — NOT the `positions` argument. Reserving with a fresh
        // cache (depth 0 → n_kv_len 1) therefore reserves the SMALLEST decode
        // graph, so galloc must reallocate as the real conversation deepens. That
        // mid-generation reallocation corrupts decode scratch → non-deterministic
        // logits and premature mid-word stops on long generations. Advance every
        // slot's cache to the context ceiling first so the reserved graph is the
        // largest one decode will ever build; galloc is then sized once and never
        // grows during a request. Positions/values are irrelevant here — only the
        // graph topology/sizes matter for reservation — so the dummy 0s are fine.
        {
            const int32_t max_depth =
                max_ctx_per_slot_ > 0 ? max_ctx_per_slot_ - 1 : 0;

            std::vector<int32_t>  tokens(max_slots_, 0);             // Dummy tokens
            std::vector<uint32_t> slot_ids(max_slots_);             // 0..max_slots_-1
            // Positions MUST equal the cache depth: build_decoding_graph sizes
            // gather_indices from get_cache_pos, while build_batched_attention
            // derives n_kv_len from this `positions` argument. Real decode keeps
            // them equal (positions = get_cache_pos); the reserve must too, or the
            // gather and the attention reshape disagree (GGML_ASSERT in reshape).
            std::vector<int32_t>  positions(max_slots_, max_depth);

            for (int i = 0; i < max_slots_; ++i) {
                slot_ids[i] = i;
            }

            if (max_depth > 0) {
                for (int i = 0; i < max_slots_; ++i) {
                    forward_pass_->advance_cache(static_cast<uint32_t>(max_depth), i);
                }
            }

            ggml_backend_sched_reset(scheduler_);

            // Build the graph for the maximum possible workload (deepest KV)
            ggml_cgraph* gf = forward_pass_->build_decoding_graph(tokens, slot_ids, positions);

            // Reserve memory in the scheduler
            if (!ggml_backend_sched_reserve(scheduler_, gf)) {
                std::cerr << "WARNING: Failed to reserve memory for max decode batch!" << std::endl;
            }

            // Restore the caches to empty — the reserve only needed the topology.
            for (int i = 0; i < max_slots_; ++i) {
                forward_pass_->clear_slot(i);
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

    // ── Attention Lens (--attention-lens) ────────────────────────────────────
    // Opt-in; arms the /v1/extract route. Inert otherwise — no lens code runs and
    // the engine is byte-inert. Builds NO grammar: the lens decodes free (Stage 2,
    // docs/note-nogrammar-refutation.md). This does not touch the per-request
    // `grammar` field on /v1/completions and /v1/chat/completions, which is a
    // separate shipped feature and still works exactly as before.
    void enable_attention_lens() {
        attention_lens_enabled_ = true;
        std::cout << "Attention Lens ON (--attention-lens): POST /v1/extract "
                     "(single-slot; document → audited key-value JSON; free decode, "
                     "tolerant parse, 422 on unparseable output)" << std::endl;
    }
    bool attention_lens_enabled() const { return attention_lens_enabled_; }

    // Run one extraction and return the lens-format JSON. EXCLUSIVE: holds the
    // model lock for the whole tapped decode and uses slot 0 (the only slot with
    // a correct qwen36 decode KV gather — architecture.md §12 / plan A3), so it
    // serializes against the inference thread's batched steps. Single-slot V1:
    // do not drive concurrent OpenAI traffic on slot 0 while extracting.
    //
    // Throws std::runtime_error on bad input (empty concepts, oversized doc) ⇒ the
    // route maps that to 400; qwenium::LensUnparseableError when the MODEL's output
    // holds no parseable object ⇒ 422. Those are different events and the route
    // must keep them apart (docs/lens-format.md §"The shape contract").
    std::string extract_lens_json(const std::string& document,
                                  const std::vector<qwenium::LensConcept>& concepts,
                                  int max_new_tokens) {
        std::lock_guard<std::mutex> lock(model_mutex_);
        qwenium::LensExtractOptions opts;
        opts.max_new_tokens = max_new_tokens;
        // One prefill, one pass, no grammar. Absent concepts come back
        // value:null/badge:"absent" by omission — the model declines natively
        // (30/30 on Leg C) once nothing forces it to fill every key.
        qwenium::LensReport rep = qwenium::run_lens_extract(
            forward_pass_.get(), scheduler_, tokenizer_.get(), model_.get_metadata(),
            (uint32_t)vocab_.size(), (uint32_t)max_ctx_per_slot_, document, concepts, opts);
        return qwenium::lens_report_to_json(rep);
    }

    void configure_server(qwenium::InferenceServer& server) {
        server.set_tokenize([this](const std::string& text) {
            // Model-aware single user-turn template (F6). System prompt is
            // handled separately via caching.
            return tokenizer_->encode(wrap_user_turn(text));
        });

        server.set_raw_tokenize([this](const std::string& text) {
            // No chat template: the /v1/chat/completions route has already
            // rendered the full <|im_start|> conversation. Encode verbatim.
            return tokenizer_->encode(text);
        });

        server.set_detokenize([this](int token_id) {
            std::string text = tokenizer_->decode(token_id);
            return normalize_output(text);
        });

        server.set_prefill([this](int slot_id, const std::vector<int32_t>& tokens, int start_pos) {
            return run_prefill(slot_id, tokens, start_pos);
        });

        // Build this slot's sampler (+ grammar) from the request before prefill.
        server.set_prepare_slot([this](int slot_id, const qwenium::InferenceRequest& req) {
            prepare_slot(slot_id, req);
        });

        // A slot with a constrained-output grammar is done once that grammar
        // reaches an accepting state (the engine stops it after that token).
        server.set_slot_complete([this](int slot_id) {
            return slot_grammars_[slot_id] != nullptr &&
                   slot_grammars_[slot_id]->is_accepting_state();
        });

        // Image input: only registered when a vision projector is loaded, so a
        // text-only server leaves this callback null and rejects image requests.
        // Delegated wholesale to ServerVision.
        if (vision_) {
            server.set_multimodal_prefill(
                [this](int slot_id, const qwenium::InferenceRequest& req, int start_pos,
                       std::vector<int32_t>& out_tokens) {
                    return vision_->run_multimodal_prefill(slot_id, req, start_pos,
                                                           out_tokens, *slot_samplers_[slot_id]);
                });
        }

        // Text prefix cache: only registered when --prefix-cache is set, so a
        // server without it always prefills text prompts whole (the stateless
        // default). The engine routes a text request here only when its
        // cacheable_prefix_text is non-empty.
        if (text_prefix_lib_) {
            server.set_cached_text_prefill(
                [this](int slot_id, const qwenium::InferenceRequest& req, int start_pos,
                       std::vector<int32_t>& out_tokens) {
                    return run_cached_text_prefill(slot_id, req, start_pos, out_tokens);
                });
        }

        server.set_batched_decode([this](const std::vector<int32_t>& tokens,
                                         const std::vector<int>& slot_ids) {
            return run_batched_decode(tokens, slot_ids);
        });

        server.set_clear_slot([this](int slot_id) {
            forward_pass_->clear_slot(slot_id);
        });

        // Engine KV append position for a slot (= bos_count + materialized
        // tokens). The warm chat-prefix path (--chat-prefix-cache) appends a
        // turn's suffix here, so the leading BOS this integration prepends at
        // pos 0 is accounted for without the (engine-agnostic) server reasoning
        // about it. Wired unconditionally; consulted only when the cache is on.
        server.set_get_cache_pos([this](int slot_id) {
            return static_cast<int>(forward_pass_->get_cache_pos(slot_id));
        });

        // Stop on ANY of the model's end tokens, not just the primary EOS. The
        // loader collects the family's end-of-turn markers (e.g. Gemma 4 IT's
        // <turn|>, Qwen's <|im_end|>) into stop_token_ids; checking only
        // eos_token_id let the model emit its turn-ender, have it ignored, and
        // run past the turn boundary to the max_tokens cap (runaway generation).
        server.set_is_stop_token([this](int token_id) {
            const auto& ids = model_.get_metadata().stop_token_ids;
            return std::find(ids.begin(), ids.end(), token_id) != ids.end();
        });
    }
    
    int max_ctx_per_slot() const { return max_ctx_per_slot_; }
    int max_slots() const { return max_slots_; }
    const ChatTemplate* chat_template() const { return chat_template_; }

    // ── Vision (image input) capability surface, read by the chat route ──────
    // All delegate to ServerVision; the accessors below image_marker_prefix /
    // image_wants_thinking are only called after vision_enabled() gates true.
    bool vision_enabled() const { return vision_ != nullptr; }
    // The projector's image-marker string the route prepends to the user turn
    // (e.g. "\n\n<start_of_image>\n\n" / "<|image>").
    const std::string& image_marker_prefix() const { return vision_->image_marker_prefix(); }
    // True when the image path must use the thinking branch (Gemma 4): a leading
    // system <|think|> turn + a generation prompt ending at "model\n". Gemma 3
    // keeps its no-think image path. See docs/server-image-multirequest-bug.md §5.
    bool image_wants_thinking() const { return vision_->image_wants_thinking(); }
    size_t max_image_bytes() const { return qwenium::kDefaultMaxImageBytes; }
    // Human-readable identity for the fail-loud capability-gate error: the
    // loaded model's architecture + name.
    std::string model_label() const {
        const auto& m = model_.get_metadata();
        return "arch='" + m.architecture + "', name='" + m.model_name + "'";
    }

    // ── F6: model-aware chat templating via the registered ChatTemplate ──────
    // A single user turn = a 1-message history with the assistant prompt opened.
    std::string wrap_user_turn(const std::string& text) const {
        return chat_template_->render({ChatMessage{"user", text}},
                                      /*add_assistant_prompt=*/true);
    }
    // A full (system, user) turn rendered together into one prompt — the
    // stateless replacement for the deleted system-prompt prefix cache. The
    // ChatTemplate maps the system role per family (e.g. Gemma → user turn).
    std::string render_system_user_turn(const std::string& system, const std::string& user) const {
        return chat_template_->render({ChatMessage{"system", system}, ChatMessage{"user", user}},
                                      /*add_assistant_prompt=*/true);
    }

    // ── Text prefix cache capability surface, read by both routes ────────────
    // True when --prefix-cache is wired. The route only populates a request's
    // cacheable_prefix_text when this holds (no cost otherwise).
    bool text_prefix_cache_enabled() const { return text_prefix_lib_ != nullptr; }
    // Render JUST the leading system-turn block (no user turn, NO assistant
    // prompt) — the cacheable prefix. It is a token-prefix of the matching
    // render_system_user_turn / render_chat output because every family delimits
    // each message as its own special-token-bounded turn. `enable_thinking` must
    // match the full render's so the prefix stays byte-aligned (Gemma 4 injects
    // <|think|> into the system turn when thinking is on).
    std::string render_system_prefix(const std::string& system,
                                     std::optional<bool> enable_thinking = std::nullopt) const {
        return chat_template_->render({ChatMessage{"system", system}},
                                      /*add_assistant_prompt=*/false, enable_thinking);
    }

private:
    // Build the per-slot sampler for a freshly assigned request. temperature 0
    // (or negative) => GreedySampler, a true argmax — which is what
    // inference_server.h has always promised this path does. (It did not until
    // GreedySampler's 1.2 repetition-penalty default was removed; a
    // temperature-0 request is now genuinely the model's argmax, and no longer
    // matches the pre-2026-08 server byte-for-byte.) temperature > 0 =>
    // TemperatureSampler honoring
    // top_p/top_k; a non-negative `seed` makes the draw stream reproducible.
    // Runs on the inference thread before this slot's prefill (no model_mutex_
    // needed: pure object construction, single-threaded with all sampling).
    void prepare_slot(int slot_id, const qwenium::InferenceRequest& req) {
        // Sampler from temperature/top_p/seed.
        std::unique_ptr<qwenium::Sampler> sampler;
        if (req.temperature > 0.0f) {
            auto ts = std::make_unique<qwenium::TemperatureSampler>(
                req.temperature, /*repetition_penalty=*/1.1f,
                /*repetition_lookback=*/64, req.top_k, req.top_p);
            if (req.seed >= 0)
                ts->seed(static_cast<uint32_t>(req.seed));
            sampler = std::move(ts);
        } else {
            sampler = std::make_unique<qwenium::GreedySampler>();
        }

        // Optional GBNF grammar for constrained output (text requests only).
        // Fail-loud (caught by the engine → named request error): a bad GBNF, or
        // the unsupported grammar+image combo, rejects the request cleanly.
        std::unique_ptr<qwenium::GrammarVocab> grammar;
        if (!req.grammar.empty()) {
            if (!req.image_bytes.empty())
                throw std::runtime_error(
                    "slot " + std::to_string(slot_id) + ": parameter 'grammar': "
                    "structured output is not supported with image input (v1)");
            grammar = qwenium::GrammarVocab::parse_impl(req.grammar);
            if (!grammar)
                throw std::runtime_error(
                    "slot " + std::to_string(slot_id) + ": parameter 'grammar': "
                    "failed to parse GBNF grammar");
            sampler->set_grammar(grammar.get());
            sampler->build_token_trie(vocab_);  // trie + cached vocab for accept_token
            // Let the grammar terminate: an accepting state adds these to the
            // valid set so the model may emit a stop token.
            for (int32_t id : model_.get_metadata().stop_token_ids)
                sampler->add_eos_token_id(id);
        }

        slot_samplers_[slot_id] = std::move(sampler);
        slot_grammars_[slot_id] = std::move(grammar);  // null clears any prior grammar
    }

    int run_prefill(int slot_id, const std::vector<int32_t>& tokens, int start_pos) {
        std::lock_guard<std::mutex> lock(model_mutex_);

        // Gemma-family -it models go DEGENERATE (greedy repeats one token) when
        // the sequence does not start with BOS — and encode() does not prepend
        // it. Honor the model's add_bos_token contract at this single text-
        // prefill entry point (both /v1/completions and /v1/chat/completions
        // route here), gated on start_pos==0 so a user turn appended after a
        // cached system prompt is not given a SECOND BOS. Mirrors the CLI
        // (cli/complete.cpp) and the image path (run_multimodal_prefill). Qwen
        // (add_bos_token=false) is unaffected.
        const auto& md = model_.get_metadata();
        const bool prepend_bos =
            start_pos == 0 && md.add_bos_token && md.bos_token_id >= 0;
        std::vector<int32_t> with_bos;
        if (prepend_bos) {
            with_bos.reserve(tokens.size() + 1);
            with_bos.push_back(md.bos_token_id);
            with_bos.insert(with_bos.end(), tokens.begin(), tokens.end());
        }
        const std::vector<int32_t>& seq = prepend_bos ? with_bos : tokens;

        ggml_backend_sched_reset(scheduler_);

        // Use build_prefill_graph with slot_id
        ggml_cgraph* gf = forward_pass_->build_prefill_graph(seq, start_pos, slot_id);
        ggml_backend_sched_alloc_graph(scheduler_, gf);
        forward_pass_->set_prefill_inputs(gf, seq, start_pos);
        ggml_backend_sched_graph_compute(scheduler_, gf);
        forward_pass_->advance_cache(seq.size(), slot_id);

        // Sample first token
        std::vector<float> logits = forward_pass_->get_output_logits(gf);
        size_t vocab_size = md.vocab_size;
        std::vector<float> last_token_logits(logits.end() - vocab_size, logits.end());

        std::vector<int32_t> context(seq);
        // Pass the vocab so a slot grammar can mask logits (no-op for non-grammar
        // slots — sample() requires grammar_ AND non-empty token_strs, so this
        // stays byte-identical to greedy). accept_token advances the grammar
        // cursor (no-op when no grammar) so the next step and the accepting-state
        // stop check see this token.
        const int tok = slot_samplers_[slot_id]->sample(last_token_logits, context, vocab_);
        slot_samplers_[slot_id]->accept_token(tok);
        return tok;
    }

    // Text prefix-cache prefill (the CachedTextPrefillFunc body). Splits the
    // prompt into [cacheable system-prefix | variable suffix]; on a cache HIT the
    // prefix KV is restored (its prefill skipped) and only the suffix is
    // prefilled; on a MISS the prefix is prefilled, captured, and stored. Either
    // way the suffix prefill yields the first-token logits, so the result is
    // byte-identical to a full cold prefill (transparent). Mirrors the vision V2
    // image-prefix split and the CLI --prefix-cache path. Fail-loud: a foreign /
    // stale cached blob is refused (never silently re-used — the F9 rule).
    int run_cached_text_prefill(int slot_id, const qwenium::InferenceRequest& req,
                                int start_pos, std::vector<int32_t>& out_tokens) {
        std::lock_guard<std::mutex> lock(model_mutex_);
        const auto& md = model_.get_metadata();

        // Tokenize the full (already-templated) prompt and the cacheable prefix.
        // Both get the same BOS treatment at pos 0 (Gemma -it needs it; mirrors
        // run_prefill), so the prefix stays a true token-prefix of the full seq.
        const bool prepend_bos =
            start_pos == 0 && md.add_bos_token && md.bos_token_id >= 0;
        auto encode_seq = [&](const std::string& text) {
            std::vector<int32_t> ids = tokenizer_->encode(text);
            if (prepend_bos) ids.insert(ids.begin(), md.bos_token_id);
            return ids;
        };
        std::vector<int32_t> full   = encode_seq(req.prompt);
        std::vector<int32_t> prefix = encode_seq(req.cacheable_prefix_text);
        out_tokens = full;

        // Fail-loud ceiling guard (same shape as the text path), BEFORE touching
        // the KV cache — an over-ceiling prompt would overflow it.
        if (max_ctx_per_slot_ > 0 &&
            static_cast<int>(full.size()) > max_ctx_per_slot_)
            throw std::runtime_error(
                "slot " + std::to_string(slot_id) + ": prompt too large; expected: "
                "<= " + std::to_string(max_ctx_per_slot_) + " tokens, actual: " +
                std::to_string(full.size()));

        const uint32_t slot = static_cast<uint32_t>(slot_id);
        ggml_backend_sched_t sched = model_.get_scheduler();
        std::vector<float> logits;

        // Self-protecting alignment check: the prefix must be a proper token-
        // prefix of the full seq (it is, for every current family — each turn is
        // special-token delimited). If a future template ever fused turns it
        // would not be, and restoring its KV would corrupt the result; so fall
        // back to a plain full prefill (transparent no-op, never a wrong restore).
        const bool aligned =
            !prefix.empty() && prefix.size() < full.size() &&
            std::equal(prefix.begin(), prefix.end(), full.begin());
        if (!aligned) {
            logits = forward_pass_->run_prefill(full, start_pos, slot, sched);
        } else {
            const std::vector<int32_t> suffix(full.begin() + prefix.size(), full.end());
            const int suffix_pos = start_pos + static_cast<int>(prefix.size());
            const uint64_t pkey = PrefixLibrary::key_for(prefix);
            const auto header = qinf::snapshot::make_snapshot_header(
                md, forward_pass_->snapshot_kv_caches());

            std::vector<uint8_t> blob;
            bool hit = false;
            try {
                hit = text_prefix_lib_->try_load(pkey, blob);
            } catch (const std::exception& e) {
                throw std::runtime_error(
                    "slot " + std::to_string(slot_id) + ": '--prefix-cache': a "
                    "stored blob for this system prefix was built under a different "
                    "model / quant / backend and is refused (" + e.what() +
                    "). Clear or re-point the prefix cache dir.");
            }
            if (hit) {
                qinf::snapshot::restore_slot(*forward_pass_, slot, blob, header);
                std::cout << "[prefix-cache] HIT: skipped prefill of "
                          << prefix.size() << " system tokens" << std::endl;
            } else {
                forward_pass_->run_prefill(prefix, start_pos, slot, sched);
                text_prefix_lib_->store(
                    pkey, qinf::snapshot::capture_slot(*forward_pass_, slot, header));
                std::cout << "[prefix-cache] MISS: prefilled + stored "
                          << prefix.size() << " system tokens" << std::endl;
            }
            // The variable suffix rides the plain text path and yields logits.
            logits = forward_pass_->run_prefill(suffix, suffix_pos, slot, sched);
        }

        const size_t vocab_size = md.vocab_size;
        std::vector<float> last_token_logits(logits.end() - vocab_size, logits.end());
        std::vector<int32_t> context(out_tokens);
        const int tok = slot_samplers_[slot_id]->sample(last_token_logits, context, vocab_);
        slot_samplers_[slot_id]->accept_token(tok);  // advance grammar (no-op if none)
        return tok;
    }

    // TODO: migrate to decode_step (src/core/decode_step.h) once the server
    // gains per-slot grammar support.  Today this path uses GreedySampler with
    // no grammar, so the sparse LM head can never fire; migrating now would add
    // dead code.  The batched nature (n > 1 slots per call) also needs a
    // batch-aware variant of decode_step that doesn't exist yet.
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
            // Rope position, not the KV row count (they diverge after an image).
            positions.push_back(forward_pass_->get_rope_pos(slot_id));
        }

        ggml_cgraph* gf = forward_pass_->build_decoding_graph(tokens, slot_ids_u32, positions);
        ggml_backend_sched_alloc_graph(scheduler_, gf);
        forward_pass_->set_decode_inputs(gf, tokens, slot_ids_u32, positions);
        ggml_backend_sched_graph_compute(scheduler_, gf);

        std::vector<int> next_tokens;
        next_tokens.reserve(n_batch);
        
        for (size_t i = 0; i < n_batch; ++i) {
            std::vector<float> slot_logits = forward_pass_->get_output_logits_for_slot(gf, i);
            // Each slot decodes through its OWN sampler (per-request temperature/
            // RNG/grammar). last_tokens stays empty here, exactly as before —
            // enabling the decode-time repetition penalty would change greedy
            // output and is a separate, deliberate behavior change. The vocab is
            // passed so a slot grammar can mask logits (no-op otherwise);
            // accept_token then advances that grammar's cursor.
            const int tok = slot_samplers_[slot_ids[i]]->sample(slot_logits, {}, vocab_);
            slot_samplers_[slot_ids[i]]->accept_token(tok);
            next_tokens.push_back(tok);

            // Advance cache for each slot in the batch
            forward_pass_->advance_cache(1, slot_ids[i]);
        }
        
        return next_tokens;
    }

    Model model_;
    std::unique_ptr<ForwardPassBase> forward_pass_;
    std::unique_ptr<Tokenizer> tokenizer_;
    // Per-slot samplers (index = slot_id). Greedy by default; rebuilt per request
    // by prepare_slot(). Each slot's TemperatureSampler owns its own RNG, so a
    // seeded request is reproducible and concurrent slots never interleave draws.
    std::vector<std::unique_ptr<qwenium::Sampler>> slot_samplers_;
    // Per-slot grammars (index = slot_id; null unless the request set a GBNF
    // grammar). Owns the cursor the slot's sampler points at via set_grammar; its
    // accepting state drives the engine's grammar-completion stop. Text-only.
    std::vector<std::unique_ptr<qwenium::GrammarVocab>> slot_grammars_;
    // Cached vocabulary (id -> token string), built once. Passed to sample() as
    // token_strs and used to build each grammar slot's trie.
    std::vector<std::string> vocab_;
    ggml_backend_sched_t scheduler_;
    std::mutex model_mutex_;  // Protects all model operations
    const ChatTemplate* chat_template_ = nullptr;  // F6: arch-registered renderer
    int max_ctx_per_slot_;
    int max_slots_;

    // Vision (image input). Null for a text-only server; non-null after a
    // successful --mmproj load. Owns the entire image pipeline + image caches.
    std::unique_ptr<ServerVision> vision_;

    // Text prefix cache (--prefix-cache). Null unless wired. Opt-in, version-
    // gated, transparent: a recurring system-prompt block skips its prefill on a
    // HIT (content-keyed by the prefix tokens). §1 decision A follow-on.
    std::unique_ptr<PrefixLibrary> text_prefix_lib_;

    // Attention Lens (--attention-lens). Drives /v1/extract only; free decode.
    // Stage 2 deleted both GrammarVocabs that used to live here — the fixed KV
    // extraction grammar and the yes/no presence grammar — with the presence gate.
    // The lens holds no grammar state at all now.
    bool attention_lens_enabled_ = false;
    ggml_type kv_type_ = GGML_TYPE_F32;  // --kv-f16 selects F16
};

// =============================================================================
// Chat-completions helpers
// =============================================================================

// Extract a chat message's text. OpenAI allows `content` to be a string OR an
// array of content parts ({type:"text", text:"..."}); join the text parts.
static std::string chat_content_to_text(const json& content) {
    if (content.is_string()) return content.get<std::string>();
    if (content.is_array()) {
        std::string out;
        for (const auto& part : content) {
            if (part.is_string()) out += part.get<std::string>();
            else if (part.is_object() && part.value("type", "") == "text")
                out += part.value("text", "");
        }
        return out;
    }
    return "";  // null / absent (e.g. an assistant tool-call message)
}

// Render an OpenAI messages[] array into the model's chat template (F6) via the
// architecture's registered ChatTemplate — handles Qwen <|im_start|>, Gemma
// <start_of_turn>, and Gemma-4's distinct markers, including per-family role
// mapping (assistant→model, system handling).
static std::string render_chat(const json& messages, const ChatTemplate* tmpl,
                               std::optional<bool> enable_thinking) {
    std::vector<ChatMessage> history;
    history.reserve(messages.size());
    for (const auto& msg : messages) {
        history.push_back(ChatMessage{
            msg.value("role", "user"),
            chat_content_to_text(msg.contains("content") ? msg["content"] : json())});
    }
    return tmpl->render(history, /*add_assistant_prompt=*/true, enable_thinking);
}

// Image-aware variant of render_chat. For every message carrying an image_url
// content part, the projector's image marker (`image_marker_prefix`, e.g.
// "\n\n<start_of_image>\n\n") is prepended to that turn's text and the decoded
// image bytes are appended to `out_images` in order — the marker is what the
// integration's expand_image_markers later turns into the soft-token span.
// Throws (named param) via extract_images_from_content on a malformed image part.
static std::string render_chat_with_images(
    const json& messages, const ChatTemplate* tmpl,
    std::optional<bool> enable_thinking, const std::string& image_marker_prefix,
    size_t max_image_bytes, std::vector<std::vector<uint8_t>>& out_images,
    bool wants_thinking) {
    std::vector<ChatMessage> history;
    history.reserve(messages.size() + 1);
    // Gemma 4 image input requires the thinking branch: force it on and ensure a
    // leading system turn exists (the template injects <|think|> into it). Gemma 3
    // image input is unaffected (its template ignores enable_thinking and emits no
    // <|think|>). See docs/server-image-multirequest-bug.md §5.
    bool has_system = !messages.empty() &&
                      messages.front().value("role", "user") == "system";
    if (wants_thinking) {
        enable_thinking = true;
        if (!has_system) history.push_back(ChatMessage{"system", ""});
    }
    for (const auto& msg : messages) {
        const json content = msg.contains("content") ? msg["content"] : json();
        std::string text = chat_content_to_text(content);
        if (qwenium::content_has_image(content)) {
            for (auto& img :
                 qwenium::extract_images_from_content(content, max_image_bytes))
                out_images.push_back(std::move(img.bytes));
            text = image_marker_prefix + text;  // marker before the user's text
        }
        history.push_back(ChatMessage{msg.value("role", "user"), std::move(text)});
    }
    return tmpl->render(history, /*add_assistant_prompt=*/true, enable_thinking);
}

// Extract an explicit thinking toggle from an OpenAI-style request body.
// Honors a top-level "enable_thinking" and the vLLM/Qwen
// "chat_template_kwargs": {"enable_thinking": bool} convention; nullopt means
// the caller did not specify, so the template's family default applies (and
// for Qwen, a "/no_think" soft-switch in the messages can still take effect).
static std::optional<bool> extract_enable_thinking(const json& body) {
    if (body.contains("enable_thinking") && body["enable_thinking"].is_boolean())
        return body["enable_thinking"].get<bool>();
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
        const auto& kw = body["chat_template_kwargs"];
        if (kw.contains("enable_thinking") && kw["enable_thinking"].is_boolean())
            return kw["enable_thinking"].get<bool>();
    }
    return std::nullopt;
}

// OpenAI chat finish_reason enum is narrower than the engine's. Map onto it.
static std::string chat_finish_reason(const std::string& engine_reason) {
    return engine_reason == "length" ? "length" : "stop";
}

// =============================================================================
// HTTP Routes
// =============================================================================
void setup_routes(httplib::Server& http, qwenium::InferenceServer& inference, QweniumServerIntegration& integration) {
    
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

        std::cout << "Received prompt: " << prompt << std::endl;
        std::cout << "Received prompt End" << std::endl;

        auto inf_req = std::make_shared<qwenium::InferenceRequest>();
        inf_req->prompt = prompt;
        inf_req->max_tokens = body.value("max_tokens", 256);
        inf_req->temperature = body.value("temperature", 0.0f);
        // Sampling controls (honored only when temperature > 0). Absent fields
        // keep the request struct's defaults. `seed` (OpenAI) makes a stochastic
        // run reproducible; omit for non-deterministic sampling.
        inf_req->top_p = body.value("top_p", inf_req->top_p);
        inf_req->top_k = body.value("top_k", inf_req->top_k);
        inf_req->seed  = body.value("seed", static_cast<long long>(-1));
        // Optional GBNF grammar for constrained (structured) output.
        inf_req->grammar = body.value("grammar", "");

        // Stop sequences: OpenAI allows a single string or an array of strings.
        if (body.contains("stop")) {
            const auto& stop = body["stop"];
            if (stop.is_string()) {
                inf_req->stop.push_back(stop.get<std::string>());
            } else if (stop.is_array()) {
                for (const auto& s : stop) {
                    if (s.is_string()) inf_req->stop.push_back(s.get<std::string>());
                }
            } else {
                res.status = 400;
                res.set_content(json({{"error", "'stop' must be a string or array of strings"}}).dump(),
                                "application/json");
                return;
            }
        }

        // A top-level system_prompt is rendered together with the user prompt
        // into one fully-templated turn and prefilled whole. Requests without it
        // are unchanged: the bare user prompt goes through set_tokenize's
        // single-user-turn wrap.
        std::string system_prompt = body.value("system_prompt", "");
        if (!system_prompt.empty()) {
            std::cout << "Received system prompt: " << system_prompt << std::endl;
            inf_req->prompt = integration.render_system_user_turn(system_prompt, prompt);
            inf_req->skip_template = true;  // already fully templated
            // Mark the system-turn block as the cacheable prefix (no-op unless
            // --prefix-cache is wired). It is a token-prefix of the full render,
            // so a recurring system_prompt skips its prefill on a cache HIT.
            if (integration.text_prefix_cache_enabled())
                inf_req->cacheable_prefix_text =
                    integration.render_system_prefix(system_prompt);
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
                    // One stateful channel filter per stream (mirrors the CLI's
                    // per-turn instance): the <|channel> thought-channel span lives
                    // across tokens, so it must be fed token-by-token, not stripped
                    // per token. Inert for non-Gemma-4 models.
                    ChannelFilter channel_filter;
                    while (true) {
                        int token_id = inf_req->token_queue->pop_blocking();

                        if (token_id == qwenium::TokenQueue::QUEUE_END) {
                            std::cout << "=========Qwenium Response===========" << std::endl;
                            std::cout << inf_req->output_text << std::endl;
                            std::cout << "=========Qwenium Response End===========" << std::endl;
                            // Send final event
                            std::string done_event = "data: [DONE]\n\n";
                            sink.write(done_event.c_str(), done_event.size());
                            sink.done();
                            return false;  // Stop
                        }

                        std::string token_text =
                            channel_filter.feed(inference.decode_token(token_id));
                        if (token_text.empty()) continue;  // suppressed (thought / marker)

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
            // Non-streaming: block until the inference thread signals completion.
            // The server owns the canonical (stop-truncated) output_text, token
            // counts, and finish_reason; drain the queue only to wait for the end.
            while (inf_req->token_queue->pop_blocking() != qwenium::TokenQueue::QUEUE_END) {
                // tokens are folded into output_text on the inference thread
            }

            // Fail-loud: a named rejection (e.g. oversized prompt) ends the
            // request with no slot and a populated error_message.
            if (!inf_req->error_message.empty()) {
                res.status = 413;  // Payload Too Large
                res.set_content(json({{"error", inf_req->error_message}}).dump(), "application/json");
                return;
            }
            std::cout << "=========Qwenium Response===========" << std::endl;
            std::cout << inf_req->output_text << std::endl;
            std::cout << "=========Qwenium Response End===========" << std::endl;


            json response = {
                {"object", "text_completion"},
                {"choices", {{
                    // Strip Gemma 4 channel framing once over the full assembled
                    // text (stateful state machine; see normalize_output). Inert
                    // for non-Gemma-4 models.
                    {"text", ChannelFilter::strip(inf_req->output_text)},
                    {"index", 0},
                    {"finish_reason", inf_req->finish_reason}
                }}},
                {"usage", {
                    {"prompt_tokens", inf_req->prompt_tokens},
                    {"completion_tokens", inf_req->completion_tokens},
                    {"total_tokens", inf_req->prompt_tokens + inf_req->completion_tokens}
                }}
            };
            res.set_content(response.dump(), "application/json");
        }
    });

    // OpenAI-compatible CHAT completions endpoint. Renders messages[] through
    // the Qwen <|im_start|> template and reuses the same generation path as
    // /v1/completions (skip_template=true so it is not re-wrapped). Supports
    // the OpenAI chat.completion (non-stream) and chat.completion.chunk (SSE)
    // response shapes so standard chat clients (e.g. Qwen Code) can connect.
    http.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::parse_error& e) {
            res.status = 400;
            res.set_content(json({{"error", "Invalid JSON"}}).dump(), "application/json");
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array() ||
            body["messages"].empty()) {
            res.status = 400;
            res.set_content(json({{"error", "Missing or empty 'messages' array"}}).dump(),
                            "application/json");
            return;
        }

        // Detect image content parts up front so the capability gate fires
        // before we attempt to decode anything.
        bool wants_image = false;
        for (const auto& msg : body["messages"]) {
            if (msg.contains("content") &&
                qwenium::content_has_image(msg["content"])) {
                wants_image = true;
                break;
            }
        }

        auto inf_req = std::make_shared<qwenium::InferenceRequest>();
        if (wants_image) {
            // Capability gate (fail-loud, CLAUDE.md): a text-only model cannot
            // consume image input — name the field, the expected capability, and
            // the actual model rather than silently dropping the image.
            if (!integration.vision_enabled()) {
                res.status = 400;
                res.set_content(json({{"error",
                    "field 'messages[].content[].image_url': image input requires "
                    "a vision-capable model started with --mmproj (Gemma 3/4 "
                    "vision); loaded model is text-only: " +
                    integration.model_label()}}).dump(), "application/json");
                return;
            }
            try {
                inf_req->prompt = render_chat_with_images(
                    body["messages"], integration.chat_template(),
                    extract_enable_thinking(body), integration.image_marker_prefix(),
                    integration.max_image_bytes(), inf_req->image_bytes,
                    integration.image_wants_thinking());
            } catch (const std::exception& e) {
                // Malformed base64 / unsupported mime / oversize / non-data URI.
                res.status = 400;
                res.set_content(json({{"error", std::string(e.what())}}).dump(),
                                "application/json");
                return;
            }
            // v1 single-image scope: refuse >1 image fail-loud (the encoder arms
            // one span) rather than encode only the first.
            if (inf_req->image_bytes.size() != 1) {
                res.status = 400;
                res.set_content(json({{"error",
                    "field 'messages[].content[].image_url': expected exactly 1 "
                    "image per request (single-image scope), got: " +
                    std::to_string(inf_req->image_bytes.size())}}).dump(),
                    "application/json");
                return;
            }
        } else {
            const std::optional<bool> enable_thinking = extract_enable_thinking(body);
            inf_req->prompt = render_chat(body["messages"],
                                          integration.chat_template(),
                                          enable_thinking);
            // Mark a LEADING system message as the cacheable prefix (no-op unless
            // --prefix-cache is wired). Rendered with the SAME thinking flag so it
            // stays a byte-aligned token-prefix of the full render. Only a system
            // message that opens the conversation is a clean leading prefix.
            const auto& msgs = body["messages"];
            if (integration.text_prefix_cache_enabled() && !msgs.empty() &&
                msgs.front().value("role", "") == "system") {
                const std::string system_content = chat_content_to_text(
                    msgs.front().contains("content") ? msgs.front()["content"] : json());
                if (!system_content.empty())
                    inf_req->cacheable_prefix_text =
                        integration.render_system_prefix(system_content, enable_thinking);
            }
        }
        std::cout << "Received prompt: " << inf_req->prompt << std::endl;
        std::cout << "Received prompt End" << std::endl;
        inf_req->skip_template = true;  // already fully <|im_start|>-rendered
        inf_req->max_tokens = body.value("max_tokens", 1024);
        // Conversation continuation handle (--conversational). 'new' starts a
        // conversation (server mints + returns a real id); a real id continues;
        // absent/empty stays stateless. The engine rejects it fail-loud unless
        // the server enabled the mode.
        inf_req->conversation_id = body.value("conversation_id", "");
        inf_req->temperature = body.value("temperature", 0.0f);
        // Sampling controls (honored only when temperature > 0). Absent fields
        // keep the request struct's defaults. `seed` (OpenAI) makes a stochastic
        // run reproducible; omit for non-deterministic sampling.
        inf_req->top_p = body.value("top_p", inf_req->top_p);
        inf_req->top_k = body.value("top_k", inf_req->top_k);
        inf_req->seed  = body.value("seed", static_cast<long long>(-1));
        // Optional GBNF grammar for constrained (structured) output.
        inf_req->grammar = body.value("grammar", "");

        // Stop sequences: OpenAI allows a single string or an array of strings.
        if (body.contains("stop")) {
            const auto& stop = body["stop"];
            if (stop.is_string()) {
                inf_req->stop.push_back(stop.get<std::string>());
            } else if (stop.is_array()) {
                for (const auto& s : stop) {
                    if (s.is_string()) inf_req->stop.push_back(s.get<std::string>());
                }
            } else {
                res.status = 400;
                res.set_content(json({{"error", "'stop' must be a string or array of strings"}}).dump(),
                                "application/json");
                return;
            }
        }

        const std::string model_name = body.value("model", "qwen-local");
        const long created = static_cast<long>(std::time(nullptr));
        const bool stream = body.value("stream", false);

        if (!inference.submit(inf_req)) {
            res.status = 503;
            res.set_content(json({{"error", "Server overloaded, queue full"}}).dump(), "application/json");
            return;
        }

        if (stream) {
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_header("X-Accel-Buffering", "no");

            res.set_chunked_content_provider(
                "text/event-stream",
                [inf_req, &inference, model_name, created](size_t /*offset*/, httplib::DataSink& sink) {
                    // One stateful channel filter per stream (mirrors the CLI's
                    // per-turn instance): the <|channel> thought-channel span lives
                    // across tokens, so it must be fed token-by-token. Inert for
                    // non-Gemma-4 models.
                    ChannelFilter channel_filter;
                    // First chunk announces the assistant role (OpenAI convention).
                    json head = {
                        {"object", "chat.completion.chunk"},
                        {"created", created},
                        {"model", model_name},
                        {"choices", {{
                            {"index", 0},
                            {"delta", {{"role", "assistant"}}},
                            {"finish_reason", nullptr}
                        }}}
                    };
                    std::string s = "data: " + head.dump() + "\n\n";
                    if (!sink.write(s.c_str(), s.size())) { inf_req->cancelled = true; return false; }

                    while (true) {
                        int token_id = inf_req->token_queue->pop_blocking();
                        if (token_id == qwenium::TokenQueue::QUEUE_END) {
                            std::string payload;
                            if (!inf_req->error_message.empty()) {
                                // Fail-loud: surface the named rejection (e.g.
                                // oversized prompt) in the stream rather than
                                // ending empty — clients otherwise report
                                // "stream ended with empty response text".
                                json err = {{"error", {
                                    {"message", inf_req->error_message},
                                    {"type", "invalid_request_error"}
                                }}};
                                payload = "data: " + err.dump() + "\n\ndata: [DONE]\n\n";
                            } else {

                                std::cout << "=========Qwenium Response===========" << std::endl;
                                std::cout << inf_req->output_text << std::endl;
                                std::cout << "=========Qwenium Response End===========" << std::endl;

                                json tail = {
                                    {"object", "chat.completion.chunk"},
                                    {"created", created},
                                    {"model", model_name},
                                    {"choices", {{
                                        {"index", 0},
                                        {"delta", json::object()},
                                        {"finish_reason", chat_finish_reason(inf_req->finish_reason)}
                                    }}}
                                };
                                // Echo the conversation handle on the terminal
                                // chunk so a streaming client can continue.
                                if (!inf_req->conversation_id.empty())
                                    tail["conversation_id"] = inf_req->conversation_id;
                                payload = "data: " + tail.dump() + "\n\ndata: [DONE]\n\n";
                            }
                            sink.write(payload.c_str(), payload.size());
                            sink.done();
                            return false;
                        }

                        std::string token_text =
                            channel_filter.feed(inference.decode_token(token_id));
                        if (token_text.empty()) continue;  // suppressed (thought / marker)
                        json chunk = {
                            {"object", "chat.completion.chunk"},
                            {"created", created},
                            {"model", model_name},
                            {"choices", {{
                                {"index", 0},
                                {"delta", {{"content", token_text}}},
                                {"finish_reason", nullptr}
                            }}}
                        };
                        std::string sse = "data: " + chunk.dump() + "\n\n";
                        if (!sink.write(sse.c_str(), sse.size())) {
                            inf_req->cancelled = true;
                            return false;
                        }
                    }
                    return true;
                },
                [inf_req](bool success) {
                    if (!success) inf_req->cancelled = true;
                }
            );
        } else {
            // Block until the inference thread signals completion.
            while (inf_req->token_queue->pop_blocking() != qwenium::TokenQueue::QUEUE_END) {
                // tokens are folded into output_text on the inference thread
            }

            if (!inf_req->error_message.empty()) {
                res.status = 413;  // Payload Too Large (fail-loud, named reason)
                res.set_content(json({{"error", inf_req->error_message}}).dump(), "application/json");
                return;
            }
            std::cout << "=========Qwenium Response===========" << std::endl;
            std::cout << inf_req->output_text << std::endl;
            std::cout << "=========Qwenium Response End===========" << std::endl;

            json response = {
                {"object", "chat.completion"},
                {"created", created},
                {"model", model_name},
                {"choices", {{
                    {"index", 0},
                    // Strip Gemma 4 channel framing once over the full assembled
                    // text (stateful; see normalize_output). Inert otherwise.
                    {"message", {{"role", "assistant"},
                                 {"content", ChannelFilter::strip(inf_req->output_text)}}},
                    {"finish_reason", chat_finish_reason(inf_req->finish_reason)}
                }}},
                {"usage", {
                    {"prompt_tokens", inf_req->prompt_tokens},
                    {"completion_tokens", inf_req->completion_tokens},
                    {"total_tokens", inf_req->prompt_tokens + inf_req->completion_tokens}
                }}
            };
            // Echo the conversation handle (minted on create) so the client can
            // continue; absent for stateless requests (--conversational off / no id).
            if (!inf_req->conversation_id.empty())
                response["conversation_id"] = inf_req->conversation_id;
            res.set_content(response.dump(), "application/json");
        }
    });

    // Conversation management (--conversational): clear one conversation or flush
    // all. The registry lives on the inference thread, so these marshal the op and
    // wait briefly for the result (fail-loud 503 if the engine is unresponsive).
    // Clearing is always safe: a continue against a cleared id fails loud (recover
    // → resend full history), never corruption.
    auto await_clear = [](std::future<int>& fut, httplib::Response& res,
                          const std::function<void(int)>& on_ok) {
        if (fut.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
            res.status = 503;
            res.set_content(json({{"error", "conversation clear timed out"}}).dump(),
                            "application/json");
            return;
        }
        on_ok(fut.get());
    };
    http.Delete(R"(/v1/conversations/([^/]+))",
                [&inference, await_clear](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.matches[1];
        auto fut = inference.clear_conversation(id);
        await_clear(fut, res, [&](int n) {
            if (n < 0) {
                res.status = 404;
                res.set_content(json({{"error", "unknown conversation_id '" + id + "'"}}).dump(),
                                "application/json");
            } else {
                res.set_content(json({{"deleted", id}, {"cleared", n}}).dump(),
                                "application/json");
            }
        });
    });
    http.Delete("/v1/conversations",
                [&inference, await_clear](const httplib::Request&, httplib::Response& res) {
        auto fut = inference.clear_all_conversations();
        await_clear(fut, res, [&](int n) {
            res.set_content(json({{"flushed", n}}).dump(), "application/json");
        });
    });

    // Attention Lens extraction endpoint (--attention-lens). Dedicated shape:
    // inputs are (document, key_vocabulary), not chat messages; output is the
    // lens format, not a completion. The OpenAI endpoints are untouched (A2).
    // Single-slot, exclusive (integration.extract_lens_json holds the model
    // lock). Returns 404 when the feature is off, 400 on bad input (fail-loud).
    http.Post("/v1/extract", [&integration](const httplib::Request& req, httplib::Response& res) {
        if (!integration.attention_lens_enabled()) {
            res.status = 404;
            res.set_content(json({{"error", "attention lens disabled — start the "
                                            "server with --attention-lens"}}).dump(),
                            "application/json");
            return;
        }
        std::string document;
        std::vector<qwenium::LensConcept> concepts;
        int max_tokens = 512;
        try {
            json body = json::parse(req.body);
            document = body.at("document").get<std::string>();
            // key_vocabulary is an array of {key, gloss} objects. `gloss` is
            // accepted but CURRENTLY UNUSED — its consumer, the presence gate,
            // was deleted in Stage 2 (see LensConcept in server_lens.h). A bare
            // string element is tolerated as {key, gloss:""} so string-array
            // callers still work; the object form is canonical.
            for (const auto& kv : body.at("key_vocabulary")) {
                if (kv.is_string()) {
                    concepts.push_back({kv.get<std::string>(), ""});
                } else if (kv.is_object()) {
                    qwenium::LensConcept c;
                    c.key   = kv.at("key").get<std::string>();
                    c.gloss = kv.contains("gloss") ? kv.at("gloss").get<std::string>() : "";
                    concepts.push_back(std::move(c));
                } else {
                    throw std::runtime_error("key_vocabulary element expected a "
                                             "string or {\"key\",\"gloss\"} object");
                }
            }
            if (body.contains("max_tokens")) max_tokens = body.at("max_tokens").get<int>();
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", std::string("bad request — expected "
                "{\"document\": string, \"key_vocabulary\": [{\"key\",\"gloss\"}|string,...], "
                "\"max_tokens\"?: int}: ") + e.what()},
                {"code", "bad_request"}}).dump(), "application/json");
            return;
        }
        try {
            std::string lens_json =
                integration.extract_lens_json(document, concepts, max_tokens);
            res.set_content(lens_json, "application/json");
        } catch (const qwenium::LensUnparseableError& e) {
            // The shape contract (docs/lens-format.md): the REQUEST was fine —
            // the model's output for THIS document could not be parsed. That is
            // not a bad request, and an importer must be able to tell the two
            // apart without string-matching a message: 400 means "fix your
            // call", 422 means "route this document to a human". Carries `raw`
            // so the failure is inspectable. Never a partial extraction, never
            // an empty one — a refusal and "the document has none of these
            // concepts" are different facts, and collapsing them would
            // re-introduce the silent data loss the grammar used to cause, one
            // layer up.
            res.status = 422;
            res.set_content(json({{"error", e.what()},
                                  {"code", "unparseable_extraction"},
                                  {"raw", e.raw}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", e.what()},
                                  {"code", "bad_request"}}).dump(), "application/json");
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
    // Quiet ggml's INFO/WARN chatter (e.g. the gallocr "cannot reallocate
    // multi buffer graph" / sched "reserving" lines emitted on every image
    // prefill); forward only errors. Same mechanism as the CLI (cli/main.cpp).
    ggml_log_set([](ggml_log_level level, const char* text, void* user_data) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
        (void)user_data;
    }, nullptr);

    // Parse arguments
    int port = 8080;
    std::string model_path;
    std::string mmproj_path;  // optional Gemma vision projector GGUF
    std::string image_embed_cache_dir;   // V1: opt-in disk image-embed cache
    std::string image_prefix_cache_dir;  // V2: opt-in disk image-prefix KV cache
    std::string prefix_cache_dir;        // text: opt-in disk system-prefix KV cache
    bool chat_prefix_cache = false;      // text: opt-in warm per-slot KV reuse (chat)
    bool conversational = false;         // text: opt-in warm conversational server (explicit handle)
    bool attention_lens = false;         // opt-in: enable POST /v1/extract (Qemmi-Lens)
    bool kv_f16 = false;                 // opt-in: F16 attention KV cache (halves KV memory)
    int max_ctx = 2048;  // per-slot context ceiling (KV cache size + fail-loud
                         // prompt guard). Raise for agent clients (e.g. Qwen
                         // Code) whose system prompt exceeds 2048 tokens.
    int max_slots = 10;  // concurrent slots. KV cache = ctx × slots × F32, so
                         // dropping to 1 frees ~10× context headroom — the right
                         // trade for one-request-at-a-time delegation.

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "--mmproj" || arg == "-j") && i + 1 < argc) {
            mmproj_path = argv[++i];
        } else if ((arg == "--ctx" || arg == "-c") && i + 1 < argc) {
            max_ctx = std::stoi(argv[++i]);
        } else if ((arg == "--slots" || arg == "-s") && i + 1 < argc) {
            max_slots = std::stoi(argv[++i]);
        } else if (arg == "--image-embed-cache" && i + 1 < argc) {
            image_embed_cache_dir = argv[++i];
        } else if (arg == "--image-prefix-cache" && i + 1 < argc) {
            image_prefix_cache_dir = argv[++i];
        } else if (arg == "--prefix-cache" && i + 1 < argc) {
            prefix_cache_dir = argv[++i];
        } else if (arg == "--chat-prefix-cache") {
            chat_prefix_cache = true;
        } else if (arg == "--conversational") {
            conversational = true;
        } else if (arg == "--attention-lens") {
            attention_lens = true;
        } else if (arg == "--kv-f16") {
            kv_f16 = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --port,   -p PORT  Port to listen on (default: 8080)\n"
                      << "  --model,  -m PATH  Path to model file\n"
                      << "  --mmproj, -j PATH  Vision projector GGUF (Gemma 3 / "
                         "Gemma 4 / Qwen 3.5-family); enables image input on "
                         "/v1/chat/completions\n"
                      << "  --ctx,    -c N     Per-slot context ceiling in tokens "
                         "(default: 2048)\n"
                      << "  --slots,  -s N     Concurrent slots, 1..10 (default: 10). "
                         "Fewer slots = more context headroom.\n"
                      << "  --image-embed-cache DIR   Opt-in disk cache: encode each "
                         "image once per node (V1, ViT skip)\n"
                      << "  --image-prefix-cache DIR  Opt-in disk cache: skip ViT + "
                         "image-position prefill for a recurring (context,image) (V2). "
                         "Refused on an M-RoPE recipe (Qwen 3.5-family): the snapshot "
                         "carries no rope coordinate\n"
                      << "  --prefix-cache DIR        Opt-in disk cache: skip the "
                         "prefill of a recurring system prompt (text path)\n"
                      << "  --kv-f16                  Opt-in: store the attention KV "
                         "cache as F16 instead of F32 (halves KV memory). Token-stable, "
                         "not byte-identical; recurrent state stays F32\n"
                      << "  --chat-prefix-cache       Opt-in: retain each slot's KV "
                         "and prefill only the new turn of a growing conversation "
                         "(transparent, token-identical; text path)\n"
                      << "  --conversational          Opt-in: warm conversational "
                         "server. A conversation_id ('new' to start) retains KV "
                         "across turns and appends only the new turn (chat.cpp-grade; "
                         "warm != cold). Excludes --chat-prefix-cache; text path\n"
                      << "  --attention-lens          Opt-in: enable POST "
                         "/v1/extract — document + complete key vocabulary → "
                         "audited key-value JSON on the attention trust layer "
                         "(single-slot; Qwen3.6). OpenAI endpoints untouched\n"
                      << "  --help,   -h       Show this help\n";
            return 0;
        }
    }

    // Setup signal handling
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        // Initialize model integration
        QweniumServerIntegration integration(model_path, max_ctx, max_slots, mmproj_path,
                                          image_embed_cache_dir, image_prefix_cache_dir,
                                          prefix_cache_dir, kv_f16);
        if (attention_lens) integration.enable_attention_lens();

        // Create inference server
        qwenium::InferenceServer::Config config;
        config.max_slots = integration.max_slots();
        config.max_queue_depth = 100;
        config.max_context = integration.max_ctx_per_slot();  // fail-loud guard on prompt size
        config.request_timeout = std::chrono::seconds(300);
        config.chat_prefix_cache = chat_prefix_cache;  // opt-in warm per-slot KV reuse
        if (chat_prefix_cache)
            std::cout << "Text: chat prefix cache ON (--chat-prefix-cache): warm "
                         "per-slot KV reuse for growing conversations" << std::endl;
        // Warm conversational server (--conversational). Excludes
        // --chat-prefix-cache: both manage warm prefix KV and would race over the
        // same slots — pick one (the doc's consolidation-watch).
        if (conversational && chat_prefix_cache) {
            std::cerr << "Fatal error: --conversational and --chat-prefix-cache are "
                         "mutually exclusive (both manage warm prefix KV; pick one)"
                      << std::endl;
            return 1;
        }
        config.conversational = conversational;
        if (conversational)
            std::cout << "Text: conversational server ON (--conversational): explicit "
                         "conversation_id continuation, chat.cpp-grade reasoning "
                         "retention (warm != cold)" << std::endl;

        qwenium::InferenceServer inference(config);
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
        std::cout << "  POST /v1/completions       (text-only)" << std::endl;
        std::cout << "  POST /v1/chat/completions"
                  << (integration.vision_enabled()
                          ? "  (text + image_url image input)"
                          : "  (text-only; start with --mmproj for images)")
                  << std::endl;
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