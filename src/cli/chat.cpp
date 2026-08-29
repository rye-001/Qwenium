#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <fstream>

#include "chat.h"
#include "../core/decode_step.h"
#include "../core/decode_graph_cache.h"
#include "../core/multimodal_prefill.h"
#include "../core/image_embedding_cache.h"
#include "../core/persistent_image_embedding_store.h"
#include "../core/prefix_library.h"
#include "../core/slot_snapshot.h"
#include "../session/compat_header.h"
#include "../session/session_manifest.h"
#include "../session/snapshot_io.h"
#include "../state/kv_cache_simple.h"
#include "../state/deltanet_state.h"
#include "../models/model_registry.h"
#include "../models/gemma3.h"
#include "../loader/channel_filter.h"
#include "../models/i_image_embeddable.h"
#include "../vision/vision_model.h"
#include "../vision/vision_loader.h"
#include "../vision/i_vision_encoder.h"
#include "../vision/siglip_encoder.h"
#include "../vision/gemma4uv_encoder.h"
#include "../vision/bitmap.h"
#include "image_loader.h"
#include "image_prompt.h"
#include "../vision/vision_profile.h"

int run_chat(
    Model& model,
    const CliArgs& args,
    std::unique_ptr<qwenium::GrammarVocab>& grammar,
    qwenium::SpeculativeDecoder* spec,
    bool use_speculative,
    std::function<void(int32_t)> log_token,
    std::function<void(const std::vector<int32_t>&)> log_tokens
) {
        std::cout << "\n--- Starting Chat Mode ---" << std::endl;
        std::cout << "Model: " << model.get_metadata().model_name << std::endl;
        std::cout << "Press Enter on an empty line to submit your message." << std::endl;
        std::cout << "Type 'exit' or 'quit' to end the conversation." << std::endl;

        Tokenizer* tokenizer = model.get_tokenizer();
        const auto vocab = tokenizer->get_vocabulary();


// Build decoded vocab: decoded_vocab[i] = actual UTF-8 string for token i
const auto& raw_vocab = tokenizer->get_vocabulary();
std::vector<std::string> decoded_vocab;
decoded_vocab.reserve(raw_vocab.size());
for (size_t i = 0; i < raw_vocab.size(); ++i) {
    decoded_vocab.push_back(tokenizer->decode(static_cast<int32_t>(i)));
}        

        std::unique_ptr<qwenium::Sampler> sampler;
        if (args.temperature > 0.0f) {
            sampler = std::make_unique<qwenium::TemperatureSampler>(
                args.temperature, args.repetition_penalty, 64, args.top_k, args.top_p);
        } else {
            // Greedy is a true argmax unless --repeat-penalty was explicitly
            // given. Passing it here is what makes the flag mean something on
            // this path; it used to be silently dropped.
            sampler = std::make_unique<qwenium::GreedySampler>(
                args.repetition_penalty_set ? args.repetition_penalty : 1.0f);
        }
        if (grammar) {
            sampler->set_grammar(grammar.get());
        }
        sampler->build_token_trie(decoded_vocab);

        for (int32_t id : model.get_metadata().stop_token_ids)
            sampler->add_eos_token_id(id);
    
        // Load pruned vocabulary if specified
        std::unordered_set<int32_t> pruned_vocab;
        if (!args.vocab_prune_list_path.empty()) {
            pruned_vocab = qwenium::load_keep_list(args.vocab_prune_list_path);
            sampler->set_pruned_vocab(&pruned_vocab);
            if (args.verbose) {
                std::cout << "Loaded pruned vocabulary: " << pruned_vocab.size() << " tokens\n";
            }
        }
        
        std::vector<int32_t> all_tokens; // Accumulate all tokens here
        const auto& chat_meta = model.get_metadata();
        std::unique_ptr<ForwardPassBase> forward_pass = create_forward_pass(
            model, &chat_meta, args.context_length, 2,
            args.kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
        ggml_backend_sched_t scheduler = model.get_scheduler();
        std::vector<ChatMessage> chat_history;

        // Opt-in warm-prefix KV cache (--prefix-cache): a recurring system prompt
        // is prefilled once and its slot-0 KV (+ recurrent) reused on later runs,
        // skipping the prefill. Version-gated fail-loud (Phase 4). Requires a
        // recipe that exposes its KV cache; refuse loudly if not (the user opted
        // in explicitly).
        std::unique_ptr<PrefixLibrary> prefix_lib;
        if (!args.prefix_cache_dir.empty()) {
            if (forward_pass->snapshot_kv_caches().empty())
                throw std::runtime_error(
                    "run_chat: parameter '--prefix-cache': expected a recipe that "
                    "exposes its KV cache (snapshot_kv_caches non-empty), got: a "
                    "recipe without L2 snapshot support");
            prefix_lib = std::make_unique<PrefixLibrary>(
                args.prefix_cache_dir,
                qinf::snapshot::make_snapshot_header(
                    chat_meta, forward_pass->snapshot_kv_caches()));
        }

        // One image attached to the first user turn.
        // All vision state is local and inert unless --image is given; the
        // text-only path below is byte-for-byte unchanged. Fail-loud at setup
        const bool image_mode = !args.image_path.empty();
        std::unique_ptr<qinf::vision::VisionModel>    vmodel;
        std::unique_ptr<qinf::vision::VisionLoader>   vloader;
        std::unique_ptr<qinf::vision::IVisionEncoder> vencoder;
        IImageEmbeddable* img_recipe = nullptr;
        qinf::vision::Bitmap image_bitmap;
        int32_t boi_id = -1, eoi_id = -1, soft_id = -1;
        // The string inserted before the user's text on the image turn: the
        // wrapped image marker (projector-specific). expand_image_markers then
        // turns the single marker token into the soft-token span.
        std::string image_render_prefix;
        bool image_pending = false;
        // Gemma 4 image input requires the thinking branch (system <|think|> turn
        // + no forced channel); Gemma 3 image input keeps its no-think path. Set
        // per projector below. See docs/server-image-multirequest-bug.md §5.
        bool image_wants_thinking = false;
        // Per-session encode cache: the same image referenced again in
        // a later turn reuses its embeddings instead of re-encoding. Also holds
        // the image-count cap.
        ImageEmbeddingCache image_cache;
        // Opt-in disk-backed embedding cache (--image-embed-cache): persists the
        // ViT output per image hash so a recurring image is encoded once per node
        // ever. Built below once the encoder identity (projector + dim + backend)
        // is known; null = in-memory session reuse only. (vision V1.)
        std::unique_ptr<PersistentImageEmbeddingStore> embed_store;
        // Opt-in image-prefix KV cache (--image-prefix-cache; vision V2): the
        // image turn's KV up to and including the image span is stored keyed by
        // (preceding context + image content_id); a later run with the SAME
        // image + context skips BOTH the ViT encode and the image-position
        // prefill. Version-gated fail-loud like --prefix-cache. Requires a recipe
        // that exposes its KV cache(s); refuse loudly otherwise (explicit opt-in).
        std::unique_ptr<PrefixLibrary> image_prefix_lib;
        if (image_mode) {
            if (args.mmproj_path.empty())
                throw std::runtime_error(
                    "run_chat: parameter '--image': requires '--mmproj' (the "
                    "vision projector GGUF), got: empty mmproj path");
            // (Gemma 3/4, Qwen 3.5-family). Fail loud at setup, not mid-conversation.
            img_recipe = dynamic_cast<IImageEmbeddable*>(forward_pass.get());
            if (img_recipe == nullptr)
                throw std::runtime_error(
                    "run_chat: parameter '--image': expected a recipe "
                    "implementing IImageEmbeddable (Gemma 3, Gemma 4, or "
                    "Qwen 3.5-family vision), "
                    "got: a different recipe");

            ggml_backend_t backend = model.has_metal_backend()
                ? model.get_backend_metal() : model.get_backend_cpu();
            vmodel  = std::make_unique<qinf::vision::VisionModel>();
            vloader = std::make_unique<qinf::vision::VisionLoader>();
            vloader->parse_metadata(args.mmproj_path, *vmodel);
            vloader->load_tensors(*vmodel, backend);

            // Projector-specific setup lives in ONE place (P0). Everything
            // family-shaped — encoder, cache tag, marker ids, framing string,
            // thinking requirement, preprocessing — comes back as a profile,
            // and an unhandled projector throws instead of defaulting.
            qinf::vision::VisionProfile vprofile = qinf::vision::make_vision_profile(
                *vmodel, backend, tokenizer->get_vocabulary(),
                "run_chat: parameter '--image'");

            const std::string projector_tag = vprofile.projector_tag;
            vencoder             = std::move(vprofile.encoder);
            boi_id               = vprofile.boi_id;
            eoi_id               = vprofile.eoi_id;
            soft_id              = vprofile.soft_id;
            image_render_prefix  = vprofile.marker_prefix;
            image_wants_thinking = vprofile.wants_thinking;
            image_bitmap = qinf::cli::load_image_to_bitmap(
                args.image_path, vprofile.preprocess);
            if (!args.image_embed_cache_dir.empty()) {
                // Key identity = projector + projection_dim + encode backend. A
                // different encoder / dim / backend ⇒ different header ⇒ miss
                // (opportunistic, never a wrong result). gemma4uv caches its raw
                // pre-LM form; SigLIP the encoded form (cache-boundary rule).
                embed_store = std::make_unique<PersistentImageEmbeddingStore>(
                    args.image_embed_cache_dir,
                    make_vision_header(projector_tag,
                                       vmodel->config().projection_dim,
                                       ggml_backend_name(backend)));
                std::cout << "Vision: image-embed cache '"
                          << args.image_embed_cache_dir << "' (encode once per node)\n";
            }
            if (!args.image_prefix_cache_dir.empty()) {
                if (forward_pass->snapshot_kv_caches().empty())
                    throw std::runtime_error(
                        "run_chat: parameter '--image-prefix-cache': expected a "
                        "recipe that exposes its KV cache(s) (snapshot_kv_caches "
                        "non-empty), got: a recipe without L2 snapshot support");
                // A 2-D image span writes nx*ny KV rows while advancing the
                // position by only max(nx, ny). The snapshot blob records a row
                // count and no rope coordinate, so such a slot cannot be
                // round-tripped (plan §4 decision 3 — VL sessions are declared
                // non-snapshottable in v1). capture_slot refuses this too, but
                // that fires only AFTER a full model load and image encode;
                // refuse here, before either is paid for.
                if (img_recipe->image_span_is_2d())
                    throw std::runtime_error(
                        "run_chat: parameter '--image-prefix-cache': expected a "
                        "recipe whose image span advances one position per KV row, "
                        "got: an M-RoPE recipe, whose image span occupies nx*ny "
                        "rows but max(nx, ny) positions. The snapshot format "
                        "carries no rope coordinate, so VL sessions are not "
                        "prefix-cacheable in v1 — drop --image-prefix-cache");
                image_prefix_lib = std::make_unique<PrefixLibrary>(
                    args.image_prefix_cache_dir,
                    qinf::snapshot::make_snapshot_header(
                        chat_meta, forward_pass->snapshot_kv_caches()));
                std::cout << "Vision: image-prefix cache '"
                          << args.image_prefix_cache_dir
                          << "' (skip ViT + image prefill on a recurring image)\n";
            }
            image_pending = true;
            std::cout << "Vision: mmproj '" << args.mmproj_path << "' + image '"
                      << args.image_path << "' ("
                      << vencoder->mm_tokens_for(image_bitmap) << " soft tokens)\n";
        }

        // System Prompt Prefill
        std::string system_content;
        std::string sys_prompt_file = args.system_prompt;
        if (!sys_prompt_file.empty()) {
            // std::ifstream file("system_prompt.txt");
            // std::ifstream file("tests/system_prompt_order_mngmt.txt");
            // std::ifstream file("tests/system_prompt_account_mngmt.txt");
            std::ifstream file(sys_prompt_file);
            if (file) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                system_content = buffer.str();
            } else {
                 std::cerr << "Warning: Could not open system_prompt.txt. Using empty system prompt." << std::endl;
            }
        }
        chat_history.push_back({"system", system_content});
        std::cout << "System Prompt: " << system_content << std::endl;

        const ChatTemplate& tmpl = *lookup_chat_template(chat_meta.architecture);
        std::string system_turn = tmpl.render({chat_history.back()}, false);
        std::vector<int32_t> system_tokens = tokenizer->encode(system_turn);

        // Only prefill and clone when there is actual system CONTENT. Gemma
        // renders an empty system prompt as a non-empty wrapper
        // (<start_of_turn>user\n<end_of_turn>\n), so guarding on system_tokens
        // would still inject a spurious empty turn and push slot 1 off pos 0 —
        // which also robs the first user turn of its BOS. Guard on the content.
        if (!system_content.empty() && !system_tokens.empty()) {
            log_tokens(system_tokens);
            all_tokens.insert(all_tokens.end(), system_tokens.begin(), system_tokens.end());

            // Warm-prefix path: on a HIT, memcpy the cached system-prompt KV
            // (+ recurrent) into slot 0 and SKIP the prefill; on a MISS, prefill
            // and store. A present blob with a mismatched node identity is
            // refused fail-loud by try_load — never silently re-prefilled (the
            // F9 rule); surfaced here as an actionable fatal so the user clears
            // or re-points the cache dir.
            bool warm = false;
            const uint64_t pkey =
                prefix_lib ? PrefixLibrary::key_for(system_tokens) : 0;
            if (prefix_lib) {
                std::vector<uint8_t> blob;
                bool hit;
                try {
                    hit = prefix_lib->try_load(pkey, blob);
                } catch (const std::exception& e) {
                    throw std::runtime_error(
                        std::string("run_chat: '--prefix-cache': a stored blob for "
                        "this system prompt was built under a different model / "
                        "quant / backend and is refused (") + e.what() +
                        "). Clear or re-point --prefix-cache " +
                        args.prefix_cache_dir);
                }
                if (hit) {
                    qinf::snapshot::restore_slot(
                        *forward_pass, 0, blob,
                        qinf::snapshot::make_snapshot_header(
                            chat_meta, forward_pass->snapshot_kv_caches()));
                    warm = true;
                    std::cout << "[prefix-cache] HIT: skipped prefill of "
                              << system_tokens.size() << " system tokens\n";
                }
            }
            if (!warm) {
                forward_pass->run_prefill(system_tokens, 0, 0, scheduler);
                if (prefix_lib) {
                    prefix_lib->store(pkey, qinf::snapshot::capture_slot(
                        *forward_pass, 0,
                        qinf::snapshot::make_snapshot_header(
                            chat_meta, forward_pass->snapshot_kv_caches())));
                    std::cout << "[prefix-cache] MISS: prefilled + stored "
                              << system_tokens.size() << " system tokens\n";
                }
            }
            forward_pass->clone_slot(0, 1, system_tokens.size());
        }
        
        // Speculative bridge for chat mode (slot 1)
        SpeculativeBridge bridge{forward_pass.get(), scheduler};

        // Persistent decode graph (opt-in --persistent-graph): one built and
        // allocated decode graph reused across steps on a dedicated scheduler
        // (measured 1.32× on Qwen 3.6; token-stable, not byte-identical —
        // docs/plan-persistent-decode-graph.md §0.1).
        // Constructed once; invalidated before each turn's decode loop because
        // every turn's prefill (run_prefill) resets fp's context out from under
        // the retained graph. Refused fail-loud on a non-persistent recipe.
        std::unique_ptr<DecodeGraphCache> graph_cache;
        if (args.persistent_graph) {
            if (!forward_pass->supports_persistent_decode()) {
                std::cerr << "--persistent-graph: architecture '"
                          << model.get_metadata().architecture << "' is not "
                          << "persistent-capable (needs qwen35/qwen36/gemma3)\n";
                return 1;
            }
            enable_persistent_decode(forward_pass.get());
            graph_cache = std::make_unique<DecodeGraphCache>(model, forward_pass.get());
        }

        while (true) {
            std::cout << "\nUser: ";
            std::string user_input;
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line.empty()) {
                    break;
                }
                if (line == "exit" || line == "quit") {
                    user_input = line;
                    break;
                }
                user_input += line + "\n";
            }

            if (user_input == "exit\n" || user_input == "quit\n" || user_input == "exit" || user_input == "quit") {
                break;
            }

            // Remove the trailing newline for cleaner processing
            if (!user_input.empty() && user_input.back() == '\n') {
                user_input.pop_back();
            }
            chat_history.push_back({"user", user_input});

            // Guard the whole turn (tokenize → prefill → decode → suffix) so a
            // KV-cache context overflow becomes a clean session reset instead of
            // an uncaught std::runtime_error that aborts the process. Only the
            // overflow error is handled here; any other error is re-thrown to
            // preserve the fail-loud contract.
            try {

            // Reset grammar state for the new turn
            if (grammar) {
                grammar->reset();
            }

            // Use Slot 1 for User Session
            const uint32_t session_slot = 1;
            // Rope position, NOT the KV row count: an image span writes nx*ny
            // rows but advances the position by max(nx, ny), so after an image
            // turn get_cache_pos would start the next turn far past where the
            // model actually is.
            int current_pos = forward_pass->get_rope_pos(session_slot);

            // Gemma requires a BOS at the very start of the conversation. encode()
            // does not prepend it, so do it here for the first prefill (pos 0).
            const int32_t bos_id = model.get_metadata().bos_token_id;
            const bool prepend_bos = (current_pos == 0) && (bos_id >= 0);

            std::vector<int32_t> new_tokens;
            std::vector<float> logits;
            if (image_pending) {
                // Image turn: put the image marker (projector-specific, wrapped
                // in image_render_prefix) in the rendered content, tokenize, then
                // expand the single marker token into the soft-token span the
                // encoder fills. expand_image_markers inserts soft×N + the
                // end-of-image token right after the begin marker.
                // TEST (docs §5): match llama.cpp mtmd thinking branch exactly —
                // prepend a system <|think|> turn and use enable_thinking=true so
                // the generation prompt ends at "<|turn>model\n" (NO forced
                // <|channel>thought\n<channel|>). llama produces coherent image
                // output with this exact sequence.
                // Gemma 4 needs the thinking branch for image input: a leading
                // system <|think|> turn (rendered by the template when thinking is
                // on) and a generation prompt that ends at "model\n". Gemma 3 keeps
                // its existing no-think image path. See docs §5.
                std::vector<ChatMessage> turn;
                if (image_wants_thinking)
                    turn.push_back({"system", ""});
                turn.push_back({"user", image_render_prefix + user_input});
                std::string turn_prompt = tmpl.render(
                    turn, /*add_assistant_prompt=*/true,
                    image_wants_thinking ? std::optional<bool>(true) : std::nullopt);
                std::vector<int32_t> raw = tokenizer->encode(turn_prompt);
                qinf::cli::ExpandedImagePrompt built = qinf::cli::expand_image_markers(
                    raw, boi_id, soft_id, eoi_id, vencoder->mm_tokens_for(image_bitmap));
                new_tokens = std::move(built.tokens);
                int img_span_start = built.span_start;
                if (prepend_bos) {
                    new_tokens.insert(new_tokens.begin(), bos_id);
                    img_span_start += 1;  // BOS shifts every position right by one
                }
                log_tokens(new_tokens);

                std::vector<ImagePromptChunk> chunks = {{&image_bitmap, img_span_start}};
                if (!image_prefix_lib) {
                    // No image-prefix cache: the original single-call path —
                    // encode + chunked [pre-image | image | question] in one go.
                    logits = prefill_multimodal(
                        *forward_pass, *vencoder, model.get_scheduler(), new_tokens,
                        chunks, current_pos, session_slot, &image_cache,
                        embed_store.get());
                } else {
                    // Vision V2: cache the KV up to and including the image span,
                    // then prefill only the variable question suffix. The split is
                    // byte-identical to the single call (the image is its own chunk
                    // either way — proven by test-image-prefix-roundtrip GATE 2).
                    const uint32_t n_img = vencoder->mm_tokens_for(image_bitmap);
                    const int img_end_local = img_span_start + static_cast<int>(n_img);
                    const std::vector<int32_t> image_inclusive(
                        new_tokens.begin(), new_tokens.begin() + img_end_local);
                    const std::vector<int32_t> suffix(
                        new_tokens.begin() + img_end_local, new_tokens.end());
                    const int img_end_pos = current_pos + img_end_local;

                    // Key over the FIXED context before the image span (slot tokens
                    // already prefilled = all_tokens, plus this turn's pre-image
                    // text) and the image content_id. The question (after the span)
                    // is NOT in the key.
                    std::vector<int32_t> preceding = all_tokens;  // slot [0, current_pos)
                    preceding.insert(preceding.end(), new_tokens.begin(),
                                     new_tokens.begin() + img_span_start);
                    const uint64_t ikey = PrefixLibrary::key_for(
                        preceding, image_bitmap.content_id);

                    std::vector<uint8_t> blob;
                    bool ihit = false;
                    try {
                        ihit = image_prefix_lib->try_load(ikey, blob);
                    } catch (const std::exception& e) {
                        throw std::runtime_error(
                            std::string("run_chat: '--image-prefix-cache': a stored "
                            "blob for this (context, image) was built under a "
                            "different model / quant / backend and is refused (") +
                            e.what() + "). Clear or re-point --image-prefix-cache " +
                            args.image_prefix_cache_dir);
                    }
                    if (ihit) {
                        // Restore the image-inclusive KV into the session slot and
                        // SKIP both the ViT encode and the image-position prefill.
                        qinf::snapshot::restore_slot(
                            *forward_pass, session_slot, blob,
                            qinf::snapshot::make_snapshot_header(
                                chat_meta, forward_pass->snapshot_kv_caches()));
                        std::cout << "[image-prefix-cache] HIT: skipped ViT + image "
                                     "prefill (" << n_img << " soft tokens)\n";
                    } else {
                        // MISS: encode + chunked-prefill [pre-image | image], then
                        // capture the post-image KV and store it under the key.
                        prefill_multimodal(
                            *forward_pass, *vencoder, model.get_scheduler(),
                            image_inclusive, chunks, current_pos, session_slot,
                            &image_cache, embed_store.get());
                        image_prefix_lib->store(ikey, qinf::snapshot::capture_slot(
                            *forward_pass, session_slot,
                            qinf::snapshot::make_snapshot_header(
                                chat_meta, forward_pass->snapshot_kv_caches())));
                        std::cout << "[image-prefix-cache] MISS: encoded + prefilled "
                                     "+ stored (" << n_img << " soft tokens)\n";
                    }
                    // The question suffix rides the plain text path either way.
                    logits = forward_pass->run_prefill(suffix, img_end_pos,
                                                       session_slot, scheduler);
                }
                image_pending = false;
            } else {
                // Format only the new user turn for tokenization
                std::string turn_prompt = tmpl.render({{"user", user_input}}, true);
                new_tokens = tokenizer->encode(turn_prompt);
                if (prepend_bos) new_tokens.insert(new_tokens.begin(), bos_id);
                log_tokens(new_tokens);
                logits = forward_pass->run_prefill(new_tokens, current_pos, session_slot, scheduler);
            }
            all_tokens.insert(all_tokens.end(), new_tokens.begin(), new_tokens.end());

            size_t vocab_size = model.get_metadata().vocab_size;
            std::vector<float> last_token_logits(logits.end() - vocab_size, logits.end());
            int next_token_id = sampler->sample(last_token_logits, all_tokens, decoded_vocab);
            

            if (grammar) {
                grammar->accept_token(next_token_id, decoded_vocab);
            }
           
            std::string assistant_response = "";
            std::cout << "Assistant: " << std::flush;

            // Gemma 4 channel-stream filter (shared with the server — see
            // loader/channel_filter.h). Per-turn instance: strips the
            // <|channel>/<channel|>/<|turn> framing, robust to token-boundary
            // splits. With --hide-thinking it also drops the thought channel
            // (the server's behavior); by default thought is shown. Inert for
            // non-Gemma-4 models.
            ChannelFilter channel_filter(args.show_thinking);

            // Emit one filtered chunk. Thought-channel text is rendered dimmed
            // and deliberately NOT appended to assistant_response: reasoning is
            // shown to the user but kept out of the saved turn history, so it is
            // not fed back as the assistant's answer on the next turn.
            auto emit_visible = [&](const std::string& vis) {
                if (vis.empty()) return;
                if (channel_filter.in_thought()) {
                    std::cout << "\033[2m" << make_readable(vis) << "\033[0m" << std::flush;
                } else {
                    print_token(vis);
                    assistant_response += vis;
                }
            };

            // generated_tokens = tokens generated in this assistant turn
            std::vector<int32_t> prompt_tokens_for_pld = all_tokens;
            std::vector<int32_t> generated_tokens;

            // Decode phase
            const auto& stop_ids = model.get_metadata().stop_token_ids;
            using Clock = std::chrono::steady_clock;
            auto t_decode_start = Clock::now();

            // This turn's prefill (above) reset fp's context, so any retained
            // persistent decode graph is dangling — force a rebuild on the
            // first decode step of the turn.
            if (graph_cache) graph_cache->invalidate();

            // Phase B: tokens decode_step emitted via forced-token elision
            // (grammar-determined, no forward pass). Drained below in the
            // same order, with the same stop/print/history handling the
            // returned token gets. Grammar state is already advanced inside.
            std::vector<int32_t> forced_run;

            for (int i = 0; i < args.max_tokens; ++i) {
                std::string decoded_token = tokenizer->decode(next_token_id);
                if (std::find(stop_ids.begin(), stop_ids.end(), next_token_id) != stop_ids.end()) {
                    break;
                }
                log_token(next_token_id);
                emit_visible(channel_filter.feed(decoded_token));
                all_tokens.push_back(next_token_id);
                generated_tokens.push_back(next_token_id);

                // if (grammar && grammar->is_accepting_state()) {
                //     break;
                // }

                // --- Normal (non-speculative) decode path ---
                next_token_id = decode_step(
                    forward_pass.get(), scheduler, sampler.get(),
                    next_token_id, session_slot,
                    all_tokens, decoded_vocab, vocab_size,
                    /*force_dense=*/false, &forced_run,
                    graph_cache.get());

                // Drain forced-elided tokens (chronologically before the
                // returned one), same handling as the loop-top emission.
                for (int32_t ft : forced_run) {
                    if (std::find(stop_ids.begin(), stop_ids.end(), ft) != stop_ids.end()) {
                        goto end_chat_generation;
                    }
                    std::string ft_str = tokenizer->decode(ft);
                    log_token(ft);
                    emit_visible(channel_filter.feed(ft_str));
                    all_tokens.push_back(ft);
                    generated_tokens.push_back(ft);
                }

                if (std::find(stop_ids.begin(), stop_ids.end(), next_token_id) != stop_ids.end()) {
                    break;
                }

            }
            end_chat_generation:
            {
                auto t_decode_end = Clock::now();
                auto decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_decode_end - t_decode_start).count();
                const size_t n_decoded = generated_tokens.size();
                double tps = (decode_ms > 0 && n_decoded > 0)
                    ? (n_decoded * 1000.0 / decode_ms) : 0.0;
                std::cout << "\n[Timing] decode=" << decode_ms << "ms ("
                          << n_decoded << " tokens, " << tps << " t/s)\n";
            }

            // After generation ends, append the turn-end marker so the next
            // turn sees a properly terminated assistant message.
            std::string im_end_suffix = tmpl.turn_end_suffix();
            std::vector<int32_t> im_end_tokens = tokenizer->encode(im_end_suffix);
            // Rope position, not the KV row count (they diverge after an image).
            int end_pos = forward_pass->get_rope_pos(session_slot);
            
            forward_pass->run_prefill(im_end_tokens, end_pos, session_slot, scheduler);
            all_tokens.insert(all_tokens.end(), im_end_tokens.begin(), im_end_tokens.end());


            chat_history.push_back({"assistant", assistant_response});

            } catch (const std::exception& e) {
                // Only a KV-cache context overflow is recoverable here; rethrow
                // anything else so genuine faults stay loud.
                const std::string msg = e.what();
                if (msg.find("overflow") == std::string::npos) throw;

                std::cout << "\n\033[1;33m[System] KV cache context limit reached. "
                             "Resetting conversation context...\033[0m\n";
                // Flush both slots (0 = system-prompt template slot, 1 = session)
                // back to pos 0 and drop the in-memory transcript so the next turn
                // starts fresh (and re-prepends BOS at pos 0).
                forward_pass->clear_slot(0);
                forward_pass->clear_slot(1);
                all_tokens.clear();
                chat_history.clear();
                continue;
            }
        }
    return 0;
}