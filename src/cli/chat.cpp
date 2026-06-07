#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>

#include "chat.h"
#include "../core/decode_step.h"
#include "../core/multimodal_prefill.h"
#include "../core/image_embedding_cache.h"
#include "../models/model_registry.h"
#include "../models/gemma3.h"
#include "../models/i_image_embeddable.h"
#include "../vision/vision_model.h"
#include "../vision/vision_loader.h"
#include "../vision/i_vision_encoder.h"
#include "../vision/siglip_encoder.h"
#include "../vision/gemma4uv_encoder.h"
#include "../vision/bitmap.h"
#include "image_loader.h"
#include "image_prompt.h"

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
            sampler = std::make_unique<qwenium::GreedySampler>();
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
            model, &chat_meta, args.context_length, 2);
        ggml_backend_sched_t scheduler = model.get_scheduler();
        std::vector<ChatMessage> chat_history;

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
        // Per-session encode cache: the same image referenced again in
        // a later turn reuses its embeddings instead of re-encoding. Also holds
        // the image-count cap.
        ImageEmbeddingCache image_cache;
        if (image_mode) {
            if (args.mmproj_path.empty())
                throw std::runtime_error(
                    "run_chat: parameter '--image': requires '--mmproj' (the "
                    "vision projector GGUF), got: empty mmproj path");
            // (Gemma 3 or Gemma 4). Fail loud at setup, not mid-conversation.
            img_recipe = dynamic_cast<IImageEmbeddable*>(forward_pass.get());
            if (img_recipe == nullptr)
                throw std::runtime_error(
                    "run_chat: parameter '--image': expected a recipe "
                    "implementing IImageEmbeddable (Gemma 3 or Gemma 4 vision), "
                    "got: a different recipe");

            ggml_backend_t backend = model.has_metal_backend()
                ? model.get_backend_metal() : model.get_backend_cpu();
            vmodel  = std::make_unique<qinf::vision::VisionModel>();
            vloader = std::make_unique<qinf::vision::VisionLoader>();
            vloader->parse_metadata(args.mmproj_path, *vmodel);
            vloader->load_tensors(*vmodel, backend);

            const auto& id_to_token = tokenizer->get_vocabulary();
            auto find_id = [&](const std::string& s) -> int32_t {
                for (size_t i = 0; i < id_to_token.size(); ++i)
                    if (id_to_token[i] == s) return static_cast<int32_t>(i);
                return -1;
            };

            using PT = qinf::vision::VisionProjectorType;
            if (vmodel->config().projector_type == PT::Gemma3Siglip) {
                vencoder = std::make_unique<qinf::vision::SiglipEncoder>(
                    *vmodel, backend, vmodel->config().projection_dim);
                boi_id  = find_id("<start_of_image>");
                eoi_id  = find_id("<end_of_image>");
                soft_id = find_id("<image_soft_token>");
                if (boi_id < 0 || eoi_id < 0 || soft_id < 0)
                    throw std::runtime_error(
                        "run_chat: parameter '--image': expected the model's "
                        "tokenizer to define <start_of_image>/<image_soft_token>/"
                        "<end_of_image>, got: a text-only (non-multimodal) vocab");
                // HF Gemma3Processor wraps the image block in "\n\n" on both sides.
                image_render_prefix = "\n\n<start_of_image>\n\n";
                image_bitmap = qinf::cli::load_image_to_bitmap(
                    args.image_path,
                    qinf::cli::gemma3_preprocess(
                        static_cast<int>(vmodel->config().image_size)));
            } else {  // VisionProjectorType::Gemma4Uv
                vencoder = std::make_unique<qinf::vision::Gemma4UvEncoder>(
                    *vmodel, backend, vmodel->config().projection_dim);
                // Markers per llama.cpp mtmd: <|image> … <image|>. The per-position
                // filler <|image|> is cosmetic — its embedding is overwritten by
                // the substitution; only the N reserved positions matter.
                boi_id  = find_id("<|image>");
                eoi_id  = find_id("<image|>");
                soft_id = find_id("<|image|>");
                if (boi_id < 0 || eoi_id < 0 || soft_id < 0)
                    throw std::runtime_error(
                        "run_chat: parameter '--image': expected the model's "
                        "tokenizer to define <|image>/<|image|>/<image|>, got: a "
                        "text-only (non-multimodal) vocab");
                // The image block is wrapped in "\n\n" — empirically load-bearing:
                // without the surrounding newlines the begin marker abuts
                // "<|turn>user\n"/the user text and the image is not read at all
                // (the model reports a blank/garbled message). With them, the
                // X-ray is read correctly. (A separate, milder degenerate-prefix
                // artifact at generation start is still under investigation.)
                image_render_prefix = "\n\n<|image>\n\n";
                const int eff_patch = static_cast<int>(
                    vmodel->config().patch_size * vmodel->config().n_merge);
                image_bitmap = qinf::cli::load_image_to_bitmap(
                    args.image_path, qinf::cli::gemma4uv_preprocess(eff_patch));
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
            forward_pass->run_prefill(system_tokens, 0, 0, scheduler);
            forward_pass->clone_slot(0, 1, system_tokens.size());
        }
        
        // Speculative bridge for chat mode (slot 1)
        SpeculativeBridge bridge{forward_pass.get(), scheduler};

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

            // Reset grammar state for the new turn
            if (grammar) {
                grammar->reset();
            }

            // Use Slot 1 for User Session
            const uint32_t session_slot = 1;
            int current_pos = forward_pass->get_cache_pos(session_slot);

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
                std::string turn_prompt = tmpl.render(
                    {{"user", image_render_prefix + user_input}}, true);
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
                logits = prefill_multimodal(*forward_pass, *vencoder, scheduler,
                                            new_tokens, chunks, current_pos,
                                            session_slot, &image_cache);
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

            // generated_tokens = tokens generated in this assistant turn
            std::vector<int32_t> prompt_tokens_for_pld = all_tokens;
            std::vector<int32_t> generated_tokens;

            // Decode phase
            const auto& stop_ids = model.get_metadata().stop_token_ids;
            using Clock = std::chrono::steady_clock;
            auto t_decode_start = Clock::now();

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
                print_token(decoded_token);
                assistant_response += decoded_token;
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
                    /*force_dense=*/false, &forced_run);

                // Drain forced-elided tokens (chronologically before the
                // returned one), same handling as the loop-top emission.
                for (int32_t ft : forced_run) {
                    if (std::find(stop_ids.begin(), stop_ids.end(), ft) != stop_ids.end()) {
                        goto end_chat_generation;
                    }
                    std::string ft_str = tokenizer->decode(ft);
                    log_token(ft);
                    print_token(ft_str);
                    assistant_response += ft_str;
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
            int end_pos = forward_pass->get_cache_pos(session_slot);
            
            forward_pass->run_prefill(im_end_tokens, end_pos, session_slot, scheduler);
            all_tokens.insert(all_tokens.end(), im_end_tokens.begin(), im_end_tokens.end());


            chat_history.push_back({"assistant", assistant_response});
        }    
    return 0;
}