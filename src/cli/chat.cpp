#include <iostream>
#include <fstream>

#include "chat.h"

int run_chat(
    Qwen3Model& model,
    const CliArgs& args,
    std::unique_ptr<qwen3::Grammar>& grammar,
    qwen3::SpeculativeDecoder* spec,
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

        std::unique_ptr<qwen3::Sampler> sampler;
        if (args.temperature > 0.0f) {
            sampler = std::make_unique<qwen3::TemperatureSampler>(
                args.temperature, args.repetition_penalty, 64, args.top_k, args.top_p);
        } else {
            sampler = std::make_unique<qwen3::GreedySampler>();
        }
        if (grammar) {
            sampler->set_grammar(grammar.get());
        }
    
        // Load pruned vocabulary if specified
        std::unordered_set<int32_t> pruned_vocab;
        if (!args.vocab_prune_list_path.empty()) {
            pruned_vocab = qwen3::load_keep_list(args.vocab_prune_list_path);
            sampler->set_pruned_vocab(&pruned_vocab);
            if (args.verbose) {
                std::cout << "Loaded pruned vocabulary: " << pruned_vocab.size() << " tokens\n";
            }
        }
        
        std::vector<int32_t> all_tokens; // Accumulate all tokens here
        auto forward_pass = create_forward_pass(model, &model.get_metadata(), args.context_length, 2);
        ggml_backend_sched_t scheduler = model.get_scheduler();
        std::vector<ChatMessage> chat_history;

        // System Prompt Prefill
        std::string system_content = args.system_prompt;
        if (system_content.empty()) {
            // std::ifstream file("system_prompt.txt");
            // std::ifstream file("tests/system_prompt_order_mngmt.txt");
            std::ifstream file("tests/system_prompt_account_mngmt.txt");
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

        std::string system_turn = apply_chat_template({chat_history.back()}, false);
        std::vector<int32_t> system_tokens = tokenizer->encode(system_turn);
        log_tokens(system_tokens);
        all_tokens.insert(all_tokens.end(), system_tokens.begin(), system_tokens.end());

        // Prefill System Prompt into Slot 0
        ggml_backend_sched_reset(scheduler);
        ggml_cgraph* gf_system = forward_pass->build_prefill_graph(system_tokens, 0, 0); // Slot 0
        ggml_backend_sched_alloc_graph(scheduler, gf_system);
        forward_pass->set_inputs(gf_system, system_tokens, 0);
        ggml_backend_sched_graph_compute(scheduler, gf_system);
        forward_pass->advance_cache(system_tokens.size(), 0); // Slot 0

        // Clone System Prefix to Slot 1 for the user session
        forward_pass->clone_slot(0, 1, system_tokens.size());
        
        const int32_t eos_token_id = model.get_metadata().eos_token_id;
        const std::string im_end_str = "<|im_end|>";
        const std::string eos_str = "<|endoftext|>";

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

            // Format only the new user turn for tokenization
            std::string turn_prompt = apply_chat_template({{"user", user_input}});
            std::vector<int32_t> new_tokens = tokenizer->encode(turn_prompt);
            log_tokens(new_tokens);
            
            // Reset grammar state for the new turn
            if (grammar) {
                grammar->reset();
            }
            
            // Use Slot 1 for User Session
            const uint32_t session_slot = 1;
            int current_pos = forward_pass->get_cache_pos(session_slot);

            // Process only the new tokens
            ggml_backend_sched_reset(scheduler);
            ggml_cgraph* gf = forward_pass->build_prefill_graph(new_tokens, current_pos, session_slot);
            ggml_backend_sched_alloc_graph(scheduler, gf);
            forward_pass->set_inputs(gf, new_tokens, current_pos);
            ggml_backend_sched_graph_compute(scheduler, gf);
            forward_pass->advance_cache(new_tokens.size(), session_slot);
            all_tokens.insert(all_tokens.end(), new_tokens.begin(), new_tokens.end());

            std::vector<float> logits = forward_pass->get_output_logits(gf);
            size_t vocab_size = model.get_metadata().vocab_size;
            std::vector<float> last_token_logits(logits.end() - vocab_size, logits.end());
            int next_token_id = sampler->sample(last_token_logits, all_tokens, vocab);
            if (grammar) {
                grammar->accept_token(next_token_id, vocab);
            }
           
            std::string assistant_response = "";
            std::cout << "Assistant: " << std::flush;

            // PLD: prompt_tokens = all tokens seen so far (system + user turns)
            // generated_tokens = tokens generated in this assistant turn
            std::vector<int32_t> prompt_tokens_for_pld = all_tokens;
            std::vector<int32_t> generated_tokens;

            // Decode phase
            for (int i = 0; i < args.max_tokens; ++i) {
                std::string decoded_token = tokenizer->decode(next_token_id);

                if (next_token_id == eos_token_id || decoded_token == im_end_str || decoded_token == eos_str) {
                    break;
                }

                log_token(next_token_id);
                print_token(decoded_token);
                assistant_response += decoded_token;
                all_tokens.push_back(next_token_id);
                generated_tokens.push_back(next_token_id);

                if (grammar && grammar->is_accepting_state()) {
                    break;
                }

                // --- Speculative step: try to get bonus tokens ---
                if (use_speculative) {
                    int spec_pos = forward_pass->get_cache_pos(session_slot);

                    // Normal decode for current token first
                    std::vector<int32_t> current_token_vec = { next_token_id };
                    int decode_pos = spec_pos;

                    ggml_backend_sched_reset(scheduler);
                    ggml_cgraph* gf_token = forward_pass->build_prefill_graph(current_token_vec, decode_pos, session_slot);
                    ggml_backend_sched_alloc_graph(scheduler, gf_token);
                    forward_pass->set_inputs(gf_token, current_token_vec, decode_pos);
                    ggml_backend_sched_graph_compute(scheduler, gf_token);
                    forward_pass->advance_cache(1, session_slot);

                    // Now try speculative step
                    int after_decode_pos = forward_pass->get_cache_pos(session_slot);
                    auto result = spec->try_speculative_step(
                        prompt_tokens_for_pld,
                        generated_tokens,
                        session_slot,
                        after_decode_pos,
                        bridge.make_verify(session_slot),
                        bridge.make_rewind(session_slot),
                        eos_token_id);

                    if (result.attempted() && result.total_tokens() > 0) {
                        // Deliver accepted draft tokens
                        for (int32_t t : result.accepted_tokens) {
                            std::string bonus_str = tokenizer->decode(t);
                            log_token(t);
                            print_token(bonus_str);
                            assistant_response += bonus_str;
                            all_tokens.push_back(t);
                            generated_tokens.push_back(t);
                            i++;

                            if (t == eos_token_id || bonus_str == im_end_str || bonus_str == eos_str) {
                                goto end_chat_generation;  // Break out of outer loop
                            }
                        }

                        // Deliver bonus token (the correct token at mismatch or after full acceptance)
                        if (result.has_bonus) {
                            next_token_id = result.bonus_token;
                            continue;  // Skip the normal decode below, we already have next token
                        }

                        // If no bonus (EOS was the mismatch), get next token normally
                        if (!result.has_bonus) {
                            // EOS was hit in draft verification
                            goto end_chat_generation;
                        }
                    } else {
                        // No draft found — get next token from the decode we already did
                        std::vector<float> token_logits = forward_pass->get_output_logits(gf_token);
                        last_token_logits.assign(token_logits.begin(), token_logits.begin() + vocab_size);
                        next_token_id = sampler->sample(last_token_logits, all_tokens, vocab);
                        if (grammar) {
                            grammar->accept_token(next_token_id, vocab);
                        }
                    }
                    continue;
                }

                // --- Normal (non-speculative) decode path ---
                std::vector<int32_t> current_token_vec = { next_token_id };
                int decode_pos = forward_pass->get_cache_pos(session_slot);

                ggml_backend_sched_reset(scheduler);
                ggml_cgraph* gf_token = forward_pass->build_prefill_graph(current_token_vec, decode_pos, session_slot);
                ggml_backend_sched_alloc_graph(scheduler, gf_token);
                forward_pass->set_inputs(gf_token, current_token_vec, decode_pos);
                ggml_backend_sched_graph_compute(scheduler, gf_token);
                forward_pass->advance_cache(1, session_slot);

                std::vector<float> token_logits = forward_pass->get_output_logits(gf_token);
                last_token_logits.assign(token_logits.begin(), token_logits.begin() + vocab_size);
                
                next_token_id = sampler->sample(last_token_logits, all_tokens, vocab);
                if (grammar) {
                    grammar->accept_token(next_token_id, vocab);
                }
            }
            end_chat_generation:

            // Print PLD stats for this turn
            if (use_speculative && args.verbose) {
                auto& s = spec->stats();
                std::cout << "\n[PLD] drafts=" << s.drafts_found
                          << " accept_rate=" << (int)(s.acceptance_rate() * 100) << "%"
                          << " tokens/step=" << s.tokens_per_step() << std::endl;
            }

            // After generation ends, we must add the <|im_end|>\n tokens to the cache
            // so the next turn sees a properly terminated assistant message
            std::string im_end_suffix = "<|im_end|>\n";
            std::vector<int32_t> im_end_tokens = tokenizer->encode(im_end_suffix);
            int end_pos = forward_pass->get_cache_pos(session_slot);
            
            ggml_backend_sched_reset(scheduler);
            ggml_cgraph* gf_end = forward_pass->build_prefill_graph(im_end_tokens, end_pos, session_slot);
            ggml_backend_sched_alloc_graph(scheduler, gf_end);
            forward_pass->set_inputs(gf_end, im_end_tokens, end_pos);
            ggml_backend_sched_graph_compute(scheduler, gf_end);
            forward_pass->advance_cache(im_end_tokens.size(), session_slot);
            all_tokens.insert(all_tokens.end(), im_end_tokens.begin(), im_end_tokens.end());


            chat_history.push_back({"assistant", assistant_response});
        }

        // Print final PLD stats
        if (use_speculative) {
            auto& s = spec->stats();
            std::cout << "\n[PLD Final Stats]"
                      << " drafts_attempted=" << s.drafts_attempted
                      << " drafts_found=" << s.drafts_found
                      << " tokens_drafted=" << s.tokens_drafted
                      << " tokens_accepted=" << s.tokens_accepted
                      << " bonus_tokens=" << s.bonus_tokens
                      << " accept_rate=" << (int)(s.acceptance_rate() * 100) << "%"
                      << " tokens_per_step=" << s.tokens_per_step()
                      << std::endl;
        }
    
    return 0;
}