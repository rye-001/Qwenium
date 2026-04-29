#include <iostream>
#include <fstream>

#include "chat.h"
#include "../models/model_registry.h"

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
        
        const int32_t eos_token_id = model.get_metadata().eos_token_id;
        sampler->set_eos_token_id(eos_token_id);
    
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
            model, &chat_meta, args.context_length, 2, args.kv_quant_bits);
        forward_pass->set_snapkv_config(args.snapkv_budget, args.snapkv_window);
        ggml_backend_sched_t scheduler = model.get_scheduler();
        std::vector<ChatMessage> chat_history;

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

        // Only prefill and clone when there is actual system content.
        // An empty render (e.g. Gemma with no system prompt) must not inject
        // a spurious turn into the KV cache — slot 1 stays at pos=0 and the
        // first user-turn encode will own the BOS.
        if (!system_tokens.empty()) {
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

            // Format only the new user turn for tokenization
            std::string turn_prompt = tmpl.render({{"user", user_input}}, true);
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
            std::vector<float> logits = forward_pass->run_prefill(new_tokens, current_pos, session_slot, scheduler);
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
            for (int i = 0; i < args.max_tokens; ++i) {
                std::string decoded_token = tokenizer->decode(next_token_id);

                if (next_token_id == eos_token_id) {
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
                std::vector<int32_t> current_token_vec = { next_token_id };
                int decode_pos = forward_pass->get_cache_pos(session_slot);

                std::vector<float> token_logits = forward_pass->run_prefill(current_token_vec, decode_pos, session_slot, scheduler);
                last_token_logits.assign(token_logits.begin(), token_logits.begin() + vocab_size);
                
                next_token_id = sampler->sample(last_token_logits, all_tokens, decoded_vocab);

                if (grammar) {
                    grammar->accept_token(next_token_id, decoded_vocab);
                }

                if (next_token_id == eos_token_id) {
                    break;
                }

            }
            end_chat_generation:

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