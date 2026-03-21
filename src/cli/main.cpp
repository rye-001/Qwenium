#include <iostream>
#include <sys/resource.h>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <inttypes.h>
#include <fstream>
#include "ggml-cpu.h"

#include "ggml.h"
#include "ggml-backend.h"

#include "../qwen3-core/qwen3-model.h"
#include "../qwen3-core/gguf-loader.h"
#include "../qwen3-core/forward-pass-factory.h"
#include "../qwen3-core/tokenizer.h"
#include "../qwen3-core/sampling.h"
#include "../qwen3-core/grammar.h"
#include "../qwen3-core/speculative.h"
#include "./pretokenized_literals.h"
#include "../qwen3-core/vocab_utils.h"

// #include "./tensor_tracer.h"

void debug_computed_tensor(ggml_tensor* tensor, const char* name);
void print_token(std::string str);
std::string make_readable(std::string str);
size_t calculate_scratch_size(const Qwen3Metadata& metadata,
                               size_t max_context_length = 2048,
                               size_t batch_size = 512);
struct CliArgs {
    std::string model_path;
    std::string prompt;
    int max_tokens = 100;
    float temperature = 0.7f;
    float repetition_penalty = 1.1f;
    int top_k = 40;
    float top_p = 0.95f;
    bool verbose = false;
    bool help = false;
    bool run_test = false;
    bool chat_mode = false; // New flag for chat mode
    std::string vocab_prune_list_path;
    uint32_t context_length = 4096;
    std::string token_log_path;
    std::string grammar_file;
    std::string system_prompt;
    // PLD options
    bool speculative = false;
    int pld_ngram_size = 3;
    int pld_max_draft = 5;
};

// Represents a single message in the chat history
struct ChatMessage {
    std::string role;
    std::string content;
};

std::string apply_chat_template(const std::vector<ChatMessage>& history, bool add_assistant_prompt = true);


void print_usage(const char* program_name) {
    std::cout << "Qwen-3 Specialized Inference Engine\n\n";
    std::cout << "Usage: " << program_name << " [OPTIONS] MODEL_PATH\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  MODEL_PATH              Path to GGUF model file\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help              Show this help message\n";
    std::cout << "  -p, --prompt PROMPT     Text prompt for generation\n";
    std::cout << "  -v, --verbose           Enable verbose output\n";
    std::cout << "  --run-test              Run the RoPE golden value test and exit\n";
    std::cout << "  -n, --max-tokens N      Maximum tokens to generate (default: 100)\n";
    std::cout << "  -t, --temperature F     Sampling temperature (default: 0.7)\n";
    std::cout << "  --top-k K               Top-K sampling (default: 40)\n";
    std::cout << "  --top-p P               Top-P (nucleus) sampling (default: 0.95)\n";
    std::cout << "  --repeat-penalty F      Repetition penalty (default: 1.1)\n";
    std::cout << "  --chat                  Enter interactive chat mode\n";
    std::cout << "  --ctx-size N            KV cache context size in tokens (default: 4096)\n";
    std::cout << "  --vocab-prune-list-path FILE Path to keep_list.bin for vocab pruning\n";
    std::cout << "  --system-prompt PROMPT  System prompt to guide the assistant's behavior in chat mode\n";
    std::cout << "  --grammar-file FILE     Path to a GBNF grammar file for constrained sampling\n";
    std::cout << "  --log-tokens-to FILE    Append all processed token IDs to a file\n";
    std::cout << "  --speculative           Enable Prompt Lookup Decoding (speculative)\n";
    std::cout << "  --pld-ngram N           PLD n-gram match size (default: 3)\n";
    std::cout << "  --pld-max-draft K       PLD max draft tokens (default: 5)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " model.gguf -p \"Hello, how are you?\"\n";
    std::cout << "  " << program_name << " model.gguf --chat\n";
    std::cout << "  " << program_name << " model.gguf --chat --speculative\n";
    std::cout << "  " << program_name << " -v -n 50 -p \"Explain quantum computing\" model.gguf\n";
}

bool parse_args(int argc, char** argv, CliArgs& args) {
    if (argc < 2) {
        return false;
    }
    
    int i = 1;
    while (i < argc) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            args.help = true;
            return true;
        } else if (arg == "--run-test") {
            args.run_test = true;
            return true; // No further args needed for test
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "-n" || arg == "--max-tokens") {
            if (i + 1 >= argc) return false;
            args.max_tokens = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--temperature") {
            if (i + 1 >= argc) return false;
            args.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-k") {
            if (i + 1 >= argc) return false;
            args.top_k = std::stoi(argv[++i]);
        } else if (arg == "--top-p") {
            if (i + 1 >= argc) return false;
            args.top_p = std::stof(argv[++i]);
        } else if (arg == "--repeat-penalty") {
            if (i + 1 >= argc) return false;
            args.repetition_penalty = std::stof(argv[++i]);
        } else if (arg == "-p" || arg == "--prompt") {
            if (i + 1 >= argc) return false;
            args.prompt = argv[++i];
        } else if (arg == "--chat") {
            args.chat_mode = true;
        } else if (arg == "--vocab-prune-list-path") {
            if (i + 1 >= argc) return false;
            args.vocab_prune_list_path = argv[++i];
        } else if (arg == "--system-prompt") {
            if (i + 1 >= argc) return false;
            args.system_prompt = argv[++i];
        } else if (arg == "--grammar-file") {
            if (i + 1 >= argc) return false;
            args.grammar_file = argv[++i];
        } else if (arg == "--log-tokens-to") {
            if (i + 1 >= argc) return false;
            args.token_log_path = argv[++i];
        } else if (arg == "--ctx-size") {
            if (i + 1 >= argc) return false;
            args.context_length = std::stoi(argv[++i]);
        } else if (arg == "--speculative") {
            args.speculative = true;
        } else if (arg == "--pld-ngram") {
            if (i + 1 >= argc) return false;
            args.pld_ngram_size = std::stoi(argv[++i]);
        } else if (arg == "--pld-max-draft") {
            if (i + 1 >= argc) return false;
            args.pld_max_draft = std::stoi(argv[++i]);
        } else if (args.model_path.empty()) {
            args.model_path = arg;
        }
        i++;
    }
    
    return args.run_test || !args.model_path.empty();
}

std::string apply_chat_template(const std::vector<ChatMessage>& history, bool add_assistant_prompt) {
    std::ostringstream prompt_stream;
    for (const auto& message : history) {
        prompt_stream << "<|im_start|>";
        prompt_stream << message.role;
        prompt_stream << "\n";
        prompt_stream << message.content;
        prompt_stream << "<|im_end|>";
        prompt_stream << "\n";
    }
    if (add_assistant_prompt) {
        prompt_stream << "<|im_start|>assistant\n";
    }
    return prompt_stream.str();
}



void limit_memory(size_t max_bytes) {
    struct rlimit rl;
    rl.rlim_cur = rl.rlim_max = max_bytes;
    if (setrlimit(RLIMIT_AS, &rl) == -1) { // Not working on MacOS
        perror("setrlimit failed (virtual memory)");
        // Continue anyway
    } else {
        std::cout << "Virtual memory limited to " << max_bytes / (1024*1024) << " MB\n";
    }
}

// ============================================================================
// Helper: Build verify/rewind lambdas for speculative decoding
// These wrap the forward pass into the interface SpeculativeDecoder expects.
// ============================================================================

struct SpeculativeBridge {
    ForwardPassBase* forward_pass;
    ggml_backend_sched_t scheduler;

    // Verify: run draft tokens as a mini-prefill, return [K * vocab_size] logits
    qwen3::SpeculativeDecoder::VerifyFunc make_verify(uint32_t slot) {
        return [this, slot](int /*slot_id*/, const std::vector<int32_t>& draft, int start_pos)
            -> std::vector<float>
        {
            ggml_backend_sched_reset(scheduler);
            ggml_cgraph* gf = forward_pass->build_prefill_graph(
                const_cast<std::vector<int32_t>&>(draft), start_pos, slot);
            ggml_backend_sched_alloc_graph(scheduler, gf);
            forward_pass->set_inputs(gf, const_cast<std::vector<int32_t>&>(draft), start_pos);
            ggml_backend_sched_graph_compute(scheduler, gf);
            forward_pass->advance_cache(draft.size(), slot);
            return forward_pass->get_output_logits(gf);
        };
    }

    // Rewind: set cache position back (discards unverified KV entries)
    qwen3::SpeculativeDecoder::RewindCacheFunc make_rewind(uint32_t slot) {
        return [this, slot](int /*slot_id*/, int new_pos) {
            forward_pass->set_cache_pos(slot, new_pos);
        };
    }
};

int main(int argc, char** argv) {
    // limit_memory(4L * 1024 * 1024 * 1024);  // 4 GB limit
    std::cout << "Qwen-3 Specialized Inference Engine v0.1.0\n" << std::endl;
    
    CliArgs args;
    if (!parse_args(argc, argv, args) || args.help) {

        print_usage(argv[0]);
        return args.help ? 0 : 1;
    }

    if (args.verbose) {
        std::cout << "Verbose mode enabled" << std::endl;
        std::cout << "Model path: " << args.model_path << std::endl;
        std::cout << "Max tokens: " << args.max_tokens << std::endl;
        std::cout << "Temperature: " << args.temperature << std::endl;
        if (args.speculative) {
            std::cout << "Speculative decoding (PLD): enabled"
                      << " ngram=" << args.pld_ngram_size
                      << " max_draft=" << args.pld_max_draft << std::endl;
        }
    }


 ggml_log_set([](ggml_log_level level, const char * text, void * user_data) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
    (void)user_data;
}, nullptr);

    ggml_backend_load_all();
    
    // Initialize and load model
    Qwen3Model model;
    try {
        model.load_metadata(args.model_path);
        model.print_metadata();
        model.load_tensors();
    } catch (const GGUFLoadError& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "An unexpected error occurred: " << e.what() << std::endl;
        return 1;
    }

    // --- Grammar Setup ---
    std::unique_ptr<qwen3::Grammar> grammar;
    if (!args.grammar_file.empty()) {
        std::ifstream file(args.grammar_file);
        if (!file) {
            std::cerr << "Error: Could not open grammar file: " << args.grammar_file << std::endl;
            return 1;
        }
        std::string grammar_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        grammar = qwen3::Grammar::parse_impl(grammar_str, pretokenized_literals);
        if (!grammar) {
            std::cerr << "Error: Failed to parse grammar file."
;            return 1;
        }
        std::cout << "Successfully loaded and parsed grammar from: " << args.grammar_file << std::endl;
    }

    // --- Token Logging Setup ---
    std::ofstream token_log_file;
    if (!args.token_log_path.empty()) {
        token_log_file.open(args.token_log_path, std::ios::app);
        if (!token_log_file) {
            std::cerr << "Warning: Could not open token log file: " << args.token_log_path << std::endl;
        }
    }

    auto log_token = [&](int32_t id) {
        if (token_log_file.is_open()) {
            token_log_file << id << "\n" << std::flush;
        }
    };

    auto log_tokens = [&](const std::vector<int32_t>& ids) {
        if (token_log_file.is_open()) {
            for (int32_t id : ids) {
                token_log_file << id << "\n";
            }
            token_log_file << std::flush;
        }
    };

    // --- Speculative Decoder Setup (shared by both modes) ---
    // PLD is disabled when grammar is active (grammar needs per-token validation)
    bool use_speculative = args.speculative && !grammar;
    std::unique_ptr<qwen3::SpeculativeDecoder> spec;
    if (use_speculative) {
        qwen3::PromptLookupConfig pld_config;
        pld_config.ngram_size = args.pld_ngram_size;
        pld_config.max_draft = args.pld_max_draft;
        spec = std::make_unique<qwen3::SpeculativeDecoder>(
            pld_config, (int)model.get_metadata().vocab_size);
        std::cout << "Prompt Lookup Decoding enabled (ngram=" << pld_config.ngram_size
                  << ", max_draft=" << pld_config.max_draft << ")" << std::endl;
    }
    if (args.speculative && grammar) {
        std::cout << "Note: Speculative decoding disabled (incompatible with grammar constraints)" << std::endl;
    }

    // --- Start of Generation ---
    if (args.chat_mode) {
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
            std::ifstream file("tests/system_prompt_order_mngmt.txt");
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
    } else if (!args.prompt.empty()) {
        // === SINGLE-PROMPT MODE (original code) ===
        std::cout << "\n--- Starting Generation ---" << std::endl;
        std::cout << "Prompt: " << args.prompt << std::endl;
        std::cout << "  Model Name: " << model.get_metadata().model_name << std::endl;

        // 1. Initialize tokenizer and sampler
        Tokenizer* tokenizer = model.get_tokenizer();
        const auto vocab = tokenizer->get_vocabulary();

        std::unique_ptr<qwen3::Sampler> sampler;
        if (args.temperature > 0.0f) {
            sampler = std::make_unique<qwen3::TemperatureSampler>(
                args.temperature, 
                args.repetition_penalty, 
                64, // lookback window
                args.top_k, 
                args.top_p
            );
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


        // 2. Tokenize the prompt
        std::vector<int32_t> tokens = tokenizer->encode(args.prompt);
        log_tokens(tokens);

        if (args.verbose) {
            std::cout << "Tokenized prompt (" << tokens.size() << " tokens): ";
            for (int token_id : tokens) {
                std::cout << token_id << " ";
            }
            std::cout << std::endl;
        }

        ggml_backend_sched_t scheduler = model.get_scheduler();
        auto forward_pass = create_forward_pass(model, &model.get_metadata(), args.context_length);

        // Prefill phase
        ggml_backend_sched_reset(scheduler);            
        ggml_cgraph* gf = forward_pass->build_prefill_graph(tokens, 0, 0); // Slot 0
        ggml_backend_sched_alloc_graph(scheduler, gf);
        forward_pass->set_inputs(gf, tokens, 0);
        ggml_backend_sched_graph_compute(scheduler, gf);
        forward_pass->advance_cache(tokens.size(), 0); // Slot 0

        std::vector<float> logits = forward_pass->get_output_logits(gf);
        size_t vocab_size = model.get_metadata().vocab_size;
        std::vector<float> last_token_logits(logits.end() - vocab_size, logits.end());
        int next_token_id = sampler->sample(last_token_logits, tokens, vocab);
        if (grammar) {
            grammar->accept_token(next_token_id, vocab);
        }
        
        // Decode phase
        const int32_t eos_token_id = model.get_metadata().eos_token_id;
        const std::string im_end_str = "<|im_end|>";
        const std::string eos_str = "<|endoftext|>";

        // PLD state for single-prompt mode
        std::vector<int32_t> prompt_tokens_for_pld = tokens;  // Original prompt
        std::vector<int32_t> generated_tokens;

        // Speculative bridge for single-prompt mode (slot 0)
        SpeculativeBridge bridge{forward_pass.get(), scheduler};
        
        for (int i = 0; i < args.max_tokens; ++i) {
            std::string decoded_token = tokenizer->decode(next_token_id);

            if (next_token_id == eos_token_id || decoded_token == im_end_str || decoded_token == eos_str) {
                break;
            }

            log_token(next_token_id);
            tokens.push_back(next_token_id);
            generated_tokens.push_back(next_token_id);
            print_token(decoded_token);

            // Check for grammar completion *after* printing the token
            if (grammar && grammar->is_accepting_state()) {
                break;
            }

            // --- Speculative step ---
            if (use_speculative) {
                const uint32_t slot = 0;

                // Normal decode for current token
                std::vector<int32_t> current_token_vec = { next_token_id };
                int decode_pos = forward_pass->get_cache_pos(slot);

                ggml_backend_sched_reset(scheduler);
                ggml_cgraph* gf_token = forward_pass->build_prefill_graph(current_token_vec, decode_pos, slot);
                ggml_backend_sched_alloc_graph(scheduler, gf_token);
                forward_pass->set_inputs(gf_token, current_token_vec, decode_pos);
                ggml_backend_sched_graph_compute(scheduler, gf_token);
                forward_pass->advance_cache(1, slot);

                // Try speculative
                int after_decode_pos = forward_pass->get_cache_pos(slot);
                auto result = spec->try_speculative_step(
                    prompt_tokens_for_pld,
                    generated_tokens,
                    slot,
                    after_decode_pos,
                    bridge.make_verify(slot),
                    bridge.make_rewind(slot),
                    eos_token_id);

                if (result.attempted() && result.total_tokens() > 0) {
                    for (int32_t t : result.accepted_tokens) {
                        std::string bonus_str = tokenizer->decode(t);
                        log_token(t);
                        tokens.push_back(t);
                        generated_tokens.push_back(t);
                        print_token(bonus_str);
                        i++;

                        if (t == eos_token_id || bonus_str == im_end_str || bonus_str == eos_str) {
                            goto end_single_generation;
                        }
                    }

                    if (result.has_bonus) {
                        next_token_id = result.bonus_token;
                        continue;
                    }

                    if (!result.has_bonus) {
                        goto end_single_generation;
                    }
                } else {
                    std::vector<float> token_logits = forward_pass->get_output_logits(gf_token);
                    last_token_logits.assign(token_logits.begin(), token_logits.begin() + vocab_size);
                    next_token_id = sampler->sample(last_token_logits, tokens, vocab);
                    if (grammar) {
                        grammar->accept_token(next_token_id, vocab);
                    }
                }
                continue;
            }

            // --- Normal decode path ---
            std::vector<int32_t> current_token_vec = { next_token_id };
            int current_pos = forward_pass->get_cache_pos(0); // Slot 0
            
            ggml_backend_sched_reset(scheduler);            
            ggml_cgraph* gf_token = forward_pass->build_prefill_graph(current_token_vec, current_pos, 0); // Slot 0
            ggml_backend_sched_alloc_graph(scheduler, gf_token);
            forward_pass->set_inputs(gf_token, current_token_vec, current_pos);
            ggml_backend_sched_graph_compute(scheduler, gf_token);
            forward_pass->advance_cache(1, 0); // Slot 0

            std::vector<float> token_logits = forward_pass->get_output_logits(gf_token);
            last_token_logits.assign(token_logits.begin(), token_logits.begin() + vocab_size);
            
            next_token_id = sampler->sample(last_token_logits, tokens, vocab);
            if (grammar) {
                grammar->accept_token(next_token_id, vocab);
            }
        }
        end_single_generation:

        // Print PLD stats
        if (use_speculative) {
            auto& s = spec->stats();
            std::cout << "\n[PLD Stats]"
                      << " drafts_attempted=" << s.drafts_attempted
                      << " drafts_found=" << s.drafts_found
                      << " tokens_drafted=" << s.tokens_drafted
                      << " tokens_accepted=" << s.tokens_accepted
                      << " bonus_tokens=" << s.bonus_tokens
                      << " accept_rate=" << (int)(s.acceptance_rate() * 100) << "%"
                      << " tokens_per_step=" << s.tokens_per_step()
                      << std::endl;
        }

        std::cout << "\n--- End of Generation ---" << std::endl;
    } else {
        std::cout << "\nModel loaded successfully. Provide a prompt to start generation."
;    }
    
    return 0;
}

std::string make_readable(std::string str) {
        size_t pos = 0;
    while ((pos = str.find("\xC4\xA0", pos)) != std::string::npos) {
        str.replace(pos, 2, " ");
        pos += 1;
    }
    return str;
}

void print_token(std::string str) {
    std::cout << make_readable(str) << std::flush;
}

void llama_log_set(ggml_log_callback log_callback, void * user_data) {
    ggml_log_set(log_callback, user_data);
}



// Calculate scratch buffer size based on model architecture
size_t calculate_scratch_size(const Qwen3Metadata& metadata, 
                               size_t max_context_length,
                               size_t batch_size) {
    // Base calculation: scales with layers, hidden dimensions, and batch
    size_t hidden = metadata.embedding_length;
    size_t layers = metadata.block_count;
    size_t ffn_intermediate = metadata.feed_forward_length;

    // Estimate intermediate tensor memory (attention + FFN activations)
    // Attention: Q, K, V matrices + attention scores
    size_t attention_mem = hidden * hidden * batch_size * 3; // Q, K, V
    attention_mem += metadata.attention_head_count * batch_size * batch_size; // attention scores
    
    // FFN: intermediate expansion (typically 4x hidden size)
    size_t ffn_mem = 2 * hidden * ffn_intermediate * batch_size; // gate & up
    ffn_mem += ffn_intermediate * hidden * batch_size; // down
    
    // Per-layer memory
    size_t per_layer = (attention_mem + ffn_mem) * sizeof(float);
    
    // Total for all layers
    size_t total = per_layer * layers;
    
    // Add graph overhead and other intermediate tensors (~20% overhead)
    total = (total * 12) / 10;
    
    // Minimum 256MB for small models
    size_t min_scratch = 256ULL * 1024 * 1024;
    
    return std::max(total, min_scratch);
}
