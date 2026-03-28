
#include <iostream>

#include "complete.h"

int run_complete(
    Qwen3Model& model,
    const CliArgs& args,
    std::unique_ptr<qwen3::GrammarVocab>& grammar,
    qwen3::SpeculativeDecoder* spec,
    bool use_speculative,
    std::function<void(int32_t)> log_token,
    std::function<void(const std::vector<int32_t>&)> log_tokens
) {
    // Responsibilities:
    //   1. Create sampler + pruned vocab
    //   2. Tokenize args.prompt
    //   3. Create forward pass (1 slot) + scheduler
    //   4. Prefill prompt
    //   5. Decode loop:
    //      a. Normal path: single-token decode, sample, repeat
    //      b. Speculative path: decode + PLD try_speculative_step, handle
    //         accepted/bonus tokens, goto end_single_generation on EOS
    //   6. Print PLD stats if speculative
    //   7. Print "End of Generation"


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
        return 0;
    }