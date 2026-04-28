
#include <iostream>
#include <chrono>

#include "complete.h"
#include "../models/model_registry.h"

int run_complete(
    Model& model,
    const CliArgs& args,
    std::unique_ptr<qwenium::GrammarVocab>& grammar,
    qwenium::SpeculativeDecoder* spec,
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

        std::unique_ptr<qwenium::Sampler> sampler;
        if (args.temperature > 0.0f) {
            sampler = std::make_unique<qwenium::TemperatureSampler>(
                args.temperature, 
                args.repetition_penalty, 
                64, // lookback window
                args.top_k, 
                args.top_p
            );
        } else {
            sampler = std::make_unique<qwenium::GreedySampler>();
        }
        if (grammar) {
            sampler->set_grammar(grammar.get());
        }

        // Load pruned vocabulary if specified
        std::unordered_set<int32_t> pruned_vocab;
        if (!args.vocab_prune_list_path.empty()) {
            pruned_vocab = qwenium::load_keep_list(args.vocab_prune_list_path);
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
        const auto& cmp_meta = model.get_metadata();
        register_builtin_models();
        std::unique_ptr<ForwardPassBase> forward_pass = create_forward_pass(
            model, &cmp_meta, args.context_length, 1, args.kv_quant_bits);
        forward_pass->set_snapkv_config(args.snapkv_budget, args.snapkv_window);

        // Prefill phase
        using Clock = std::chrono::steady_clock;
        const size_t n_prompt_tokens = tokens.size();
        auto t_prefill_start = Clock::now();
        std::vector<float> logits = forward_pass->run_prefill(tokens, 0, 0, scheduler);
        auto t_prefill_end = Clock::now();
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
        
        auto t_decode_start = Clock::now();
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

                std::vector<float> decode_logits = forward_pass->run_prefill(current_token_vec, decode_pos, slot, scheduler);

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
                    last_token_logits.assign(decode_logits.begin(), decode_logits.begin() + vocab_size);
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

            std::vector<float> token_logits = forward_pass->run_prefill(current_token_vec, current_pos, 0, scheduler);
            last_token_logits.assign(token_logits.begin(), token_logits.begin() + vocab_size);
            
            next_token_id = sampler->sample(last_token_logits, tokens, vocab);
            if (grammar) {
                grammar->accept_token(next_token_id, vocab);
            }
        }
        end_single_generation:
        auto t_decode_end = Clock::now();

        // Print timing
        {
            auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                t_prefill_end - t_prefill_start).count();
            auto decode_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                t_decode_end - t_decode_start).count();
            const size_t n_decoded = generated_tokens.size();

            double prefill_tps = (prefill_ms > 0)
                ? (n_prompt_tokens * 1000.0 / prefill_ms) : 0.0;
            double decode_tps  = (decode_ms  > 0 && n_decoded > 0)
                ? (n_decoded * 1000.0 / decode_ms) : 0.0;

            std::cout << "\n[Timing]"
                      << " prefill=" << prefill_ms << "ms"
                      << " (" << n_prompt_tokens << " tokens, "
                      << static_cast<int>(prefill_tps) << " t/s)"
                      << "  decode=" << decode_ms << "ms"
                      << " (" << n_decoded << " tokens, "
                      << static_cast<int>(decode_tps) << " t/s)"
                      << std::endl;
        }

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