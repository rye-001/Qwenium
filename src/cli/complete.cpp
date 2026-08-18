
#include <iostream>
#include <chrono>

#include "complete.h"
#include "../core/decode_step.h"
#include "../core/decode_graph_cache.h"
#include "../models/model_registry.h"
#include "../models/i_mtp_draftable.h"
#include "../sampling/draft_source.h"
#include "../state/deltanet_state.h"

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
        sampler->build_token_trie(vocab);

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


        // 2. Tokenize the prompt.  encode() does not prepend BOS; honor the
        //    model's add_bos_token contract here (chat.cpp does the same at
        //    its first prefill).  Gemma -it models go degenerate without BOS.
        std::vector<int32_t> tokens = tokenizer->encode(args.prompt);
        {
            const auto& md = model.get_metadata();
            if (md.add_bos_token && md.bos_token_id >= 0)
                tokens.insert(tokens.begin(), md.bos_token_id);
        }
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
            model, &cmp_meta, args.context_length, 1);

        // --speculative mtp: build the MtpDraft-backed decoder here — the head
        // lives on the recipe, which only now exists. Fail-loud if the loaded
        // GGUF has no NextN head (D2). The head graphs run on a DEDICATED
        // scheduler: a new graph shape must not share galloc state with the
        // main graphs (docs/server-image-multirequest-bug.md precedent).
        const bool mtp_mode = use_speculative && args.speculative_mode == "mtp";
        std::unique_ptr<qwenium::SpeculativeDecoder> mtp_spec;
        ggml_backend_sched_t mtp_sched = nullptr;
        if (mtp_mode) {
            auto* mtp_cap = dynamic_cast<IMtpDraftable*>(forward_pass.get());
            if (!mtp_cap || !mtp_cap->mtp_supported()) {
                throw std::runtime_error(
                    "--speculative mtp: MTP capability expected present, actual "
                    "absent — architecture '" + cmp_meta.architecture +
                    "' has no NextN head (needs an MTP-converted GGUF with "
                    "qwen35moe.nextn_predict_layers > 0)");
            }
            forward_pass->set_output_hidden(true);
            if (model.has_metal_backend()) {
                ggml_backend_t backends[] = {model.get_backend_metal(),
                                             model.get_backend_cpu()};
                mtp_sched = ggml_backend_sched_new(backends, nullptr, 2,
                                                   FP_GRAPH_SIZE, true, false);
            } else {
                ggml_backend_t backends[] = {model.get_backend_cpu()};
                mtp_sched = ggml_backend_sched_new(backends, nullptr, 1,
                                                   FP_GRAPH_SIZE, false, false);
            }
            auto bridge_fn = [mtp_cap, mtp_sched](
                uint32_t slot, const std::vector<float>& h, int32_t t, int p,
                uint32_t k) {
                return mtp_cap->mtp_draft(slot, h, t, p, k, mtp_sched);
            };
            mtp_spec = std::make_unique<qwenium::SpeculativeDecoder>(
                std::make_unique<qwenium::MtpDraft>(bridge_fn, args.mtp_max_draft),
                (int)cmp_meta.vocab_size);
            spec = mtp_spec.get();
            std::cout << "MTP speculative decoding enabled (head-draft="
                      << args.mtp_max_draft << ")" << std::endl;
        }

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

        // Persistent decode graph (opt-in --persistent-graph). Reuses one
        // built and allocated decode graph across steps on a dedicated
        // scheduler (measured 1.32× on Qwen 3.6; token-stable, not
        // byte-identical — docs/plan-persistent-decode-graph.md §0.1). Applies
        // to the normal (non-speculative) decode step below; speculative verify
        // shapes are out of scope for v1. Refused fail-loud on a recipe that is
        // not persistent-capable, so the flag never silently no-ops.
        std::unique_ptr<DecodeGraphCache> graph_cache;
        if (args.persistent_graph) {
            if (!forward_pass->supports_persistent_decode()) {
                std::cerr << "--persistent-graph: architecture '"
                          << cmp_meta.architecture << "' is not persistent-"
                          << "capable (needs qwen35/qwen36/gemma3)\n";
                return 1;
            }
            enable_persistent_decode(forward_pass.get());
            graph_cache = std::make_unique<DecodeGraphCache>(model, forward_pass.get());
        }

        // Hybrid-safety cost accounting (docs/plan-mtp-decode.md §9:
        // per-draft recurrent-checkpoint cost is a first-class number).
        double dn_checkpoint_ms = 0.0, dn_refeed_ms = 0.0;
        int    dn_restores = 0;

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

                // Normal decode for current token. MTP additionally captures
                // the pre-final-norm hidden (D3) — the head's conditioning.
                std::vector<int32_t> current_token_vec = { next_token_id };
                int decode_pos = forward_pass->get_cache_pos(slot);

                std::vector<float> last_hidden;
                std::vector<float> decode_logits = forward_pass->run_prefill(
                    current_token_vec, decode_pos, slot, scheduler,
                    mtp_mode ? &last_hidden : nullptr);
                int after_decode_pos = forward_pass->get_cache_pos(slot);

                // Sample FIRST: y is both the fallback token and the
                // first-token guard — draft[0] occupies y's position, so a
                // draft that disagrees with y must not be accepted. (Grammar
                // is mutually exclusive with speculative, so no accept_token
                // bookkeeping here.)
                last_token_logits.assign(decode_logits.begin(),
                                         decode_logits.begin() + vocab_size);
                int32_t y = sampler->sample(last_token_logits, tokens, vocab);

                // Hybrid safety: verify advances the recurrent state over ALL
                // draft tokens, and overwrite semantics can't rewind — so
                // checkpoint before, restore + refeed the accepted prefix on
                // partial reject (feed_tokens re-advances KV and recurrent
                // together, head-less). Pure-attention recipes: dn == nullptr,
                // all of this is skipped.
                DeltaNetState* dn = forward_pass->snapshot_recurrent();
                CheckpointId dn_cp = kInvalidCheckpoint;
                if (dn) {
                    auto t0 = Clock::now();
                    dn_cp = dn->checkpoint((int)slot);
                    dn_checkpoint_ms += std::chrono::duration<double, std::milli>(
                        Clock::now() - t0).count();
                }

                auto result = spec->try_speculative_step(
                    prompt_tokens_for_pld,
                    generated_tokens,
                    slot,
                    after_decode_pos,
                    bridge.make_verify(slot),
                    bridge.make_rewind(slot),
                    eos_token_id,
                    last_hidden,
                    /*expected_first=*/y);

                if (result.attempted() && result.total_tokens() > 0) {
                    const int accepted_n = (int)result.accepted_tokens.size();
                    if (dn) {
                        if (accepted_n < result.draft_length) {
                            // Rejected tokens polluted the recurrent state:
                            // rewind KV to pre-verify, restore the checkpoint,
                            // re-advance over the accepted prefix only.
                            auto t0 = Clock::now();
                            forward_pass->set_cache_pos(after_decode_pos, slot);
                            dn->restore(dn_cp);
                            if (accepted_n > 0)
                                forward_pass->feed_tokens(
                                    result.accepted_tokens, slot, scheduler);
                            dn_refeed_ms += std::chrono::duration<double, std::milli>(
                                Clock::now() - t0).count();
                            dn_restores++;
                        }
                        dn->release(dn_cp);
                    }

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
                    // No draft (or draft contradicted y): normal step. The
                    // verify never ran, so the checkpoint is just dropped.
                    if (dn) dn->release(dn_cp);
                    next_token_id = y;
                }
                continue;
            }

            // --- Normal decode path ---
            next_token_id = decode_step(
                forward_pass.get(), scheduler, sampler.get(),
                next_token_id, 0,
                tokens, vocab, vocab_size,
                /*force_dense=*/false, /*forced_run=*/nullptr,
                graph_cache.get());
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
            std::cout << "\n[Speculative Stats — "
                      << (mtp_mode ? "MTP" : "PLD") << "]"
                      << " drafts_attempted=" << s.drafts_attempted
                      << " drafts_found=" << s.drafts_found
                      << " tokens_drafted=" << s.tokens_drafted
                      << " tokens_accepted=" << s.tokens_accepted
                      << " bonus_tokens=" << s.bonus_tokens
                      << " accept_rate=" << (int)(s.acceptance_rate() * 100) << "%"
                      << " tokens_per_step=" << s.tokens_per_step()
                      << std::endl;
            if (dn_checkpoint_ms > 0.0 || dn_restores > 0) {
                std::cout << "[Hybrid rollback] checkpoint_total="
                          << (int)dn_checkpoint_ms << "ms"
                          << " restores=" << dn_restores
                          << " refeed_total=" << (int)dn_refeed_ms << "ms"
                          << std::endl;
            }
        }
        if (mtp_sched) ggml_backend_sched_free(mtp_sched);

        std::cout << "\n--- End of Generation ---" << std::endl;
        return 0;
    }