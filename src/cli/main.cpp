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

#include "complete.h"
#include "chat.h"

#include "../qwen3-core/qwen3-model.h"
#include "../qwen3-core/gguf-loader.h"
#include "../qwen3-core/forward-pass-factory.h"
#include "../qwen3-core/tokenizer.h"
#include "../sampling/sampling.h"
#include "../sampling/speculative.h"
#include "../sampling/vocab_utils.h"
#include "../sampling/grammar_vocab.h"

// #include "./tensor_tracer.h"

void debug_computed_tensor(ggml_tensor* tensor, const char* name);
size_t calculate_scratch_size(const Qwen3Metadata& metadata,
                               size_t max_context_length = 2048,
                               size_t batch_size = 512);

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
    std::cout << "  --pld-max-draft K       PLD max draft tokens (default: 5)\n";
    std::cout << "  --snapkv-budget N       SnapKV: keep top-N KV positions per head after prefill (0 = disabled)\n";
    std::cout << "  --snapkv-window W       SnapKV: observation window size (default: 32)\n\n";
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
        } else if (arg == "--kv-quant-bits") {
            if (i + 1 >= argc) return false;
            args.kv_quant_bits = std::stoi(argv[++i]);
        } else if (arg == "--speculative") {
            args.speculative = true;
        } else if (arg == "--pld-ngram") {
            if (i + 1 >= argc) return false;
            args.pld_ngram_size = std::stoi(argv[++i]);
        } else if (arg == "--pld-max-draft") {
            if (i + 1 >= argc) return false;
            args.pld_max_draft = std::stoi(argv[++i]);
        } else if (arg == "--snapkv-budget") {
            if (i + 1 >= argc) return false;
            args.snapkv_budget = std::stoi(argv[++i]);
        } else if (arg == "--snapkv-window") {
            if (i + 1 >= argc) return false;
            args.snapkv_window = std::stoi(argv[++i]);
        } else if (args.model_path.empty()) {
            args.model_path = arg;
        }
        i++;
    }
    
    return args.run_test || !args.model_path.empty();
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
    std::unique_ptr<qwen3::GrammarVocab> grammar;
    if (!args.grammar_file.empty()) {
        std::ifstream file(args.grammar_file);
        if (!file) {
            std::cerr << "Error: Could not open grammar file: " << args.grammar_file << std::endl;
            return 1;
        }
        std::string grammar_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        grammar = qwen3::GrammarVocab::parse_impl(grammar_str);
        if (!grammar) {
            std::cerr << "Error: Failed to parse grammar file." << std::endl;
            return 1;
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
        run_chat(model, args, grammar, spec.get(), use_speculative, log_token, log_tokens);
    } else if (!args.prompt.empty()) {
        return run_complete(model, args, grammar, spec.get(), use_speculative, log_token, log_tokens);
    } else {
        std::cout << "\nModel loaded successfully. Provide a prompt to start generation."
;    }
    
    return 0;
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