#pragma once

#include "../loader/chat_template.h"

#include <string>
#include <vector>
#include <cstdint>
#include <sstream>

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
    bool chat_mode = false;
    std::string vocab_prune_list_path;
    uint32_t context_length = 4096;
    std::string token_log_path;
    std::string grammar_file;
    std::string system_prompt;
    // TurboQuant KV cache compression (0 = disabled, 2/3/4 = bit width)
    int kv_quant_bits = 0;
    // SnapKV post-prefill KV eviction (0 = disabled)
    uint32_t snapkv_budget = 0;
    uint32_t snapkv_window = 32;
    // PLD options
    bool speculative = false;
    int pld_ngram_size = 3;
    int pld_max_draft = 5;
};

inline std::string make_readable(std::string str) {
        size_t pos = 0;
    while ((pos = str.find("\xC4\xA0", pos)) != std::string::npos) {
        str.replace(pos, 2, " ");
        pos += 1;
    }
    return str;
}

inline void print_token(std::string str) {
    std::cout << make_readable(str) << std::flush;
}
