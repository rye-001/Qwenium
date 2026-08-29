#pragma once
// vocab_utils.h — the vocab-pruning keep list.
//
// Responsibility: read keep_list.bin, the set of token ids a run is allowed to
//   emit. One function, load_keep_list(); the file format is
//   [count:uint32][id:uint32]*, and any short read throws naming what failed.
// Who uses it: the CLI's --vocab-prune-list-path flag (cli/main, chat, complete)
//   hands the resulting set to the sampler as a mask. Nothing in the engine
//   reads it — pruning is a caller policy, not an engine behaviour.
// Header-only and STL-only on purpose, so a consumer needs neither ggml nor a
//   loaded model to read a keep list.
// Exercised by: tests/integration/test_parallel_grammar.cpp and
//   tests/integration/test_parallel_order_management.cpp.

#include <unordered_set>
#include <string>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace qwenium {

// Load keep_list.bin: [count(uint32)] [id(uint32), id, ...]
inline std::unordered_set<int32_t> load_keep_list(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open keep_list: " + path);
    }
    
    uint32_t count = 0;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!file) {
        throw std::runtime_error("Failed to read keep_list count");
    }
    
    std::unordered_set<int32_t> result;
    result.reserve(count);
    
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t token_id = 0;
        file.read(reinterpret_cast<char*>(&token_id), sizeof(token_id));
        if (!file) {
            throw std::runtime_error("Failed to read token_id at index " + std::to_string(i));
        }
        result.insert(static_cast<int32_t>(token_id));
    }
    
    return result;
}

} // namespace qwenium