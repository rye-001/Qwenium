#pragma once

#include "engine/model.h"
#include "tokenizer_config.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <utility>
#include <regex>
#include <functional>
#include <mutex>

// Custom hash for std::pair, enabling its use as a key in std::unordered_map.
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class Tokenizer {
    friend class TokenizerTest_TestUnknownCharacters_Test;
public:
    Tokenizer(const ModelMetadata* metadata, const TokenizerConfig& config = {});

    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(int32_t token_id) const;
    std::string decode(const std::vector<int32_t>& token_ids) const;
    int32_t get_special_token_id(const std::string& token_str) const;
    int32_t get_eos_token_id() const;

    // Vocabulary access
    const std::vector<std::string>& get_vocabulary() const;

private:
    std::string _decode_original_token(int32_t original_id) const;

    // Configuration supplied at construction (arch-specific knobs).
    TokenizerConfig config_;

    // Core BPE components
    const ModelMetadata* metadata_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::pair<int32_t, int32_t>, std::pair<int, int32_t>, pair_hash> merges_;
    
    // BPE cache (thread-safe)
    mutable std::unordered_map<std::string, std::vector<int32_t>> bpe_cache_;
    mutable std::mutex bpe_cache_mutex_;

    // Special tokens management
    int32_t unk_token_id_;
    std::unordered_set<int32_t> special_token_ids_;
    std::unordered_map<std::string, int32_t> special_tokens_;
    
    // ── Pre-tokenization ─────────────────────────────────────────────────────
    // Which pattern to segment with. This is a property of the CHECKPOINT, not
    // of the architecture: the GGUF states it in `tokenizer.ggml.pre`, and two
    // models of the same family can differ (Qwen3-1.7B says "qwen2",
    // Qwen3.5/3.6/3.8 say "qwen35"). Gpt2 is the fallback for any value we have
    // not measured, so an unknown checkpoint keeps today's behaviour rather
    // than silently acquiring someone else's segmentation.
    //
    // Qwen2 and Qwen35 differ only in whether combining marks join a letter
    // run: Qwen35 uses [\p{L}\p{M}]+, Qwen2 uses \p{L}+.
    enum class PreTokenizerKind { Gpt2, Qwen2, Qwen35 };
    PreTokenizerKind pre_kind_ = PreTokenizerKind::Gpt2;

    // Gpt2 only. The Qwen kinds cannot use std::regex at all — see
    // pretokenize_qwen().
    std::regex pretokenization_regex_;

    std::vector<std::string> pretokenize(const std::string& text) const;
    std::vector<std::string> pretokenize_gpt2(const std::string& text) const;
    std::vector<std::string> pretokenize_qwen(const std::string& text) const;
    // Emits special tokens as whole pre-tokens and hands the gaps to `scan`.
    std::vector<std::string> split_on_special_tokens(
        const std::string& text,
        const std::function<void(const std::string&, std::vector<std::string>&)>& scan) const;
    
    // BPE core algorithm
    std::vector<int32_t> apply_bpe(const std::vector<int32_t>& byte_tokens) const;
    
    // Special token handling
    void initialize_special_tokens();
    std::vector<int32_t> encode_with_special_tokens(const std::string& text) const;
    std::vector<int32_t> encode_single_token(const std::string& text) const;
    bool is_special_token(int32_t token_id) const;
    
    // UTF-8 byte mapping
    std::vector<std::string> byte_encoder_;
    std::unordered_map<std::string, int32_t> byte_decoder_;
    void initialize_byte_mapping();

    // ── SentencePiece-derived tokenizer (NormalizerKind::SpaceToUnderscore) ──
    // When config_.normalizer == SpaceToUnderscore, the GPT-2 byte-level BPE
    // path is bypassed in favor of score-based merge over ▁-normalized text
    // with byte-fallback for unknown codepoints.
    bool                    is_llama_tokenizer_ = false;
    std::vector<int32_t>    byte_fallback_ids_;     // 256 entries when llama
    std::vector<float>      scores_;                // alias of metadata_->scores
    void initialize_llama_byte_fallback();
    std::vector<int32_t>    encode_llama(const std::string& text) const;
    std::string             decode_llama(int32_t token_id) const;
};