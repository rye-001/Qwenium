#pragma once

#include <string>
#include <vector>

// How the tokenizer normalizes whitespace before BPE encoding.
// None     — GPT-2 byte-level BPE; spaces are encoded via the byte encoder.
// SpaceToUnderscore — SentencePiece / Llama style: each space is replaced
//   with U+2581 (▁) before segmentation; decode reverses the substitution.
enum class NormalizerKind {
    None,
    SpaceToUnderscore,
};

// Per-architecture tokenizer parameters.  Plain data — no logic.
// Populated by each recipe and stored in the model registry.
//
// normalizer         — whitespace normalization algorithm (see above).
// byte_fallback      — true for Llama/Gemma: unknown codepoints decompose
//                      into <0xNN> byte tokens instead of UNK.
// add_bos_token      — prepend BOS when encoding a prompt.
// extra_chat_specials — token strings to force-promote to the special-token
//                      table even when the GGUF marks them as NORMAL type.
//                      Needed for GGUF exports that mislabel chat-control
//                      tokens (e.g. some gemma-2b-it-v1.1 exports).
struct TokenizerConfig {
    NormalizerKind           normalizer          = NormalizerKind::None;
    bool                     byte_fallback       = false;
    bool                     add_bos_token       = false;
    std::vector<std::string> extra_chat_specials;
};
