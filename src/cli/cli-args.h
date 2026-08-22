#pragma once

#include "../loader/chat_template.h"

#include <string>
#include <vector>
#include <cstdint>
#include <sstream>
#include <iostream>

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
    // Gemma 4 channel stream: show the model's thought channel (reasoning) in
    // chat, rendered dimmed, instead of suppressing it. On by default; the
    // thought is shown but never saved into the assistant turn history.
    bool show_thinking = true;  // --hide-thinking to suppress
    // Speculative decoding (docs/plan-mtp-decode.md D5). Bare `--speculative`
    // and `--speculative pld` mean PLD (back-compat); `--speculative mtp`
    // drafts with the model's trained NextN head (MTP-converted GGUFs only).
    bool speculative = false;
    std::string speculative_mode = "pld";  // "pld" | "mtp"
    int pld_ngram_size = 3;
    int pld_max_draft = 5;
    // --mtp-max-draft: tokens drafted BY THE HEAD per step (the sampled token
    // rides the verify batch in addition); upstream guidance: start at 2.
    int mtp_max_draft = 2;
    // Persistent decode graph (docs/plan-persistent-decode-graph.md). Opt-in:
    // build+alloc the decode graph once per n_kv bucket and reuse it across
    // steps, skipping the per-step galloc replan (measured 1.32× on Qwen 3.6,
    // 20 → 27 tok/s). NOT byte-identical to the default exact-n_kv decode —
    // bucketing re-blocks the attention reduction, so it is token-stable
    // modulo ties (same status as speculative). Qwen3.5/3.6 + Gemma3 only
    // (persistent-capable recipes).
    bool persistent_graph = false;  // --persistent-graph
    // Attention KV cache element type. F32 (default) is the byte-identical
    // historical behaviour; F16 halves KV memory and is token-stable but not
    // byte-identical, so receipts baselines are per-kv-type. Recurrent
    // (DeltaNet/SSM) state is unaffected -- it is always F32.
    bool kv_f16 = false;            // --kv-f16
    // Vision: a single image applied to the first user turn. Both
    // must be set together; requires a Gemma multimodal-capable tokenizer.
    std::string image_path;    // --image: JPEG/PNG to attach to the first turn
    std::string mmproj_path;   // --mmproj: Gemma vision projector GGUF
    // Vision V1 (docs/plan-session-snapshot.md): opt-in disk-backed embedding
    // cache. A recurring image is encoded once per node ever (ViT skip);
    // empty = no persistent cache (per-session in-memory reuse only).
    std::string image_embed_cache_dir;  // --image-embed-cache: directory for cached embeddings
    // L2 prefix library (docs/plan-session-snapshot.md, Phase 4). Opt-in,
    // version-gated warm-prefix KV cache for a recurring system prompt: prefill
    // it once, reuse the blob to skip the prefill on later runs. Empty = off.
    std::string prefix_cache_dir;   // --prefix-cache: dir for warm system-prompt KV blobs
    // Vision V2 (docs/plan-session-snapshot.md): opt-in image-prefix KV cache.
    // The image turn's KV up to and including the image span is stored keyed by
    // (preceding context + image content_id); a later run with the SAME image +
    // context skips BOTH the ViT encode and the image-position prefill. Empty =
    // off.
    std::string image_prefix_cache_dir;  // --image-prefix-cache: dir for warm image-prefix KV blobs
    // L1 portable session snapshot (docs/plan-session-snapshot.md, Phase 2).
    std::string save_session_path;  // --save-session: write an L1 snapshot mid-decode
    std::string load_session_path;  // --load-session: resume a saved L1 snapshot
    int save_session_at = 4;        // --save-session-at: generated-token index to snapshot
};

inline std::string make_readable(std::string str) {
        size_t pos = 0;
    while ((pos = str.find("\xC4\xA0", pos)) != std::string::npos) {
        str.replace(pos, 2, " ");
        pos += 1;
    }
    // NOTE: Gemma 4 channel/turn control markers are removed upstream by the
    // ChannelFilter in run_chat (loader/channel_filter.h) before print_token is
    // called, so no tag-stripping is needed here.
    return str;
}

inline void print_token(std::string str) {
    std::cout << make_readable(str) << std::flush;
}
