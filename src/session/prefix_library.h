#pragma once
// prefix_library.h — Phase 4 L2 service surface (docs/plan-session-snapshot.md).
//
// A bounded, disk-backed library of warm KV-prefix blobs keyed by prompt hash.
// A recurring prefix (e.g. a system prompt) is prefilled once, its slot KV span
// serialized via KvCacheSection into a SessionManifest blob, and stored here
// under key = key_for(prefix_tokens). On a later run/process the blob is loaded,
// memcpy'd into a fresh slot (NO prefill), and the user turn starts at
// pos = len(prefix) — skipping the prefix prefill entirely.
//
// EXPLICIT, OPT-IN, VERSION-GATED — and deliberately the HARD-FAIL counterpart
// of PersistentImageEmbeddingStore. That store is opportunistic: a header
// mismatch is a MISS (re-encode is always the correct, cheap answer). Here a
// header mismatch is REFUSED FAIL-LOUD: a prefix blob built under a different
// model / quant / kernel path would resume into a WRONG result, so it must never
// be used — and must never be SILENTLY re-prefilled either (that reopens the F9
// transparency problem the deleted server-side prefix cache caused). The opt-in
// flag makes reuse explicit; a stale/foreign blob is a loud error, not a quiet
// slow-path. See docs/plan-session-snapshot.md (Phase 4, "Non-negotiable").
//
//   absent key                 -> Miss   (false): first use; caller prefills + store()s
//   present, header matches     -> Hit    (true) : caller restores via SessionManifest
//   present, header MISMATCHES   -> THROWS        : refused; never silently re-prefilled
//
// The library only owns the key->file mapping, bounded eviction, atomic publish,
// and the fail-loud header gate. KV (de)serialization stays co-located in
// KvCacheSection (the owning module); the caller builds/restores the manifest.

#include <cstdint>
#include <string>
#include <vector>

#include "session/compat_header.h"

class PrefixLibrary {
public:
    // `dir` is created if absent. `expected` is THIS node's identity (arch +
    // shape + quant + build_path_tag + weights_hash); a stored blob's embedded
    // header must match it to be reused. `max_entries` bounds the library
    // (oldest-by-mtime evicted on store when full); 0 = unbounded.
    PrefixLibrary(std::string dir, qinf::session::CompatHeader expected,
                  size_t max_entries = 16);

    // Stable content key for a prefix token span: FNV-1a over the int32 token
    // bytes (little-endian). PURE — a given span always maps to the same key, on
    // any host. Empty span returns 0 (the non-cacheable sentinel; see store()).
    static uint64_t key_for(const std::vector<int32_t>& prefix_tokens);

    // Stable content key for an IMAGE-prefix span (vision V2,
    // docs/plan-session-snapshot.md): FNV-1a over the preceding-context token
    // bytes (everything BEFORE the image span — must be fixed for reuse) mixed
    // with the image content_id (Bitmap::content_id, FNV of normalized pixels).
    // The soft-token placeholder ids are not hashed: they are identical for any
    // image, so the image content_id is what discriminates. PURE / host-
    // independent. content_id 0 (an unset bitmap) returns 0 (non-cacheable);
    // empty preceding context is still cacheable when the image is set (an image
    // that opens the turn). The model/quant/kernel-path version is enforced by
    // the embedded CompatHeader at load, not folded into the key.
    static uint64_t key_for(const std::vector<int32_t>& preceding_tokens,
                            uint64_t image_content_id);

    // Hit -> true + fills `out` with the stored blob (caller restores it via
    // SessionManifest::restore, which re-validates the header harmlessly).
    // Miss (absent) -> false. Present-but-incompatible -> THROWS fail-loud,
    // naming the mismatched header field (build_path_tag / weights_hash / arch /
    // shape). key 0 always misses (never touches disk).
    bool try_load(uint64_t key, std::vector<uint8_t>& out) const;

    // Persist `blob` (a full SessionManifest capture — header + sections) under
    // `key`, atomically (temp file + rename). Evicts the oldest entry first when
    // at capacity. No-op for key 0. The blob must already embed a CompatHeader
    // identical to `expected` (the caller built it from this same node).
    void store(uint64_t key, const std::vector<uint8_t>& blob) const;

private:
    std::string path_for(uint64_t key) const;
    void evict_if_full() const;

    std::string dir_;
    qinf::session::CompatHeader expected_;
    size_t max_entries_;
};
