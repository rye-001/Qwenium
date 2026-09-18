#pragma once
// routing_trace.h — the expert selection a pass took, as data.
//
// Responsibility: carry one pass's MoE top-k selections so a LATER pass can be
// made to take the same ones. Plain value type; no ggml, no ownership of
// anything but its own buffers.
//
// WHY THIS EXISTS. An MoE router's top-k is an `argmax`: a discrete choice with
// no margin. Perturb the arithmetic by ~1e-4 — a different prefill shape, batch
// width, driver, GPU or ggml build — and it selects a DIFFERENT expert. Measured
// on Qwen3.6-35B-A3B: 1-6% of selections flip and 1 extraction in 15 changes,
// while dense stacks under the same perturbation are token-identical 15/15.
// That is what makes an MoE receipt reproducible per configuration but not
// ACROSS configurations, which is not a receipt.
//
// Replaying the recorded selection removes the discrete channel and leaves only
// the ~1e-2 arithmetic residue that dense models already carry. Measured: a
// pinned 35B lands at worst 1.447e-02 against the dense 9B's own 1.384e-02 —
// the same band, a 57-81x reduction against its own unpinned arm
// (docs/note-moe-cache-transparency.md §8).
//
// WHAT IT DOES NOT BUY. Sameness, not quality. A route flip is roughly as often
// helpful as harmful and no observable router statistic predicts which
// (arXiv 2608.11212: margin predicts THAT a flip happens at AUC 0.772, whether
// it hurts at 0.490 — chance). Replay makes a verify pass agree with the
// extraction it is checking; it does not make either one better, and it
// narrows what verify independently checks to "attention over THIS
// computation" rather than "the model's free behaviour".
//
// Public surface: write() to record, at() to replay, both fail loud.
// State owned: `by_layer`, one row per (layer, position).
// Unit test: tests/unit/test_routing_trace.cpp

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Where a MoE layer's expert selection comes from. Router is every recipe's
// default and is byte-identical to the behaviour that predates this type.
enum class RoutingSource { Router, Replay };

class RoutingTrace {
public:
    // Record layer `il`'s selection for one absolute position. `topk` points at
    // `top_k` expert ids. The first write fixes top_k for the whole trace.
    void write(int il, size_t position, const int32_t* topk, int top_k) {
        if (top_k <= 0)
            throw std::runtime_error(
                "RoutingTrace::write: slot 'top_k' expected a positive count, actual: " +
                std::to_string(top_k));
        if (top_k_ == 0) top_k_ = top_k;
        if (top_k != top_k_)
            throw std::runtime_error(
                "RoutingTrace::write: slot 'top_k' expected " + std::to_string(top_k_) +
                " (fixed by the first write), actual: " + std::to_string(top_k));
        std::vector<int32_t>& rows = by_layer_[il];
        const size_t need = (position + 1) * static_cast<size_t>(top_k_);
        if (rows.size() < need) rows.resize(need, kUnwritten);
        std::copy(topk, topk + top_k_, rows.begin() + position * top_k_);
    }

    // The `top_k` expert ids recorded for layer `il` at absolute `position`.
    // Fails loud rather than returning a default: a replay that silently
    // invented a selection would produce a receipt nobody could trust, which
    // is the exact failure this type exists to remove.
    const int32_t* at(int il, size_t position) const {
        auto it = by_layer_.find(il);
        if (it == by_layer_.end())
            throw std::runtime_error(
                "RoutingTrace::at: expected a recorded selection for layer " +
                std::to_string(il) + ", actual: layer not in the trace (" +
                std::to_string(by_layer_.size()) + " layers recorded)");
        const size_t off = position * static_cast<size_t>(top_k_);
        if (off + static_cast<size_t>(top_k_) > it->second.size())
            throw std::runtime_error(
                "RoutingTrace::at: layer " + std::to_string(il) + " expected at least " +
                std::to_string(off + top_k_) + " recorded ids to reach position " +
                std::to_string(position) + ", actual: " + std::to_string(it->second.size()));
        if (it->second[off] == kUnwritten)
            throw std::runtime_error(
                "RoutingTrace::at: layer " + std::to_string(il) + " position " +
                std::to_string(position) + " expected a recorded selection, actual: "
                "never written (a gap in the trace, not a zero selection)");
        return it->second.data() + off;
    }

    // ── Digests ──────────────────────────────────────────────────────────
    //
    // FNV-1a, the same fold ModelMetadata::weights_hash uses, and deliberately
    // NOT std::hash: std::hash<std::string> is not specified to agree across
    // standard libraries or processes, and a receipt digest that changed with
    // the toolchain would be worse than no digest at all.
    //
    // WHAT A DIGEST IS FOR, and what it is not. Same configuration, it is an
    // exact and nearly free regression detector: did this refactor, ggml bump
    // or kernel patch change which experts run? Across configurations it is
    // NOT a pass/fail gate — 1-6% of selections flip under any perturbation,
    // so the whole-trace digest differs on essentially every cross-config run
    // while the output changes about 1 document in 15. A gate built on
    // equality would cry wolf ~14 times out of 15.
    //
    // That is why per_layer() exists alongside it: a caller comparing two
    // traces can report HOW MUCH diverged and WHERE, which is a magnitude,
    // rather than a boolean that is almost always "differs".
    uint64_t digest() const {
        uint64_t h = kFnvBasis;
        fold(h, static_cast<uint64_t>(top_k_));
        for (const auto& kv : by_layer_) {          // std::map: ascending, stable
            fold(h, static_cast<uint64_t>(kv.first));
            fold(h, static_cast<uint64_t>(kv.second.size()));
            for (int32_t id : kv.second) fold(h, static_cast<uint64_t>(id));
        }
        return h;
    }

    // One layer's digest, so a comparison can localise divergence to a layer
    // without carrying the selections themselves (40 layers x 8 bytes, not
    // 40 x n_tokens x top_k x 2).
    uint64_t digest(int il) const {
        auto it = by_layer_.find(il);
        if (it == by_layer_.end())
            throw std::runtime_error(
                "RoutingTrace::digest: expected a recorded selection for layer " +
                std::to_string(il) + ", actual: layer not in the trace");
        uint64_t h = kFnvBasis;
        fold(h, static_cast<uint64_t>(top_k_));
        fold(h, static_cast<uint64_t>(il));
        fold(h, static_cast<uint64_t>(it->second.size()));
        for (int32_t id : it->second) fold(h, static_cast<uint64_t>(id));
        return h;
    }

    std::map<int, uint64_t> per_layer() const {
        std::map<int, uint64_t> out;
        for (const auto& kv : by_layer_) out[kv.first] = digest(kv.first);
        return out;
    }

    // ── Serialization, and the binding that makes replay SAFE ────────────
    //
    // A trace is only replayable onto the SAME TOKENS. /v1/verify exists to
    // check a SUPPLIED extraction, which the caller is free to have edited —
    // and an edited extraction tokenizes differently, so position p no longer
    // holds the token whose experts were recorded there. Replaying anyway
    // would pin one token's routing onto another and produce a confident,
    // wrong receipt.
    //
    // So a trace carries a digest of the token ids it was captured over, and
    // from_blob()'s consumer must compare it against its own tokens and REFUSE
    // on mismatch. That makes replay a re-verification tool — same report, same
    // tokens, different machine — and explicitly not a way to verify an edited
    // extraction. Nothing else in the payload can express that difference.
    void bind_tokens(const std::vector<int32_t>& tokens) {
        tokens_.assign(tokens.begin(), tokens.end());
        tokens_digest_ = 0;
    }

    // Record the tokens of one batch at their ABSOLUTE positions. Called by the
    // forward pass wherever it sets a graph's inputs while capture is armed, so
    // a trace binds itself to exactly the tokens its captured graphs processed
    // and no caller can forget to. Out-of-order and chunked batches are fine;
    // a position written twice with the same id is a no-op.
    void note_tokens(size_t pos, const std::vector<int32_t>& ids) {
        if (tokens_.size() < pos + ids.size()) tokens_.resize(pos + ids.size(), kUnwritten);
        std::copy(ids.begin(), ids.end(), tokens_.begin() + pos);
        tokens_digest_ = 0;   // invalidate the cache
    }

    // FNV-1a over the bound token ids. Zero-length binding yields 0, which is
    // how "this trace is not bound to any tokens" is expressed.
    uint64_t tokens_digest() const {
        if (tokens_digest_ != 0 || tokens_.empty()) return tokens_digest_;
        uint64_t h = kFnvBasis;
        fold(h, static_cast<uint64_t>(tokens_.size()));
        for (int32_t t : tokens_) fold(h, static_cast<uint64_t>(t));
        tokens_digest_ = h ? h : 1;   // never collide with the unbound sentinel
        return tokens_digest_;
    }
    void set_tokens_digest(uint64_t h) { tokens_digest_ = h; tokens_.clear(); }
    size_t n_tokens_bound() const { return tokens_.size(); }

    // Base64 of a compact binary blob. `max_layer` keeps only layers <= it,
    // which is not a size hack but a statement of reach: a verify pass builds
    // only max(citation_layer, coverage_layer)+1 blocks and can never consult a
    // deeper one, so carrying deeper layers would be carrying what the consumer
    // provably cannot use. Negative keeps every layer.
    std::string to_blob(int max_layer = -1) const;
    static RoutingTrace from_blob(const std::string& b64);

    bool   empty() const { return by_layer_.empty(); }
    int    top_k() const { return top_k_; }
    size_t n_layers() const { return by_layer_.size(); }
    size_t n_positions(int il) const {
        auto it = by_layer_.find(il);
        return it == by_layer_.end() || top_k_ == 0
                   ? 0 : it->second.size() / static_cast<size_t>(top_k_);
    }
    const std::map<int, std::vector<int32_t>>& by_layer() const { return by_layer_; }
    void clear() { by_layer_.clear(); top_k_ = 0; }

private:
    // A real expert id is never negative, so an unwritten slot is
    // distinguishable from a recorded selection of expert 0.
    static constexpr int32_t kUnwritten = -1;
    static constexpr uint64_t kFnvBasis = 1469598103934665603ull;

    static void fold(uint64_t& h, uint64_t v) {
        for (int b = 0; b < 8; ++b) {
            h ^= static_cast<uint8_t>(v >> (8 * b));
            h *= 1099511628211ull;
        }
    }

    std::map<int, std::vector<int32_t>> by_layer_;
    int top_k_ = 0;
    std::vector<int32_t> tokens_;
    mutable uint64_t tokens_digest_ = 0;
};
