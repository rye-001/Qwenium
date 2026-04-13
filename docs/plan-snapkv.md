# SnapKV Implementation Plan

**Paper:** SnapKV: LLM Knows What You are Looking for Before Generation (arXiv:2404.14469)  
**Goal:** After prefill, evict unimportant KV positions per-head to reduce cache size and speed up decode attention.

---

## How SnapKV Works (30-second summary)

1. Run full prefill normally (all tokens).
2. Look at the attention weights from an **observation window** (last W tokens, typically W=32).
3. For each attention head, pool/vote across the window to find which KV positions get the most attention.
4. Keep the top-B positions per head (budget B, e.g. 1024) + the full observation window.
5. Discard everything else from the KV cache.
6. Decode runs on the reduced cache — fewer keys = faster softmax + less memory.

Key property: this is a **one-shot post-prefill filter**. No changes to the model, no training, no changes to decode logic.

---

## Architecture Considerations

### Toggle hierarchy

SnapKV must be independently toggleable and compose with TurboQuant:

```
--kv-quant-bits 0                    → F32 KV, no SnapKV
--kv-quant-bits 4                    → TQ only
--snapkv-budget 1024                 → SnapKV only (F32 KV, evicted)
--kv-quant-bits 4 --snapkv-budget 1024 → SnapKV + TQ (evict first, then compress survivors)
```

CLI flag: `--snapkv-budget N` (0 = disabled, default).  
Optional: `--snapkv-window W` (observation window size, default 32).

### Where it plugs in

SnapKV runs **once, between prefill and the first decode step**. The insertion point is clear:

```
run_prefill(prompt_tokens, ...)      ← full prefill, all tokens
snapkv_evict(slot_idx)               ← NEW: evict unimportant KV positions
advance_cache(...)
// decode loop begins — sees reduced cache
```

### Which architectures

- **Qwen2/3:** All 48 layers are attention. SnapKV applies to all.
- **Qwen3.5:** Only 6 attention layers have KV cache. SnapKV applies to those 6 only. SSM state is unaffected.

---

## Implementation Plan

### Phase 0: CLI + Config Plumbing

**Files:** `cli-args.h`, `main.cpp`, `forward-pass-base.h`, `forward-pass-factory.h`

1. Add `snapkv_budget` and `snapkv_window` to `CliArgs`.
2. Add `--snapkv-budget N` and `--snapkv-window W` to CLI parser.
3. Store in forward pass (either as member or passed to a new method).

### Phase 1: Attention Score Extraction

**The problem:** SnapKV needs the attention weight matrix `softmax(Q @ K^T / sqrt(d))` from the last few layers' prefill pass. Currently, `build_attn_mha` computes `kq_soft` (the post-softmax attention weights) inside the ggml graph, but we never read it back — it's consumed in-graph by `V @ kq_soft`.

**Approach — separate scoring pass:**

Rather than modifying the main prefill graph (which would complicate the hot path for everyone), run a **lightweight scoring pass** after prefill completes:

1. For the last L attention layers (SnapKV paper uses L=1, just the final layer):
   - Build a small ggml graph: `Q_obs @ K_full^T → softmax → attention_scores`
   - Q_obs = queries from just the observation window (last W tokens)
   - K_full = full key cache (all prefill tokens)
   - Output: `attention_scores [n_heads, n_kv, W]`
2. Read back `attention_scores` to CPU.
3. For each head: sum/pool across the W observation queries → importance score per KV position.
4. Keep top-B positions per head + the full observation window.

**Why a separate pass:** 
- Zero impact on the non-SnapKV path.
- The scoring graph is tiny (just one matmul + softmax per layer, no V multiplication, no FFN).
- Works identically for TQ and non-TQ paths — the scratch/KV cache is already populated after prefill.

**Files:** New method on `ForwardPassBase`:
```cpp
// Returns per-head importance scores: [n_layers][n_heads][n_kv_positions]
std::vector<std::vector<std::vector<float>>> compute_snapkv_scores(
    uint32_t slot_idx, uint32_t obs_window,
    ggml_backend_sched_t scheduler);
```

### Phase 2: KV Eviction

**The core data structure change:** After eviction, positions in the KV cache are no longer contiguous. Each head retains a different subset (in general). Two approaches:

#### Approach A: Per-Head Index Maps (full SnapKV)
Each head has its own retained-position list. Attention gathers K/V via index indirection. This is the paper's approach but requires significant changes to the attention builder (gather ops).

#### Approach B: Uniform Eviction (simpler, still effective)  
Take the **union** of important positions across all heads in a layer, keep those. All heads in a layer see the same reduced KV. This:
- Preserves the existing contiguous KV layout (just fewer tokens).
- Requires no gather ops — just compact the cache.
- Loses some per-head selectivity but is much simpler.
- The paper's ablations show uniform eviction retains most of the benefit.

**Recommendation: Start with Approach B.** It's simpler, composes trivially with TQ, and we can upgrade to A later if needed.

**Implementation (Approach B — uniform eviction):**

New file: `src/kv-cache/snapkv-eviction.h`

```cpp
struct SnapKVResult {
    // Per-layer: sorted list of retained KV positions (union across heads).
    std::vector<std::vector<uint32_t>> retained_positions;  // [n_layers][variable]
    uint32_t original_length;   // pre-eviction sequence length
};

SnapKVResult compute_eviction_mask(
    const std::vector<std::vector<std::vector<float>>>& scores,  // [layers][heads][positions]
    uint32_t budget,
    uint32_t obs_window);
```

Algorithm per layer:
1. For each head, find top-B positions by importance score.
2. Take union across heads.
3. Always include the observation window (last W positions).
4. Sort retained positions.

### Phase 3: Cache Compaction

After computing `SnapKVResult`, compact the KV cache so retained positions are contiguous.

#### Non-TQ path (simple_kv_cache):
- Read retained K/V rows from the Metal buffer (via `tensor_get`).
- Write them back contiguously starting at position 0 (via `tensor_set`).
- Update cache position to `retained_count`.

#### TQ path (CompressedKVStore):
- Compact the compressed byte arrays in-place: copy retained tokens' compressed data to positions `[0..retained_count)`.
- Invalidate scratch watermarks (force full re-decompress on next decode step, since positions shifted).
- Update store position to `retained_count`.

New method on `CompressedKVStore`:
```cpp
// Compact: keep only the positions listed in `retained` (sorted, unique).
// Moves compressed data so retained[i] becomes position i.
void compact(uint32_t layer, uint32_t slot,
             const std::vector<uint32_t>& retained_positions);
```

New method on `simple_kv_cache`:
```cpp
void compact(uint32_t layer, uint32_t slot,
             const std::vector<uint32_t>& retained_positions,
             uint32_t new_length);
```

### Phase 4: Position Remapping for RoPE

**Critical correctness issue:** After compaction, token at new position `i` was originally at position `retained_positions[i]`. RoPE embeddings during decode must use the **original** positions, not the compacted ones. Otherwise the model sees wrong positional information.

**Solution:** Store a position remap table per slot:
```cpp
// Maps compacted position → original position for RoPE.
std::vector<uint32_t> rope_position_map_;  // [n_ctx]
```

Before SnapKV: `rope_position_map_[i] = i` (identity).  
After SnapKV: `rope_position_map_[i] = retained_positions[i]`.

During decode, when setting `inp_pos` for a new token at compacted position `p`:
- The new token's RoPE position = its true sequence position (p_original).
- The cached tokens' RoPE positions are baked into K already (RoPE was applied during prefill before eviction).

Actually — **RoPE is already baked into K during prefill.** The K values in the cache already have their positional encoding applied. So compacting the cache doesn't break RoPE for existing tokens. The only thing we need is: when writing a new decode token's K at compacted position `new_len`, its RoPE position must be `original_seq_len` (not `new_len`). 

This means `get_cache_pos` for decode should return the **original** sequence length, not the compacted one. We need two position counters:
- `cache_len_`: physical length of the compacted cache (for attention mask sizing, K/V write offset).
- `seq_pos_`: logical sequence position for RoPE (continues from original prefill length).

### Phase 5: Integration into run_prefill / decode

**Prefill (in `run_prefill`):**
```
1. Run full prefill (existing code, unchanged)
2. If snapkv_budget > 0:
   a. compute_snapkv_scores(slot, obs_window, scheduler)
   b. compute_eviction_mask(scores, budget, obs_window)
   c. For each layer: compact(layer, slot, retained_positions)
   d. Store seq_pos = original prefill length
   e. Update cache_len = retained_count
   f. Invalidate scratch watermarks (TQ path)
3. advance_cache / return logits (existing)
```

**Decode (unchanged!):**
The attention builder already reads `n_kv = cache_pos + n_tokens` tokens from the KV cache. After compaction, `cache_pos` is smaller, so attention naturally runs on fewer tokens. No decode changes needed (if position tracking is handled correctly).

### Phase 6: Qwen3.5 Adaptation

Same logic, but only for the 6 attention layers (skip SSM layers). The `kv_layer_map_` already handles the physical→cache layer mapping. SnapKV scoring and eviction use `kv_idx` (0-5), not physical layer index.

---

## File Change Summary

| File | Change |
|------|--------|
| `cli-args.h` | Add `snapkv_budget`, `snapkv_window` |
| `main.cpp` | Parse `--snapkv-budget`, `--snapkv-window` |
| `forward-pass-base.h` | Add `snapkv_budget_`, `snapkv_window_` members; virtual `apply_snapkv()` |
| `forward-pass.h` | Override `apply_snapkv()` for Qwen2/3 |
| `forward-pass-qwen35.h` | Override `apply_snapkv()` for Qwen3.5 |
| `forward-pass.cpp` | Implement scoring pass + call compact in `run_prefill` |
| `forward-pass-qwen35.cpp` | Same, scoped to attention layers |
| `compressed_kv_store.h/cpp` | Add `compact()` method |
| `simple-kv-cache.h/cpp` | Add `compact()` method |
| **NEW** `snapkv-eviction.h/cpp` | Pure C++ scoring→eviction logic (no ggml dep) |
| `http_server.cpp` | Optionally wire through `snapkv_budget` from request JSON |

---

## Testing Strategy

### Unit tests (`test_snapkv_eviction.cpp`)
- Eviction mask computation: verify budget is respected, window is always retained.
- Compact on CompressedKVStore: verify data integrity after compaction.
- Compact on simple_kv_cache: verify K/V data matches retained positions.
- Edge cases: budget >= seq_len (no eviction), budget = 0 (evict all but window).

### Integration test (`test_snapkv_generation.cpp`)
- Compare greedy output: F32 full cache vs F32+SnapKV (budget=large) — should be identical.
- Compare: TQ vs TQ+SnapKV — quality should be close.
- Memory reduction: verify cache_pos after eviction equals expected retained count.

---

## Execution Order

1. Phase 0: CLI plumbing (30 min)
2. Phase 2 + Phase 3: Eviction logic + cache compaction + unit tests (core, no ggml dependency) (2-3 hrs)
3. Phase 1: Attention score extraction (ggml graph work) (2-3 hrs)  
4. Phase 4: Position remapping / dual position counters (1-2 hrs)
5. Phase 5: Integration into run_prefill (1 hr)
6. Phase 6: Qwen3.5 adaptation (1 hr)
7. Integration tests + tuning (1-2 hrs)

**Total estimate: ~2-3 days of focused work.**

---

## Marketing Angle

"**32x KV compression: SnapKV token eviction × TurboQuant 4-bit quantization.**  
The only inference engine combining attention-guided cache pruning with outlier-aware WHT quantization. Not available in llama.cpp."
