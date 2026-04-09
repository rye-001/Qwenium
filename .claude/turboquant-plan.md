## TurboQuant KV Cache Compression - Implementation Plan

**Technique**: Google Research, ICLR 2026, arXiv 2504.19874
**Mode**: Memory-only (quantize on write, dequantize to F32 on read, standard ggml attention)

---

### Critical Architecture Constraint

The forward pass builds one monolithic ggml graph across all layers. KV write (`cpy_k`/`cpy_v`) and read (`get_k`/`get_v`) are ggml ops in the same graph (forward-pass.cpp L437-447). Cannot intercept between write and read within a single compute. This forces a two-phase approach.

---

### Phase 1 - Compressed Backing Store

F32 cache stays full-size. Compressed store runs alongside for **fast `clone_slot`** (prefix caching) and as foundation for Phase 2.

**Flow:**
```
graph compute (F32 cache, as today)
  -> post_compute_hook: compress new tokens from F32 into byte buffer
  -> clone_slot: memcpy compressed bytes (~5x faster than F32 tensor copy)
```

Memory: no savings yet. But `clone_slot` gets ~5x faster.

### Phase 2 - Per-Layer Compute (real memory savings)

Restructure forward pass to build+compute one layer at a time. Shared F32 scratch tensors hold only 1 layer's KV. Before each layer: decompress history. After: compress new tokens.

**Memory savings (Qwen3, 28 layers, 4096 ctx, 1024 KV dim):**

| | F32 (current) | TQ 4-bit |
|---|---|---|
| Per slot | ~896 MB | ~158 MB |
| 10 slots | ~8.7 GB | ~1.5 GB |

---

### New Files

| File | Purpose | ~Lines |
|---|---|---|
| `src/kv-cache/turboquant.h` | WHT + Lloyd-Max quantize/dequantize API | 80 |
| `src/kv-cache/turboquant.cpp` | Implementation | 200 |
| `src/kv-cache/compressed-kv-cache.h` | Compressed cache wrapping `simple_kv_cache` | 90 |
| `src/kv-cache/compressed-kv-cache.cpp` | Compressed store + compress/decompress hooks | 250 |
| `tests/unit/test_turboquant.cpp` | WHT round-trip, quantize MSE, block layout | 150 |

### Modified Files

| File | What changes |
|---|---|
| `forward-pass.cpp` | Add `post_compute_hook()`, Phase 2: per-layer loop |
| `forward-pass.h` | Store `compressed_kv_cache`, virtual hook |
| `forward-pass-factory.cpp` | Thread `TurboQuantConfig` through |
| `cli-args.h` | Add `--kv-quant-bits {0,2,3,4}` |
| `src/cli/main.cpp` | Parse flag |
| `src/kv-cache/CMakeLists.txt` | Add new sources |
| `tests/CMakeLists.txt` | Add new tests |

---

### The Math (`turboquant.cpp`)

**1. Fast Walsh-Hadamard Transform** - per KV head (128 dims, power of 2):
```
fwht_inplace(float* data, int n):   // O(n log n), in-place
    for len = 1; len < n; len <<= 1:
        butterfly passes (u+v, u-v)
    normalize by 1/sqrt(n)
```
Inverse WHT = same function (self-inverse when normalized). Spreads outlier energy uniformly.

**2. Lloyd-Max scalar quantization** - precomputed centroids for N(0,1):
- 4-bit: 16 centroids (static constexpr table)
- 3-bit: 8 centroids
- 2-bit: 4 centroids ~ {+-0.4528, +-1.510}

No calibration data needed. After WHT, components are approximately Gaussian by CLT.

**3. Block layout** (block_size=32):
```
[2 bytes F16 scale | ceil(32 * bits / 8) bytes packed indices]
```
- 4-bit: 18 bytes/block -> 7.1x compression vs F32
- 2-bit: 10 bytes/block -> 12.8x compression

---

### Interception Points

| Operation | Location | Interception |
|---|---|---|
| KV write | `_build_attention_layer` L437-443 | After `graph_compute`: read F32 via `ggml_backend_tensor_get`, compress |
| KV read | `_build_attention_layer` L446-447 | Phase 2: before graph build, decompress via `ggml_backend_tensor_set` |
| Batched KV read | `_build_batched_attention_layer` L397-398 | Same pattern via `gather_k`/`gather_v` |
| Clone | `simple-kv-cache.cpp` L136-169 | `memcpy` compressed bytes instead of F32 tensor copy |

---

### CLI

```
--kv-quant-bits 4    # TurboQuant 4-bit (recommended, quality-neutral)
--kv-quant-bits 2    # Aggressive, minor quality loss
                     # 0 = disabled (default)
```

---

### Test Plan

1. WHT round-trip: random vectors -> WHT -> inverse WHT -> verify eps < 1e-5
2. Quantize MSE: Gaussian vectors -> compress -> decompress -> verify SQNR ~20dB (4-bit)
3. Clone correctness: clone slot, verify compressed bytes match
4. E2E generation: same prompt with/without TQ, compare output quality
5. Memory benchmark: measure actual RSS with 10 slots

---

### Execution Order

1. `turboquant.h/cpp` + unit tests (standalone, testable immediately)
2. `compressed-kv-cache` Phase 1 (wrap existing cache, compress-after-compute)
3. Wire up CLI flag + factory
4. Phase 2 per-layer refactor (the big memory win)

**Estimate: ~770 new lines + ~80 modified. Phase 1: 2-3 days, Phase 2: 3-5 days.**

---

### Risks

1. **Per-layer compute overhead (Phase 2)**: Multiple `ggml_backend_graph_compute` calls may add overhead on Metal. Mitigate by batching 2-4 layers per compute.
2. **Non-power-of-2 dims**: Qwen3 head dim is 128 (ok). Future models: pad to next power of 2.
3. **Backend round-trip**: `ggml_backend_tensor_get/set` copies to/from CPU. Acceptable since graph already computed.
4. **No QJL stage**: Community found QJL hurts in practice. We skip it (MSE-only, stage 1).
