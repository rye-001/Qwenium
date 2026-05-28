# Prior Art

Pointers to external work relevant to specific Phase 4 / Phase 5 PRs.
Not commitments — scoping notes only. Read before implementing the PR
they are tagged to.

---

## ik_llama.cpp (Iwan Kawrakow's llama.cpp fork)

CPU/CUDA focus, no Metal. Relevant slices:

### Fused MoE graph shape — PR 4.1 DROPPED

**Status:** not load-bearing for our codebase. Retained for context only.

Originally read as "single fused custom op vs chained `mul_mat_id` per
expert eliminates launch overhead." Two facts make this not apply to
us — see [docs/phase4-investigation.md](phase4-investigation.md):

1. `ik_llama.cpp`'s baseline graph chains one `mul_mat` per expert (or
   per active expert × top-k). Our `src/layers/moe.cpp` issues exactly
   three batched `ggml_mul_mat_id` calls per layer (gate / up / down),
   each backed by a native Metal `MUL_MAT_ID` kernel. Their fix doesn't
   address a problem we have — C-level launch count is already O(1) in
   `n_experts` for us.
2. The `ggml_custom_op` implementation route doesn't reach the Metal
   backend in this ggml tag. Custom ops fall through to CPU, which would
   slow MoE down rather than speed it up.

If MoE-side optimization is ever revisited, the candidate is patching
ggml-metal's `MUL_MAT_ID` kernel for sparse routing. The CUDA kernel in
`ggml-cuda/moe.cu` is not the reference for that — graph shape and
kernel are both wrong starting points for our case.

### Hadamard transform on K/V + Q8_KV — PR 5.1b (DROPPED)

`ik_llama.cpp` ships a Hadamard rotation applied to K and V before
INT8 quantization to suppress outliers. TurboQuant (which attacked the
same problem via Walsh–Hadamard + Lloyd–Max) was removed — outside the
workload envelope. PR 5.1b is dropped with it.

### Self-speculative + ngram decoding

Reference for the speculative decoding initiative (separate from the
modular-layer refactor, no Phase gate). The ngram prompt-lookup draft
is already partially implemented in Qwenium; `ik_llama.cpp`'s
self-speculative variant (draft from the same model at reduced layers)
is worth reading before resuming that work.

### Not relevant to Qwenium

- **FlashMLA** — MLA (multi-head latent attention) is a DeepSeek-V2/V3
  variant; Qwen and Gemma do not use it.
- **Tensor overrides for hybrid CPU/GPU storage** — targets NUMA /
  discrete GPU setups. Unified memory on Apple Silicon makes this
  irrelevant.
- **Trellis quants (IQ*_KT)** — novel weight-quant family. Watch quality
  numbers upstream; do not port speculatively.

---

## ggml `ggml_flash_attn_ext` — PR 4.4

Native ggml op for fused attention. Parameters cover: scale, logit
soft-cap, sliding-window size, QK-norm. Evaluate switching
`src/layers/attention.cpp` from raw `ggml_mul_mat` + `ggml_soft_max`
to this op. Win is asymmetric: meaningful for prefill (avoids
materializing the full N×N score matrix), negligible for single-token
decode. Switching would also delete the custom mask-building logic in
`attention.cpp`.
