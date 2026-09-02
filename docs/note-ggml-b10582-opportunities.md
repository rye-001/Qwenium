# What b10582 offers that we don't use — opportunity ledger (2026-09-02)

The b8390 → b10582 bump was analysed for **cost and performance**
([`note-ggml-upgrade-pretest.md`](note-ggml-upgrade-pretest.md),
[`note-ggml-upgrade-b10582.md`](note-ggml-upgrade-b10582.md)) and never for
**opportunity**. In the port note new upstream ops appear only as an
enum-renumbering hazard ("Upstream appended four ops … so the two qinf ops now
follow `DSV4_HC_POST`"), never as capabilities. This note is the missing half.

Method: true delta against the local `b8390` tag in
`build-release/_deps/ggml-src` (the tag is present, so nothing here is inferred
from release notes). Op enum, public `ggml.h` API, `ggml-backend.h` API, and the
Metal kernel set — the last **suffix-normalized**, because b10582 templated many
kernels (`kernel_swiglu_f32` → `kernel_swiglu`) and a naive name diff reports
18 "new" kernels that are renames.

**Scope note:** candidates are NOT filtered to the ≤10-slot / ≤10K envelope
(user instruction, 2026-09-02). Several below are interesting *because* they
relax it.

## Raw delta

- **Ops: 97 → 102.** `COL2IM_1D`, `LIGHTNING_INDEXER`, `DSV4_HC_{PRE,COMB,POST}`.
  The port note recorded four of these and **missed `COL2IM_1D`**.
- **`ggml.h`: 12 new public functions, 0 removed.** `ggml_build_forward_order`,
  `ggml_col2im_1d`, `ggml_dsv4_hc_{comb,post,pre}`, `ggml_is_contiguous_to_{1,2,3}`,
  `ggml_lightning_indexer`, `ggml_mul_mat_set_hint`, `ggml_rope_set_offset`,
  `ggml_type_sizef`.
- **`ggml-backend.h`: 7 new.** `ggml_backend_tensor_{get,set}_2d{,_async}`,
  `ggml_backend_meta_{device,split_axis_name,split_state}`.
- **Metal: 16 genuinely new kernels** after normalization (from 99 → 117 raw).

## Candidates, ranked

### A. Quantized KV cache **with** flash attention — the one that relaxes the envelope
`kernel_flash_attn_ext_kv_f16` is genuinely new. It is a pre-pass that
materializes an F16 copy of K and V into scratch (`use_kv_f16`,
`ggml_metal_op_flash_attn_ext_kv_f16_k_size`) so flash attention can run over a
KV cache that is **not** F16 — including a quantized one.

Separately, `SET_ROWS` already accepted quantized destinations at b8390
(`Q8_0/Q4_0/Q4_1/Q5_0/Q5_1`) — **that half is not new and has simply never been
exploited.** b10582 adds one case: `src0 F16 → dst F16`.

Why it matters here: memory is the binding constraint on this machine, KV bytes
scale as **ctx × slots**, and §12 already flags that the TurboQuant/SnapKV
deletion rationale rests on a 4K-era measurement. A Q8_0 or Q4_0 KV cache is
roughly 1/4 to 1/8 of F32, and it is the direct lever on both axes the envelope
is written in. It would be opt-in beside `--kv-f16` for the same §11 reason (KV
dtype is part of "config"; not a calibrated receipts path).
**Unknown to check first:** whether the F16 pre-pass cost eats the bandwidth win
at decode's single query row. Measure before building.

### B. `ggml_backend_tensor_get_2d` — strided readback for per-slot logits
New. `note-batch-scaling-cross-family.md` measures a real gap between "plain"
and "server-realistic" batching (per-lane logits readback + argmax): 2.96× vs
2.48× at B=8 on Gemma 4. Per-slot logit extraction is exactly a strided 2D read.
Small, well-scoped, and tied to an already-measured cost — the cheapest item here.

### C. `ggml_rope_set_offset` — partial-rotation offset
New; marks a leading unrotated span (`n_embd=10, n_dims=4, offset=2 → [00xxxx0000]`).
We already do partial RoPE via `build_rope_pruned`. **Not investigated** —
whether this subsumes our pruned-rope path, and whether it interacts with the
M-RoPE bookkeeping in §12, is unknown. Header says "vision RoPE is not supported",
which likely rules out the M-RoPE angle.

### D. `ggml_is_contiguous_to_{1,2,3}` — fewer `ggml_cont` calls
New predicates. §3 says `ggml_cont` pins scratch memory and should only be
called when an op genuinely requires it. These allow a finer test than the
current all-or-nothing one. Speculative; needs a survey of our `ggml_cont` sites
to see if any are provably unnecessary under the weaker predicate.

### E. New quant-type matvec kernels — `mul_mv_{q1_0,q2_0,tq2_0}`
New. Ternary/2-bit weight types. Not an engine change — a model-file question
(we have no such GGUFs). Listed only because memory is the constraint and these
are the smallest weight formats Metal now has a matvec for.

## Killed before costing anything

- **`ggml_mul_mat_set_hint`** — looked like a kernel-selection lever on the
  `mul_mv_ext`/`mul_mm` thresholds that drive our whole batch-scaling story.
  It is not: `enum ggml_op_hint` has exactly one value,
  `GGML_HINT_SRC0_IS_HADAMARD`, and the only backend consuming it is SYCL.
  Dead end on Metal. (*Read the dispatch source before the stopwatch*, §10.)
- **Metal 4 tensor API** — present in this ggml, hardware-gated off here:
  `ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices`.
  Unreachable on M1 Pro.
- **`LIGHTNING_INDEXER`, `DSV4_HC_*`** — DeepSeek-V4 / sparse-attention-indexer
  ops for architectures we do not host and have no reason to (no model zoo).
- **`COL2IM_1D`, `conv_2d_dw`, `conv_3d`, `fwht`, `roll`, `snake`,
  `silu_back`** — conv/audio/training ops. Nothing in our graphs.
- **`ggml_backend_meta_*`** — multi-device model splitting. Single-GPU here.

## A finding that is NOT a b10582 opportunity, but is a real defect

`src/layers/moe.cpp` builds the expert and shared-expert activations
**unfused** — `ggml_mul(ctx, ggml_silu(ctx, gate), up)` at lines 120 and 166 —
while `src/layers/ffn.cpp` uses the fused `ggml_swiglu_split` /
`ggml_geglu_split`. Metal has had `GGML_OP_GLU` and `kernel_swiglu_f32` since
b8390, so **this predates the bump and is not caused by it.** On Qwen 3.6 that is
~2 extra nodes per MoE layer plus an intermediate write+read, on ~40 layers, on
every decode step. Whether it is worth fusing is an Amdahl question (§10) that
this note does not answer — and the fused form must clear the standard
token-stable gate, since it is a different Metal kernel, not a rearrangement.

## What was not examined
CPU backend, CUDA/other backends, `ggml_build_forward_order` (graph-build
ordering — possible interaction with `decode_graph_cache`, unread), quantization
type internals, and everything in `llama.cpp` above ggml.
