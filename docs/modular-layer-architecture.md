# Modular Layer Architecture

A blueprint for refactoring the inference engine from a model-specific design
to a composable, layer-type-driven architecture.

## Motivation

The engine was built around a single model family: Qwen3 (pure transformer).
Every optimization — KV cache management — assumes that every layer is an
attention layer backed by a key-value cache.

That assumption is now obsolete. Qwen 3.6 introduces a hybrid architecture
where 30 of 40 layers are Gated DeltaNet (linear recurrence) and only 10 are
traditional attention. Jamba (AI21) mixes Mamba SSM layers with attention.
The industry is converging on hybrid models that combine multiple layer types
with Mixture-of-Experts feed-forward networks. New architectures will keep
arriving.

We do not want to fork the engine for each new model. We want to compose
existing, tested layer modules into any model configuration — and when a new
layer type appears, add only that module without touching the rest.

## Goal

Refactor the engine so that:

- Each layer type is an independent module with its own compute logic,
  Metal kernel, state management, quantization strategy, and unit tests.
- A model is a declarative recipe that composes layer modules.
- Architecture-agnostic systems (sampling, speculative decoding, weight
  quantization, Metal dispatch) sit above or below the layer modules and
  are never duplicated.
- Adding a new model requires zero changes to existing layer modules.
- Adding a new layer type requires zero changes to existing models.

## Module Stack

```
┌─────────────────────────────────────────────┐
│           Sampling (Grammar/Trie)            │  arch-agnostic
├─────────────────────────────────────────────┤
│         Speculative Decoding                 │  mostly arch-agnostic *
├──────────┬──────────┬───────────┬───────────┤
│Attention │   SSM    │ DeltaNet  │    MoE    │  layer-type
│(KV cache)│ (Mamba)  │ (delta    │ (expert   │  modules
│          │          │  rule)    │  dispatch) │  (each owns its .metal kernel)
├──────────┴──────────┴───────────┴───────────┤
│        Weight Quantization (K-Quants)        │  arch-agnostic
├─────────────────────────────────────────────┤
│   Metal Infrastructure (dispatch, buffers)   │  arch-agnostic
└─────────────────────────────────────────────┘
```

The top and bottom of the stack are shared by all models. The middle row
is architecture-specific: each layer-type module owns both its graph-building
code and its Metal kernel (`.metal` file). The "Metal Infrastructure" layer
at the bottom is the dispatch runtime, buffer allocator, and command-queue
glue — not the kernels themselves.

**\* Speculative decoding caveat.** Standard (draft-model) speculation is
arch-agnostic. Any scheme that reuses the target model's own layers as a
draft (see `future-work.md`) is not — recurrent state advances per accepted
token and makes layer-skipping non-trivial.

## Layer-Type Modules

### Attention

Traditional multi-head / grouped-query attention with KV cache.

- **Compute:** Q/K/V projection, RoPE, scaled dot-product attention, output projection.
- **State:** KV cache (grows linearly with sequence length).
- **Quantization:** K-Quant on weights.
- **Metal kernel:** Fused QKV projection, flash-attention-style tiled matmul.
- **Models that use it:** Qwen3 (all layers), Qwen 3.6 (every 4th layer), Jamba (interleaved).

### SSM (Mamba)

Selective State Space Model. Structured recurrence with input-dependent gating.

- **Compute:** Input projection, selective scan (A, B, C, D matrices), output projection.
- **State:** Fixed-size hidden state per layer (does not grow with sequence length).
- **Quantization:** State quantization (research target — the hidden state is compact
  but precision-sensitive).
- **Metal kernel:** Selective scan with causal convolution. Two paths:
  chunked parallel scan (prefill) and sequential step (decode).
- **Models that use it:** Jamba, Zamba, Mamba-family, future hybrids.

### DeltaNet (Gated Delta Rule)

Linear attention via delta-rule recurrence. Updates a key-value state matrix
by computing the error between the current value and the state's prediction,
then applying a learned correction.

- **Compute:** QKV + gate projection, 1D causal convolution, L2-normalized Q/K,
  delta-rule state update (S = S * decay + k * beta * (v - S*k)), gated output
  with RMS normalization.
- **State:** Fixed-size matrix per head (state_size x state_size x n_heads).
  For Qwen 3.6: 128 x 128 x 32 = ~524K floats per layer, ~60 MB for 30 layers at fp32.
- **Quantization:** State quantization (open research — the state receives low-rank
  updates via the delta rule, suggesting compressibility).
- **Metal kernel:** Three paths following llama.cpp's reference:
  - Autoregressive (1 token): direct recurrent state update.
  - Chunked (prefill): chunk into blocks of 64, intra-chunk triangular solve,
    inter-chunk recurrence.
  - Fused: single `gated_delta_net` op combining both paths.
- **Models that use it:** Qwen 3.6 (30 of 40 layers).

### MoE Dispatch (Mixture of Experts)

Routes each token to a subset of expert FFN networks via a learned gating function.

- **Compute:** Gating network (top-k softmax over expert scores), per-expert
  SwiGLU FFN (gate + up + down projections), weighted sum of expert outputs.
  Optional shared expert with sigmoid gating added to the routed output.
- **State:** Stateless (no cache). But memory footprint is large: all experts must
  be resident (Qwen 3.6: 256 experts x 3 matrices x 512 dim = bulk of the 16 GB model).
- **Quantization:** K-Quant on expert weights. Key challenge: scatter-gather access
  pattern across 256 experts is adversarial for memory bandwidth. Expert weight
  layout optimization (grouping frequently co-activated experts) is a potential
  technique.
- **Metal kernel:** Batched small-matrix multiply across active experts. The naive
  approach (8 sequential expert FFNs) leaves the GPU underutilized. The target is
  a fused kernel that dispatches all active experts in one pass.
- **Models that use it:** Qwen 3.6 (all layers), DeepSeek-V3, Mixtral, Jamba.

### Dense FFN

Standard feed-forward network (SwiGLU). The non-MoE baseline.

- **Compute:** Gate + up projection, SiLU activation, element-wise multiply, down projection.
- **State:** Stateless.
- **Quantization:** K-Quant on weights.
- **Metal kernel:** Existing fused SwiGLU kernel.
- **Models that use it:** Qwen3 (all layers), attention-only transformers.

## Not a Layer-Type Module: the Vision Tower (C3 — provisional)

> **Status:** Provisional answer, validated against exactly **one** recipe
> (Gemma 3 + SigLIP). Per CLAUDE.md's cross-family rule, this is "presumed
> Gemma-shaped until a second multimodal recipe proves otherwise." See
> [`plan-gemma-vision-impl.md`](plan-gemma-vision-impl.md) → "C3 deliverable".

When multimodal support landed (Gemma 3 vision, Phase 2), it forced the
question: **is the vision tower a sixth layer-type module, alongside the five
above — or is it something else?**

**Working answer: it is a pre-decoder pipeline, not a layer-type module.**
The five modules above share one defining property — they participate in the
**decoder's residual stream**, building into the shared `ggml_cgraph` via the
norm-then-branch-then-add recipe pattern (see Key Design Principle). The
vision tower does none of that:

- **Different signal flow.** It runs *once, before* decode, transforming a
  bitmap into image embeddings. It does not sit inside the per-layer residual
  loop; it terminates at the **embedding boundary**, where its output
  substitutes for token embeddings at image-placeholder positions.
- **Its own graph, context, and scheduler.** `src/vision/` owns a separate
  `ggml_cgraph`, a separate `ggml_context`, and a separate
  `ggml_backend_sched_t`. It *shares* the engine's Metal `ggml_backend_t`
  (one device, two schedulers) but nothing else. It has zero dependency on
  `Model` / `ForwardPassBase` — the separation is directory-enforced.
- **Internal sub-layers stay internal.** The vision tower has its own
  attention / MLP / LayerNorm blocks, but these are *not promoted* to
  `src/layers/`. They are SigLIP-internal, built inside `src/vision/`, and
  governed by a different contract (LayerNorm not RMSNorm, learned absolute
  position embeddings not RoPE, bidirectional not causal, GELU not SwiGLU).
  Promoting them would be a false abstraction: they share a *name* with the
  decoder's attention/FFN, not a *contract*.

This mirrors the separation llama.cpp consciously chose (`mtmd` outside
`libllama`), expressed in our directory conventions.

**Concretely grounded (Phase 2.4).** The standalone encoder
(`src/vision/vision_encoder.cpp`) builds the entire SigLIP pipeline —
conv2d patch embed → 27 transformer layers → post-LN → 4×4 average pool →
RMS soft-emb-norm → linear projection — as one self-contained graph, and
its output matches the llama.cpp reference to cosine ≈ 0.99997 / relative
L2 ≈ 0.3% on the gray-input fixture (low-bit BF16 + reduction-order FP
divergence, the same class documented for `feed_tokens`; the shipping gate
is coarse cosine/rel-L2, not bitwise). That the encoder ships as its own
graph terminating at the embedding boundary — never touching the decoder's
residual stream — is the evidence for the "pipeline, not module" answer.

**What would promote this from provisional to validated.** A *second*
vision-bearing recipe in a different family, with a structurally different
vision encoder (e.g. Gemma 3n's MobileNetV5, or a Qwen-VL ViT variant),
integrating through the same pre-decoder-pipeline boundary **without
requiring logic edits to `src/vision/`'s contract** — only new recipe code.
Until then, the answer is grounded on Gemma 3 alone and is a working
hypothesis, not a cross-family conclusion. If a second recipe instead
forces edits to the pipeline boundary itself, that is an interface defect
to fix, per the Architecture Pressure Test — not a second module to add.

## Refactored Source Structure

```
src/
  layers/
    layer.h                  # common LayerInterface
    attention.h / .cpp       # extracted from current forward-pass
    ssm.h / .cpp             # new — Mamba selective scan
    deltanet.h / .cpp        # new — gated delta rule recurrence
    moe.h / .cpp             # new — expert routing + dispatch
    ffn.h / .cpp             # extracted from current forward-pass
  metal/
    attention.metal          # flash-attention-style tiled matmul
    ssm.metal                # selective scan kernel
    deltanet.metal           # fused gated delta net kernel
    moe.metal                # batched expert dispatch
    ffn.metal                # existing SwiGLU kernel
  state/
    kv_cache.h / .cpp        # KV cache (used by attention only)
    ssm_state.h / .cpp       # SSM recurrent state
    deltanet_state.h / .cpp  # DeltaNet recurrent state
    state_quant.h / .cpp     # recurrent state quantization (future)
  models/
    model.h                  # common ModelInterface — layer composition
    qwen3.cpp                # composes: attention + ffn
    qwen3_moe.cpp            # composes: attention + moe
    qwen36.cpp               # composes: deltanet + attention + moe
    jamba.cpp                 # composes: ssm + attention + moe
  sampling/
    grammar.h / .cpp         # already arch-agnostic
    token_trie.h / .cpp      # already arch-agnostic
    speculative.h            # already arch-agnostic
    sampling.h / .cpp        # already arch-agnostic
  quant/
    kquant.h / .cpp          # weight quantization (arch-agnostic)
  loader/
    gguf_loader.h / .cpp     # model loading (arch-agnostic, metadata-driven)
  cli/                       # unchanged
  server/                    # unchanged

tests/
  unit/
    test_attention.cpp
    test_ssm.cpp
    test_deltanet.cpp
    test_moe.cpp
    test_ffn.cpp
    test_kv_cache.cpp
    test_ssm_state.cpp
    test_deltanet_state.cpp
    test_model_qwen3.cpp
    test_model_qwen36.cpp
```

## Where Current Code Maps To

| Current location                   | Target module          | Action   |
|------------------------------------|------------------------|----------|
| `src/core/forward-pass.cpp`  | `src/layers/attention` + `src/layers/ffn` | Extract |
| `src/core/forward-pass-qwen35.cpp` | `src/layers/deltanet` + `src/layers/attention` + `src/layers/moe` | Extract |
| `src/kv-cache/compressed_kv_store`  | `src/state/kv_cache`   | Move     |
| `src/kv-cache/ssm-state-cache`      | `src/state/ssm_state` + `src/state/deltanet_state` | Split |
| `src/sampling/*`                    | `src/sampling/*`       | Keep     |
| `src/core/gguf-loader`        | `src/loader/gguf_loader` | Move   |
| `src/core/qwen3-model`        | `src/models/qwen3`    | Move     |
| `src/core/tokenizer`          | `src/loader/tokenizer` | Move    |

## Key Design Principle

A model file is a recipe, not an implementation. Example for Qwen 3.6:

```
residual = input
for each layer:
    normed = RMSNorm(residual)

    if layer_index % 4 == 3:
        hidden = AttentionLayer(normed)     // with KV cache, RoPE, GQA
    else:
        hidden = DeltaNetLayer(normed)      // with recurrent state, causal conv

    residual = residual + hidden            // skip connection

    normed = RMSNorm(residual)
    ffn_out = MoELayer(normed)              // 256 experts, top-8 + 1 shared
    residual = residual + ffn_out           // skip connection
```

All logic lives in the layer modules. The model file owns the residual stream
and the norm-then-branch-then-add pattern. This is intentional: the residual
stream is the model's backbone, not a layer's responsibility.

## Engineering Constraints

These are hard constraints imposed by our runtime (ggml + Metal on Apple Silicon)
that the modular design must respect. Ignoring any of them will cause either
correctness bugs or severe performance regression.

### Shared Computation Graph

ggml builds a monolithic `ggml_cgraph` before execution. All layers — regardless
of type — must append their operations to the same graph and allocate intermediate
tensors from the same `ggml_context`. Layer modules are **not** independent
executables; they are graph-building functions that receive a shared context and
return tensor handles.

The interface is therefore:

```cpp
// Every layer module builds into the caller's graph context.
// It receives the current tensor, appends ops to ctx, and returns
// the output tensor. It never allocates its own ggml_context.
ggml_tensor * AttentionLayer::build(ggml_context * ctx, ggml_tensor * input, int layer_idx);
ggml_tensor * DeltaNetLayer::build(ggml_context * ctx, ggml_tensor * input, int layer_idx);
ggml_tensor * MoELayer::build(ggml_context * ctx, ggml_tensor * input, int layer_idx);
```

This preserves ggml's scratch buffer pooling: the allocator sees the full graph
and reuses memory across layers. If we broke this — for example by giving each
layer its own context — scratch buffer efficiency collapses and VRAM fragments.

**Unit testing implication:** layer tests must create a `ggml_context`, call the
layer's `build()`, then execute the subgraph. They test graph construction and
numerical correctness, not standalone execution.

### Scratch Buffer Discipline

ggml's memory allocator (`ggml_gallocr`) plans buffer reuse across the entire
graph in one pass. The key invariant: **a tensor's memory can be reused as soon
as all downstream consumers have executed.** This works naturally as long as all
layers share one graph.

The risk with modular layers is premature materialization — if a layer module
forces `ggml_cont()` or `ggml_cpy()` on intermediate tensors (to satisfy its
internal layout assumptions), those copies pin memory that could otherwise be
reused. Layer modules must prefer in-place views (`ggml_view_*`, `ggml_reshape_*`,
`ggml_permute`) over copies wherever possible.

**Don't translate reference implementations line-by-line.** PyTorch / JAX
reference code allocates intermediate tensors for clarity; ggml's gallocr
pins each intermediate's memory for its full downstream lifetime. A direct
translation of the DeltaNet forward pass produced over 16 GB of scratch
for a 500 M model in llama.cpp's Qwen3-Next PR
([#16095](https://github.com/ggml-org/llama.cpp/pull/16095)); the fix was
composing into fewer, wider ggml ops with views rather than materializing
each step. Treat the reference as a correctness oracle, not a template to
port op-for-op.

**Reviewer checklist for a new layer module:**

1. What is the peak scratch footprint per token, measured with
   `ggml_gallocr_get_buffer_size`? Record it in the module's test output.
2. Is any intermediate tensor bigger than `max(input_size, output_size)`?
   If yes, justify — it should be rare.
3. Count `ggml_cont` / `ggml_cpy` / `ggml_dup` calls in the module. Each
   one must be defensible ("Metal kernel X requires contiguous Y");
   unjustified copies are bugs, not stylistic choices.

### State Management Lifecycles

KV cache and recurrent state have fundamentally different update semantics.
The two implementations share almost no code — the base interface must stay
minimal or it will invent methods that are no-ops on one side.

| | KV Cache (Attention) | Recurrent State (DeltaNet/SSM) |
|---|---|---|
| **Growth** | Appends K/V for each new token. Grows linearly with sequence. | Fixed-size matrix. Does not grow. |
| **Update** | Write-once append at position `head`. Pointer advances. | Full overwrite via exponential moving average: S = S*g + delta. |
| **Eviction** | Not applicable (outside workload envelope; SnapKV was removed). | No eviction — state is always fully overwritten. |
| **Quantization** | K-Quant on weights. KV-precision compression (TurboQuant) was removed — outside workload envelope. | Open research. State has structure (low-rank delta updates) but no proven compression scheme yet. |
| **Batch reset** | Clear cache range for new sequence. | Zero the state matrix. |

The common interface:

```cpp
class LayerState {
    virtual void     reset_sequence(int seq_id) = 0;  // clear for new sequence
    virtual size_t   memory_bytes() const = 0;        // for budget tracking
    virtual ~LayerState() = default;
};

class KVCache : public LayerState {
    void truncate_to_position(int pos);   // O(1) head-pointer adjustment
    // ...
};

class RecurrentState : public LayerState {
    CheckpointId checkpoint();            // snapshot state (full-size copy)
    void         restore(CheckpointId);   // overwrite state from snapshot
    void         release(CheckpointId);   // free snapshot memory
    // intentionally NO truncate_to_position — partial rewind is not a legal op
};
```

**No shared `advance_step()`.** State mutation happens inside each layer's
`build()` call: attention's `build()` writes K/V at the current head and
increments it; DeltaNet's `build()` emits the `S = S*g + delta` update as
part of its subgraph. There is no post-step bookkeeping that both kinds
need to do, so forcing a common method produces a no-op on one side — a
smell that would compound as new state-bearing layer types are added.

**Rewind contract.** Attention supports cheap positional truncation via
head-pointer adjustment. Recurrent state supports checkpoint/restore at
caller-chosen positions. **Partial positional rewind of recurrent state is
not supported at any layer of the stack** — once the state has absorbed a
delta update for token `t`, there is no algebraic inverse. Callers that
need rollback (grammar backtracking, speculative decoding rejection) must
explicitly checkpoint before the speculative work and restore on rejection.
Each checkpoint is a full-state copy (~60 MB for DeltaNet on Qwen 3.6);
budget accordingly.

This is the specific failure class that llama.cpp still leaks — their
`seq_rm` tries to be position-based for both cache kinds and returns
`false` on recurrent state, which in turn breaks prompt-cache invalidation
and checkpoint/restore on hybrid models
([#21831](https://github.com/ggml-org/llama.cpp/issues/21831),
[#19794](https://github.com/ggml-org/llama.cpp/issues/19794)). We avoid
it by making the asymmetry part of the public contract from day one,
rather than hiding it behind a shared method that silently no-ops.

**Downstream implications:**
- **Grammar rollback:** checkpoint before emitting a candidate token, release
  on acceptance, restore on rejection. Single-step; memory cost is one
  checkpoint's worth at a time.
- **Speculative decoding:** checkpoint before drafting a batch of N tokens,
  restore + re-run the accepted prefix if verify rejects. One checkpoint
  per active draft batch.
Per-kind methods (`KVCache::evict`, `RecurrentState::checkpoint`, etc.) live
on the concrete class and are called only by code that knows which kind it
is dealing with.

### Prefill vs Decode: Graph Topology

Attention has two graph shapes (batched tiled matmul for prefill, single-token
dot-product against the cache for decode). DeltaNet has three (chunked parallel
scan for prefill, autoregressive step for decode, and a fused path that covers
both). These are not interchangeable at the graph level — the allocator plans
memory reuse for the specific ops in the graph, so a single `build()` that
emits both shapes via runtime branching would defeat `ggml_gallocr`.

The layer interface therefore takes an explicit phase:

```cpp
enum class Phase { Prefill, Decode };

ggml_tensor * AttentionLayer::build(ggml_context * ctx,
                                    ggml_tensor * input,
                                    int           layer_idx,
                                    Phase         phase);
```

The caller (the model recipe) already knows which phase the current forward
pass is running in and passes it through. Each module internally dispatches
to its phase-specific subgraph builder. Modules that only have one shape
(Dense FFN, MoE) ignore the argument.

The graph is rebuilt per batch anyway (ggml is not a retained-mode engine),
so there is no overhead in emitting a different topology per phase.

### Model Dispatch: GGUF Architecture String → Recipe

"Adding a new model requires zero changes to existing modules" needs a
concrete dispatch mechanism. The GGUF loader reads `general.architecture`
from the file header and looks up the matching recipe in a registry:

```cpp
// Each model file self-registers at static-init time.
REGISTER_MODEL("qwen3",     Model);
REGISTER_MODEL("qwen3moe",  Qwen3MoEModel);
REGISTER_MODEL("qwen35moe", Qwen36Model);
REGISTER_MODEL("jamba",     JambaModel);
```

Adding a new model means one new file under `src/models/` and one
`REGISTER_MODEL` line. No existing module is touched. Unknown architecture
strings fail loudly at load time, not with a silent fallback.

### Weight Binding: GGUF Tensors → Module Slots

`REGISTER_MODEL` solves *graph* dispatch. Weight binding is a separate
surface: which GGUF tensor ends up in which module's weight slot, and
who parses hparams out of the file header. This is where llama.cpp's
multi-file sprawl originates — we address it explicitly rather than
letting it leak into the model recipe.

**Ownership split:**

- The **loader** (`src/loader/gguf_loader.cpp`) parses GGUF key-value
  metadata into a typed `ModelHparams` struct and enumerates tensor
  descriptors (name, shape, dtype, data pointer). It does not know about
  layer types.
- Each **layer module** exposes a static `weight_schema(layer_idx, hparams)`
  that returns a list of `WeightSlot { slot_name, expected_shape, expected_dtype }`
  describing what tensors it needs for a given layer index.
- The **model recipe** owns the mapping from GGUF tensor names to module
  slots via a `TensorNameMap` — e.g. `blk.{i}.attn_q.weight` →
  `(AttentionModule, layer=i, slot="q_proj")`. The map is declarative
  and lives in the model recipe file, not in a shared switch.

**Binding is explicit and fails loud:**

1. The loader walks every tensor in the GGUF file and asks the model's
   `TensorNameMap` to resolve it.
2. A tensor name that resolves to no slot fails at load time with the
   unrecognized name in the error message.
3. A slot that no tensor resolves to (missing weight) fails at load time
   with the expected name, shape, and dtype.
4. A resolved tensor whose shape or dtype mismatches the slot's schema
   fails at load time with both values in the error message.

No silent fallbacks, no best-effort loading, no half-initialized models.
A corrupt or mismatched GGUF is rejected before the first forward pass.

**Hparams validation** is the model recipe's responsibility. It reads
the fields it needs from `ModelHparams` in its constructor and raises
on missing or out-of-range values. Each module receives only the
hparams it needs as explicit constructor arguments — no global
`hparams` blob passed through the stack.

This closes the loop on Pain #1 from the llama.cpp audit: adding a new
model is *one* recipe file plus *one* `REGISTER_MODEL` line, because
the tensor-name convention, the hparam extraction, and the weight
binding all live inside that single file.

### MoE Kernel: Fused Gating + Expert Dispatch

The naive ggml approach to MoE — chaining `ggml_mul_mat` per expert — launches
8 separate Metal kernel dispatches per token per layer. On Apple Silicon, kernel
launch overhead (~5-10 us each) dominates when the per-expert matmul is tiny
(2048 x 512, ~2M FLOPs — finishes in ~1 us on M-series).

The MoE module must therefore provide a **single fused Metal kernel** that:

1. Evaluates the gating network (top-k softmax over 256 experts).
2. Gathers the 8 selected expert weight matrices.
3. Computes all 8 SwiGLU FFNs (gate + up + SiLU + mul + down) in one dispatch.
4. Blends outputs by gating weights.
5. Adds the shared expert contribution.

On the ggml graph side, this is represented as a single custom op
(`ggml_custom_op`) that encapsulates the entire MoE forward pass. The layer
module builds this single op node rather than a subgraph of individual matmuls.

This is the highest-risk engineering item in the refactor. Without the fused
kernel, MoE layers on Metal will be 5-10x slower than they should be.

**Fallback plan.** If the fused custom op lands outside the acceptable
performance window (target: within 1.5x of llama.cpp's MoE path on the same
model), the refactor does not block. The fallback is a batched-matmul
approach using `ggml_mul_mat_id` with a routing table, which keeps dispatch
count at O(1) per layer even if it leaves some perf on the floor. The fused
kernel then becomes an optimization target in a later phase rather than a
correctness gate for Phase 3.

### MoE Expert Residency

Expert residency is part of the MoE module's contract — not a runtime
afterthought. On Apple Silicon, unified memory removes the PCIe-transfer
cost that dominates MoE on CUDA, but it does **not** remove the
page-management cost: expert weights can be paged out under memory pressure,
and re-paging during a decode step stalls the pipeline. llama.cpp currently
defers expert residency to the OS, and on M4 Pro this has caused kernel
panics from swap thrash on MoE workloads
([#19825](https://github.com/ggml-org/llama.cpp/issues/19825),
[#20757](https://github.com/ggml-org/llama.cpp/issues/20757)). We will not
inherit that failure mode.

Routing on trained MoE models is skewed: published Mixtral-style analyses
and the above issues both show roughly 15–20% of experts absorb the
majority of traffic on typical workloads. That skew is exploitable — pin
the hot experts, let the cold tail page naturally, measure cold-path
dispatches as a signal.

**Residency primitives owned by the MoE module:**

1. **Pin top-K experts** by observed activation frequency using Metal
   resource residency APIs (`MTLResidencySet` on supported OS versions,
   `setPurgeableState:NonVolatile` as fallback). The frequency set is
   initialized from an imatrix pass and updated from a rolling window of
   recent routing decisions.
2. **Cold-path dispatch counter.** Every routed expert that is not in the
   resident set increments a metric. A sustained spike indicates the
   routing distribution has shifted (conversation moved to a new domain);
   that is the signal to rebuild the residency set.
3. **Hardware-adaptive budget.** On 32 GB systems, pin roughly 40 of 256
   experts (~2.5 GB at Q3_K); on 64 GB+, pin all experts and skip the
   residency logic entirely. The budget is configured at load time from
   measured free memory, not hardcoded.

Residency is orthogonal to the fused dispatch kernel: the kernel does
per-token compute, residency controls which expert weights the kernel can
touch without a page fault. Both are required; neither substitutes for
the other. The residency state lives under `src/state/moe_residency.h`
(it is state, not a weight), even though it is not a `LayerState` in the
per-sequence sense — it is per-model runtime metadata.

**Metric exposure.** The cold-path dispatch counter is a first-class
telemetry signal from day one. It has two downstream uses: residency-set
rebuild triggering (above), and domain-shift detection for future
conversation-analysis features. That dual use is why it is exposed on
the MoE module's public surface, not hidden as an internal debug counter.

## Testing Strategy

Each layer module ships with unit tests that build a minimal ggml graph,
execute it on Metal, and compare the output against a reference. The
reference hierarchy (strongest oracle first):

1. **llama.cpp** on the same GGUF file, same prompt, same sampler seed —
   compared at the logit level (top-k overlap, then top-1 divergence token
   index). Primary oracle for full-model integration tests.
2. **PyTorch reference** (Hugging Face `transformers`) for layer-level
   known-answer tests with synthetic weights — where llama.cpp doesn't
   expose a single-layer entry point. Used for DeltaNet and SSM.
3. **Bit-for-bit self-comparison** against the pre-refactor build, for
   Phase 1 extraction gates on Qwen3.

**Tolerances.** Float math on Metal is not bit-exact against CPU PyTorch;
attention tests tolerate up to 5e-3 relative error on individual logits
and require top-8 logit rank preservation on a canary prompt set. KV-cache
tests run at fp16 and tolerate 1e-2. Recurrent-layer tests step the state
for 256 tokens and compare the final state plus every 16th output — small
per-step errors compound, so a single-step check is not sufficient.

**Canary prompts.** A fixed set (in `tests/fixtures/canary_prompts.txt`)
is used for every gate. Regressions show up as either a top-1 token
divergence within the first 128 generated tokens or a perplexity shift on
a held-out text above the phase's threshold.

## Phasing Strategy

The refactor follows a strict discipline: **one kind of change at a time.**
Extracting logic, changing interfaces, and optimizing kernels must never
happen in the same step. Each phase ends with a passing test suite that
proves bit-for-bit equivalence with the previous phase.

### Phase 1 — Mechanical Extraction (zero behavior change)

Move existing code into the new module structure without modifying a single
line of logic. `AttentionLayer::build()` is a cut-paste of the attention
block from `forward-pass.cpp`. `MoELayer::build()` is a cut-paste of the
MoE block from `forward-pass-qwen35.cpp`. The model file calls these
functions in the same order, with the same residual wiring.

**Gate:** bit-for-bit identical output on the existing Qwen3 integration tests,
**plus** CLI and server integration tests pass unchanged. If a single logit
changes, the extraction was wrong. No exceptions.

What this phase produces:
- `src/layers/attention.cpp` — extracted, not rewritten
- `src/layers/ffn.cpp` — extracted, not rewritten
- `src/models/qwen3.cpp` — calls the extracted modules
- All existing tests pass without modification (unit, integration, CLI, server)

What this phase does NOT touch:
- Metal kernels (still called the same way, from the same code paths)
- State management (KV cache stays exactly where it is)
- Public CLI flags, server endpoints, or GGUF loader behavior
- DeltaNet, SSM, MoE (not yet — Qwen3 doesn't use them)

**Scope sizing.** Phase 1 moves most of `src/core/` under the new tree
but changes no logic. Done on a branch, behind no feature flag. Estimated
~1-2 weeks of work; revertable with a single `git revert` if the integration
gate fails.

### Phase 2 — Interface Stabilization

With the extraction proven correct, now formalize the `LayerState` interface
and migrate `KVCache` behind it. Introduce the
`build(ctx, input, layer_idx, phase)` signature as the formal contract.
Add unit tests for each extracted module in isolation (small ggml graph,
synthetic weights, known-answer checks against the oracles from the
Testing Strategy section).

**Gate:** all Phase 1 tests still pass. New unit tests pass. No output change.

### Phase 3 — New Layer Types

With the interfaces stable, implement `DeltaNetLayer` and `MoELayer` as new
modules conforming to the same contracts. Use llama.cpp's `delta-net-base.cpp`
and `qwen35moe.cpp` as reference implementations. Add `RecurrentState` as a
`LayerState` implementation.

Compose them into `src/models/qwen36.cpp`. Validate against llama.cpp's output
for the same model and prompt (the Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf file we
already have).

**Gate:** Qwen 3.6 produces coherent output. Cross-check a fixed prompt against
llama.cpp for logit-level agreement.

### Phase 4 — Metal Kernel Surgery

Only now, with correct behavior proven, begin decoupling and optimizing Metal
kernels per module. This is where the fused MoE dispatch kernel, the optimized
DeltaNet recurrence kernel, and any attention kernel improvements happen.

Each kernel change is validated against the Phase 3 output: same input, same
output, faster execution. Profile before and after with Metal GPU capture.

**Gate:** identical output, measured speedup, no regression on other layer types.

### Phase 5 — State Optimizations

Apply quantization and compression techniques to the state modules. Research
and prototype recurrent state quantization for DeltaNet. Experiment with MoE
expert weight layout optimization. (TurboQuant was removed — outside workload
envelope.)

**Gate:** output quality within acceptable perplexity bounds (measured, not assumed).

### The Rule

At no point does a phase combine extraction with optimization, or interface
change with kernel rewrite. If you are tempted to "clean up" a Metal kernel
while extracting logic into a module, stop. Finish the extraction. Pass the
tests. Commit. Then start the kernel work in the next phase.

## Techniques Preserved and Extended

| Technique             | Current scope        | After refactor                      |
|-----------------------|----------------------|-------------------------------------|
| K-Quants              | All weights          | All weights (unchanged)             |
| Grammar/Trie          | Output sampling      | Output sampling (unchanged)         |
| Speculative Decoding  | Full model           | Full model (unchanged)              |
| State Quantization    | Not yet              | SSM + DeltaNet recurrent states     |
| MoE Expert Caching    | Not yet              | MoE dispatch module                 |

## Collaboration Invariants

The invariants below serve two constituencies equally: human engineers
reasoning under time pressure, and LLM-based agents reasoning under
context limits. They are phrased as good-engineering rules that happen
to also be legible to agents — not the other way around. If any of
them ever conflicts with good engineering, good engineering wins and
the invariant is revised.

**Locality of reasoning.** A change to one layer type should not require
reading more than two other files to verify correctness. This is the
reason the architecture is layer-type-modular in the first place; the
invariant exists to protect that property from erosion. Reviewers
reject changes that violate it, whether or not the change compiles.

**File-size prompt (soft).** When a module file exceeds ~1500 lines,
treat it as a prompt to check whether the module is holding more than
one responsibility. The right response is often to split; sometimes it
is to leave the file long because the responsibility is genuinely
cohesive. Do not split purely to satisfy a line count.

**Module header contract.** Every module's primary source file opens
with a doc block of ~15-25 lines stating: (1) the module's single
responsibility, (2) its public surface, (3) any state it owns,
(4) invariants it maintains, (5) reference implementation it was
validated against (with URL where applicable), (6) path to its unit
test file. The block is not prose; it is a scannable contract. Update
it when the module's responsibility changes.

**Test co-location.** `src/<path>/<module>.cpp` has its unit test at
`tests/unit/test_<module>.cpp`. Always. No exceptions, no nesting
variants. A reviewer or agent looking for a module's test should
never have to search.

**Naming discipline.** Layer types end in `Layer` (`AttentionLayer`,
`DeltaNetLayer`). State types end in `State` (`KVCacheState`,
`RecurrentState`). Metal kernels are named `<module>_<op>.metal`
(`moe_dispatch.metal`, `attention_flash.metal`). Namespaced under
`qinf::layers`, `qinf::state`, `qinf::models`, `qinf::sampling`.
Consistency matters more than the specific scheme — once three
variants of the same concept exist, grep-uniqueness dies and both
humans and agents rank false positives.

**No dumping grounds.** `src/` contains only directories that name a
concept from this spec (`layers/`, `state/`, `graph_inputs/`, `models/`,
`sampling/`, `loader/`, `cli/`, `server/`, `image/`, `engine/`,
`session/`, `vision/`). Do not add
`util/`, `common/`, `misc/`, `helpers/`. If a piece of code doesn't
fit in one of the named directories, either (a) it belongs in a new
concept-named directory whose purpose is stated in this spec, or
(b) it belongs inside an existing directory. New directories are
allowed when they represent a new architectural concept; they are
not allowed as catch-alls.

**Fail-loud error contract.** Errors at module boundaries state the
slot or parameter name, the expected value, and the actual value, in
that order. Example: `weight_binding: slot "q_proj" expected shape
[4096, 4096] dtype f16, got [4096, 2048] dtype f16`. Silent fallbacks
and best-effort recovery are forbidden at module boundaries. A human
reading the error should know which module to open; an agent reading
the error should be able to correct its own mistake from the message
alone.

**No cleverness that hides the public surface.** Macros beyond trivial
registration (`REGISTER_MODEL`, `QINF_ASSERT`), template
metaprogramming, runtime reflection, and multi-level indirection all
make the code harder to reason about — for humans and agents both.
Prefer verbose and explicit over clever and terse. Generated code is
permitted under `src/generated/`, must be produced by a committed
generator script, and must never be hand-edited.

**Single canonical path per concept.** There is one way to register a
model (`REGISTER_MODEL`), one way to declare a weight slot
(`WeightSlot`), one way to bind a tensor (`TensorNameMap` in the model
recipe). If a second path appears, delete one; do not deprecate in
place.

## What This Enables

The refactor commits to these outcomes — they are the justification for the work:

- **Qwen 3.6 support** without forking the engine.
- **Jamba/Zamba support** by composing SSM + Attention + MoE modules.
- **Future hybrid architectures** by adding one new layer module and one new
  model recipe.
- **Per-module benchmarking** — profile and optimize each layer type in isolation.
- **Per-module quantization research** — experiment with DeltaNet state
  compression without risking the attention path.
- **Clean test boundaries** — each module has its own unit tests, independent
  of model integration tests.

Further capabilities the architecture could enable — speculative, unproven,
or requiring their own design work — are collected in
[`future-work.md`](future-work.md). Those are not commitments of this refactor
and must not be used to justify scope creep within any phase.
