# Gemma Implementation Plan (G1 → G4)

Executable plan to bring Gemma 1, 2, 3, and 4 (text-only) into Qwenium.

This document is the companion to
[`plan-modular-layer-arch-impl.md`](plan-modular-layer-arch-impl.md). Where the
modular plan refactors Qwen-shaped code into a layer-typed architecture, this
plan **uses Gemma as the forcing function** to verify that the architecture is
genuinely cross-family and not accidentally Qwen-specific. Supporting Gemma is
the means; clean modules and clearer boundaries are the end. We will not grow a
model zoo: each Gemma generation is added because its requirements expose a
specific structural assumption that must be invalidated and replaced with a
proper interface.

This plan does **not** re-litigate the modular spec
([`modular-layer-architecture.md`](modular-layer-architecture.md)). It applies
the same TDD discipline, the same branch strategy, and the same per-phase gate
philosophy.

---

## Sequencing relative to the modular plan

- **All four Gemma phases (G1 → G4) precede modular Phase 4** (Metal kernel
  surgery). Reasoning: even the "mandatory" MoE fusion benefits from being
  designed against two MoE families (Qwen 3.6 A3B and Gemma 4 A4B) rather
  than one — designing against only Qwen 3.6 risks a refactor when A4B's
  router shape differs. The 4a/4b split that previously protected against
  premature fusion is folded back into a single Phase 4 once Gemma has done
  the interface validation. See [`plan-modular-layer-arch-impl.md`](plan-modular-layer-arch-impl.md)
  → Phase 4 for the merged shape.
- **During the Gemma window, MoE runs in `QINF_MOE_FALLBACK=ON` mode.** This
  is the single accepted cost of the Gemma-first ordering: Qwen 3.6 stays in
  fallback for the duration of G1 → G4. If a production workload depends on
  Qwen 3.6 throughput today, that cost is real and the ordering should be
  revisited; otherwise the architectural payoff dominates.
- **G2 → G3 → G4 are not strictly ordered by date** — they are ordered by
  *interface stress*. Each generation introduces exactly one new structural
  requirement that the previous generation did not. Implementing them in order
  means each PR has a single new dimension of risk.
- **Vision and audio encoders are out of scope** for every phase below. Gemma 3
  and Gemma 4 will be supported as text-only checkpoints. Multimodal is a
  separate plan with its own phasing.

---

## Working Discipline

### TDD cycle

Identical to the modular plan: Red → Green → Refactor → Commit. Phase G1
inherits the modular plan's TDD discipline from PR G1.1 onward; there is no
"mechanical extraction" exception here because we are adding behavior, not
rearranging it.

### Test gates per phase

| Phase | Gate |
|---|---|
| G1 | Gemma 1 (2B + 7B) produces logit-level agreement with HF transformers reference on canary prompts (top-8 rank preservation, top-1 divergence index ≥128). All Qwen tests still pass. Modular Phase 2 gate still holds. |
| G2 | Gemma 2 (2B + 9B) matches reference within tolerance on canary prompts. Soft-cap, sandwich norm, and alternating sliding/global attention are exercised by unit tests in isolation and by the integration suite. G1 gate still holds. |
| G3 | Gemma 3 (4B text-only checkpoint) matches reference. QK-norm replaces soft-cap; dual RoPE base frequencies (10K local / 1M global) load correctly from GGUF. 5:1 local:global pattern verified. G2 gate still holds. |
| G4 | Gemma 4 26B-A4B (MoE, text-only, IT) matches reference in `QINF_MOE_FALLBACK=ON` — performance is not a gate, it lands with modular Phase 4. **PLE and shared-KV are inactive in the 26B-A4B checkpoint** (`embedding_length_per_layer_input=0`, `shared_kv_layers=0`); their interfaces ship in G4 but are gated by **synthetic-config unit tests**, not by the canary. p-RoPE pruning fraction loads from GGUF. Final logit soft-cap (cap=30) is re-enabled at the LM head. Per-layer attention shapes (head_dim_k/v, n_kv_heads, rope_dim, rope_base, window) are first-class on `LayerSpec`. The A4B router config surfaces through the same `MoEConfig` struct as Qwen 3.6 so modular Phase 4 can host both families behind one fused op. The dense 31B variant — which *would* exercise PLE and shared-KV end-to-end — is deferred (no checkpoint in hand); add it as a follow-up PR once available. |

### Branch strategy

- One long-lived branch per Gemma phase: `phase-g1-gemma1`, `phase-g2-gemma2`,
  `phase-g3-gemma3`, `phase-g4-gemma4`.
- PRs land into the phase branch, not into `main`.
- A phase branch merges into `main` only when its gate is met.
- `main` stays shippable. Every Gemma phase is independently revertable.

### Rollback

- Every PR must be revertable as a single `git revert` without collateral.
- A phase's introduction of a new structural feature (sandwich norm, sliding
  window, QK-norm, PLE, shared KV) lands behind a feature flag at the
  *recipe* level (not the kernel level) until the gate is met. That way a
  bug in the new feature cannot regress Qwen or earlier-generation Gemma.

### Canary prompt set extension

- Gemma-specific canary prompts land in `tests/fixtures/canary_prompts_gemma.txt`
  (sibling of the Qwen set). Prompts cover: short prompt + long generation;
  long-context retrieval (≥4K tokens) to stress sliding window; multi-turn
  using the Gemma chat template (`<start_of_turn>` / `<end_of_turn>`);
  grammar-constrained output through the existing TokenTrie machinery.
- Reference logits for the canary set are produced by HF `transformers` with
  `attn_implementation="eager"` (not SDPA) on float32, captured once per Gemma
  generation, checked into `tests/fixtures/gemma_ref_logits/`.

### Out of scope (across all Gemma phases)

- Vision encoder (SigLIP for G3, learned 2D pos for G4).
- Audio encoder (USM-style conformer, G4).
- Bidirectional attention mode (Gemma encoder variants).
- Non-text instruction tuning: function calling, code execution, etc.

---

## Phase G1 — Gemma 1 (interface validation)

**Goal:** support Gemma 1 (2B and 7B) end-to-end. Use the work to expose every
loader/tokenizer/dispatcher seam that is implicitly Qwen-shaped, and replace it
with a family-generic interface. **The deliverable is not "Gemma runs"; the
deliverable is "the architecture can host a non-Qwen family."**
**Duration estimate:** 2–3 weeks, one engineer.
**Branch:** `phase-g1-gemma1`.
**Gate:** see table above.

### Why G1 is the interface validator

Gemma 1 is a pure decoder-only transformer (attention + dense FFN + RMSNorm +
RoPE + GQA). Mathematically it is the simplest possible non-Qwen family. Every
piece of friction that surfaces while adding it is therefore an interface
problem, not an algorithmic problem. That is exactly the falsification test
the modular spec needs.

The friction is concentrated in five places, all known up front:

1. Loader architecture allow-list ([`gguf_loader.cpp:215`](src/loader/gguf_loader.cpp:215))
   rejects anything that is not `qwen{2,3,35,35moe}`.
2. Tensor-name validation assumes Qwen tensor naming; Gemma uses
   `blk.N.attn_q.weight` / `blk.N.ffn_gate.weight` style which is mostly
   compatible but the validators are hand-coded per family.
3. Tokenizer ([`tokenizer.cpp`](src/loader/tokenizer.cpp)) is BPE with explicit
   merge rules. Gemma's GGUF tokenizer is BPE-with-byte-fallback derived from
   a SentencePiece-trained vocabulary, with `▁` (U+2581) for spaces and
   different special-token IDs (BOS=2, EOS=1, PAD=0).
4. Dispatcher chains ([`chat.cpp:59-64`](src/cli/chat.cpp:59),
   `complete.cpp`, `main.cpp`) hard-code the Qwen family.
5. Three Gemma-1-specific correctness footguns — see "Correctness footguns"
   below.

### Correctness footguns (Gemma RMSNorm, embedding scale, GeGLU)

These are known traps. Each gets its own targeted test.

1. **RMSNorm uses `(1 + weight)` form, not `weight`.** Gemma stores the
   normalization weight such that the operation is `output = (x / rms(x)) * (1 + w)`.
   Implementing the standard `x * w` form silently produces near-zero outputs
   in the first few layers. This is the single most common Gemma porting bug.
2. **Input embeddings are scaled by `sqrt(hidden_size)`** in fp32 before the
   first transformer block. This must be done in fp32 even when the rest of
   the graph is bf16/f16, otherwise the first layer's logits are off by
   ~`sqrt(d)` and downstream attention probabilities collapse.
3. **GeGLU uses the tanh-approximate GELU** (`gelu_pytorch_tanh`), not the
   exact erf-based GELU. Wrong variant produces drift that compounds across
   28 layers.
4. **Tied input/output embeddings.** No separate `output.weight` — the logit
   head reuses `token_embd.weight`. Loader and graph builder must both honor
   this.

### PRs

**PR G1.0 — Reference logit harness.** Tests first: a Python script in
`scripts/gen_gemma_ref_logits.py` produces float32 reference logits from HF
`transformers` for Gemma 1 2B on the canary prompt set, with
`attn_implementation="eager"`. Stores into `tests/fixtures/gemma_ref_logits/g1_2b/`.
Includes a checksum so the file's identity is auditable in PR review. Gate:
script runs deterministically and produces identical bytes on two consecutive
invocations.

**PR G1.1 — Loader generalization (architecture allow-list).** Tests first:
parametrized loader test loads minimal GGUFs for `qwen3`, `gemma`, and a
deliberately-malformed `unknown` arch. The first two pass; the third fails
with the spec-mandated error format ("expected one of: ..., got '...'").
Refactor `validate_architecture` ([`gguf_loader.cpp:213`](src/loader/gguf_loader.cpp:213))
to accept an extensible allow-list driven by the model registry, not a
hand-coded `if`/`||` chain. Gate: tests green, no Qwen regression.

**PR G1.2 — Tensor inventory schema generalization.** Tests first: the four
spec failure modes from `WeightSlot` (unknown name; missing slot; shape
mismatch; dtype mismatch) fire correctly for both Qwen and a synthetic Gemma
inventory. Generalize `validate_qwen35moe_inventory`-style validators
([`gguf_loader.cpp:433`](src/loader/gguf_loader.cpp:433)) into a
schema-per-family lookup keyed off `metadata_.architecture`. Gate: existing
Qwen inventory tests pass; new Gemma inventory test passes.

**PR G1.3 — Tokenizer: byte-fallback BPE with `▁` normalization.** Tests
first: encode/decode round-trip on a curated string set covering ASCII,
multi-byte UTF-8, leading-space sensitivity, digit splitting, and the
`<start_of_turn>` / `<end_of_turn>` special tokens. Reference outputs come
from `transformers.AutoTokenizer.from_pretrained("google/gemma-2b")`. Extend
`Tokenizer` ([`tokenizer.cpp`](src/loader/tokenizer.cpp)) to handle:
(a) `▁` (U+2581) ↔ space substitution as a normalizer step,
(b) byte-fallback for unknown codepoints (already required by Gemma's GGUF),
(c) configurable special-token table loaded from GGUF (Gemma's BOS/EOS/PAD
IDs differ from Qwen's). Do **not** introduce a separate `SentencePieceTokenizer`
class — the GGUF serialization is BPE-shaped already; the diff is normalization
and special-token plumbing, not a new algorithm. Gate: round-trip tests green
on both Qwen and Gemma test strings.

**PR G1.4 — Gemma-1 layer ops.** Tests first, three small unit tests, each
against a hand-computed oracle:
(a) `build_rms_norm_gemma` produces `x/rms(x) * (1 + w)` for a synthetic
3-element vector;
(b) `build_embed_scale` multiplies a synthetic embedding by `sqrt(d_model)`
in fp32;
(c) `build_geglu_tanh` matches PyTorch's `gelu_pytorch_tanh` on a synthetic
2-element input.
Implement these as additions to `src/layers/norm.cpp` and `src/layers/ffn.cpp`,
gated by a per-layer config flag (no breaking change to existing builders).
Gate: three unit tests green; bit-for-bit Qwen output unchanged.

**PR G1.5 — Model recipe `src/models/gemma1.cpp`.** Tests first: integration
test that loads a small Gemma 1 GGUF (2B-Q4 or, for CI speed, a 100M-param
synthetic checkpoint with the same architecture string) and produces logits
agreeing with the reference harness within the canary tolerance. Build the
recipe by composing the existing `transformer_block` machinery with the new
RMSNorm-(1+w), embedding scale, and GeGLU-tanh hooks. **No edits to existing
Qwen recipes.** Gate: integration test green; reference-logit gate met.

**PR G1.6 — Dispatcher generalization.** Tests first: a registry-driven
dispatcher test where a fake architecture string maps to a fake forward-pass
factory. Replace the if/else chain in [`chat.cpp:59-64`](src/cli/chat.cpp:59),
mirrored in `complete.cpp` and `main.cpp`, with a `REGISTER_MODEL`-style
table lookup (the modular spec already calls for this in Phase 2; G1 is what
forces it to actually exist). Gate: all existing CLI/server integration
tests pass; `qwen3` and `gemma` dispatch through the same code path.

**PR G1.7 — Chat template + special-token plumbing.** Tests first: render
the canonical Gemma chat conversation
(`<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`) from a
`ChatMessage` vector and tokenize it; verify the special-token IDs match
the GGUF metadata. Implement the Gemma template alongside the Qwen template;
selection driven by `metadata_.architecture`. Gate: end-to-end chat session
with Gemma 1 IT 2B produces sensible multi-turn output (qualitative); special
tokens roundtrip.

**PR G1.8 — *(removed).*** The `src/cli/pretokenized_literals*.h` machinery
and the standalone `qwenium::Grammar` class were superseded by `GrammarVocab`
+ `TokenTrie` (the resolve-once path). The headers and `src/sampling/grammar.cpp`
were dead code at the time the original Gemma plan was written; G1.8 has been
deleted from the phase rather than perpetuating an unused literal cache.

### Phase G1 exit criteria

- [ ] Gemma 1 2B and 7B reach the canary logit-agreement gate against the HF
      reference harness.
- [ ] Loader allow-list, tensor schema, dispatcher, and tokenizer all run
      family-generic — no `if architecture == "gemma"` branches outside the
      *recipe* file.
- [ ] All four correctness footguns (RMSNorm `(1+w)`, embed scale, GeGLU
      tanh, tied embeddings) have dedicated unit tests with hand-computed
      oracles.
- [ ] Qwen integration tests unchanged (no logit drift).
- [ ] **Architecture self-check:** changes to a layer module file required to
      add Gemma 1 are zero. If any existing Qwen layer file was modified to
      accommodate Gemma, the modular spec's locality invariant is at risk —
      flag it in the phase retrospective and refactor before G2.

### What G1 retires

After G1 lands, these assumptions are formally dead in the codebase:
- "Architecture is Qwen" as a hard-coded conditional outside model recipes.
- "Tokenizer is BPE-with-merges as Qwen ships it" — replaced by BPE with
  configurable normalizer and special-token table.
- "Logit head is a separate output projection" — tied embeddings are now a
  first-class option.
- "RMSNorm is `x * w`" — the form is now per-recipe configurable.

---

## Phase G2 — Gemma 2 (sandwich norm, sliding window, soft-cap)

**Goal:** support Gemma 2 (2B, 9B, 27B). Each new feature retires one
structural assumption from the post-G1 codebase.
**Duration estimate:** 2 weeks.
**Branch:** `phase-g2-gemma2`.
**Gate:** see table above.

### Structural changes Gemma 2 forces

1. **Sandwich norm.** Gemma 2 wraps both the attention block *and* the FFN
   block with both pre-norm and post-norm: `x + post_norm(attn(pre_norm(x)))`
   instead of `x + attn(pre_norm(x))`. Today's `transformer_block`
   ([`transformer_block.cpp:31-102`](src/layers/transformer_block.cpp:31)) is
   pre-norm only.
2. **Alternating sliding/global attention.** Local layers use a 4096-token
   sliding window; global layers use full attention (8192). The pattern is
   "every other layer." This breaks the assumption that every attention layer
   has identical KV-cache topology.
3. **Logit soft-capping.** Attention scores are capped at 50.0 via
   `softcap * tanh(x/softcap)` before the softmax; final logits are capped at
   30.0 the same way. This is a new op in the attention path and a new op in
   the LM head.

### PRs

**PR G2.0 — Reference logit harness for G2.** Same pattern as G1.0, for
Gemma 2 2B (small enough for fast CI), with reference logits stored in
`tests/fixtures/gemma_ref_logits/g2_2b/`.

**PR G2.1 — Sandwich-norm hook in `transformer_block`.** Tests first:
unit test the existing pre-norm path (regression), then a new test that
verifies a `transformer_block` configured with `post_attn_norm` and
`post_ffn_norm` weight slots produces the sandwich form bit-identically to
a hand-rolled reference. Add two optional `LayerWeights` slots
(`w.post_attn_norm`, `w.post_ffn_norm`); when null, behavior is unchanged
(Qwen path). When non-null, post-norm is applied. Gate: green; Qwen output
bit-identical.

**PR G2.2 — Per-layer attention type tag.** Tests first: a `LayerSpec`
struct with `AttentionKind { Global, Sliding }` and a window size. The
**recipe's `from_metadata` factory** extracts the per-layer alternation
from `meta.raw_kv` (Gemma 2 encodes it as a per-layer config) and stores
it on `Gemma2Config`. The loader is not touched. Verify the populated
vector matches the documented pattern for Gemma 2 9B. Gate: green; no
behavior change yet (kind is read but not acted upon).

**PR G2.3 — Sliding-window mask in attention.** Tests first: unit test
`build_attention` with `AttentionKind::Sliding` and window=4 on a synthetic
12-token input; verify the masked attention scores match a hand-computed
oracle. Implement the sliding-window mask via `ggml_diag_mask_inf` plus a
window offset, or by extending the existing causal mask to a banded mask.
**Do not introduce a second attention module — extend the existing one.**
The whole point of the layer-typed architecture is that attention is one
concept; the window is a parameter, not a new type. Gate: oracle test green;
Qwen tests unchanged.

**PR G2.4 — Soft-cap op (attention + final).** Tests first: unit test
`build_softcap(x, cap)` returns `cap * tanh(x/cap)` matching a hand-rolled
oracle to f16 tolerance. Wire it into the attention path conditionally
(parameter `attn_softcap_value` on the layer; null = off). Wire it into
the LM head conditionally (parameter `final_softcap_value` on the model;
null = off). Gate: oracle tests green; Qwen output unchanged when both
parameters are null.

**PR G2.5 — Gemma 2 model recipe.** Tests first: integration test against
the G2.0 reference logits. Compose the recipe from sandwich-norm + per-layer
attention kind + soft-cap. Gate: canary logit-agreement gate met for Gemma
2 2B and 9B.

**PR G2.6 — KV cache topology for mixed sliding/global.** Tests first:
verify that a long-context decode (>4K tokens) with mixed sliding/global
layers produces correct outputs and that the sliding layers' KV slabs are
not over-allocated. Today's KV cache assumes uniform per-layer sizing. Add
a per-layer `kv_capacity` so sliding layers allocate only `window` slots
where appropriate. Gate: long-context test green; total KV memory for
Gemma 2 9B is below the all-global baseline by the expected fraction.

### Phase G2 exit criteria

- [ ] Gemma 2 2B and 9B meet the logit-agreement gate.
- [ ] `transformer_block` supports both pre-only and sandwich norm; the
      switch is a weight-slot presence check, not a config string.
- [ ] Sliding-window attention is a parameter on the existing attention
      module, not a fork.
- [ ] Soft-cap is an opt-in op; Qwen graphs are unchanged when it is null.
- [ ] KV cache supports per-layer capacity.
- [ ] G1 + Qwen tests still pass.

---

## Phase G3 — Gemma 3 (QK-norm, dual RoPE, 5:1 alternation)

**Goal:** support Gemma 3 1B/4B/12B/27B (text-only checkpoints; vision is
out of scope).
**Duration estimate:** 1.5 weeks. Smaller than G2 because we're reusing
G2's sliding/global infrastructure.
**Branch:** `phase-g3-gemma3`.
**Gate:** see table above.

### Structural changes Gemma 3 forces

1. **Soft-cap removed; replaced by QK-norm.** Q and K are RMS-normed *per
   head* before the attention dot product. We already have Q-norm and K-norm
   in the Qwen 3 path ([`transformer_block.cpp:64-79`](src/layers/transformer_block.cpp:64)),
   so the op exists — the question is whether the head-dimensional vs
   hidden-dimensional broadcast semantics match. They do not exactly: Qwen 3's
   norms operate over the head-dim with a per-head weight; Gemma 3's QK-norm
   shares one weight across heads. This is a config, not a new op.
2. **Dual RoPE base frequencies.** Local layers use base 10K, global layers
   use base 1M (then optionally further scaled at the end of pre-training to
   support 128K context). Today's RoPE call assumes one base per model.
3. **Alternation pattern is 5:1, not 1:1.** Five sliding layers, then one
   global, repeating. The G2 per-layer attention-kind tag already supports
   arbitrary patterns — verify, don't rebuild.

### PRs

**PR G3.0 — Reference logit harness for G3.**

**PR G3.1 — QK-norm config dimensionality.** Tests first: unit test that
verifies QK-norm with `weight_shape={head_dim}` (Qwen 3 style) and
`weight_shape={1}` broadcast (Gemma 3 style) produce the same operation
when given the same weights. Add the broadcast option to `build_rms_norm`.
Gate: green; Qwen 3's QK-norm output bit-identical.

**PR G3.2 — Per-layer RoPE base frequency.** Tests first: extend `LayerSpec`
with `rope_base`. The recipe's `from_metadata` factory reads the value(s)
from `meta.raw_kv` and populates the per-layer vector (some Gemma 3 GGUFs
encode it per-layer-type, others encode it as a global plus an override
array — handle both in the recipe; the loader is untouched). Gate: green.

**PR G3.3 — Per-layer RoPE call site.** Tests first: unit test that two
synthetic layers, configured with different `rope_base`, produce different
attention outputs in the expected way. Refactor the RoPE call in
[`transformer_block.cpp:70-83`](src/layers/transformer_block.cpp:70) to read
`rope_base` from `LayerSpec` rather than `Hyperparams`. Gate: Qwen tests
unchanged (Qwen layers all share one base, so the spec is uniform).

**PR G3.4 — Gemma 3 model recipe.** Compose from G2 sliding/global
infrastructure + QK-norm + dual RoPE. Gate: canary gate met for Gemma 3
4B-text.

**PR G3.5 — Verify 5:1 pattern loads correctly.** Tests first: load a
Gemma 3 12B checkpoint metadata and assert the per-layer attention-kind
vector matches the documented `[L,L,L,L,L,G]*` pattern. Gate: green.

### Phase G3 exit criteria

- [ ] Gemma 3 4B-text meets the logit-agreement gate.
- [ ] QK-norm supports both per-head and broadcast weights.
- [ ] RoPE base frequency is a per-layer parameter.
- [ ] No new attention module variant; G3 is config + recipe only.
- [ ] G2 + G1 + Qwen tests still pass.

### Note on multimodal scope

Gemma 3 ships with a SigLIP vision encoder. **We do not implement it in
G3.** The text-only checkpoints are a real distribution; we target those.
Vision is a separate plan. We do, however, need to make sure the loader
does not choke on multimodal-extended GGUFs that include vision tensors —
unknown tensor names should be tolerated with a warning, not rejected, when
they belong to declared-out-of-scope subsystems. PR G3.6 (optional) handles
this: extend the **gemma3 recipe's registered inventory validator** to
skip tensors matching `mm.*` / `vision.*` prefixes (the loader is generic
and dispatches inventory validation through the registry — no edit to
`src/loader/` is required or permitted). Gate: a
multimodal Gemma 3 GGUF loads, the vision tensors are not bound to any
slot, and inference proceeds on text only.

---

## Phase G4 — Gemma 4 (PLE, shared KV, p-RoPE, MoE A4B)

**Goal:** support Gemma 4 MoE 26B-A4B (text-only, instruction-tuned).
The dense 31B variant is deferred (no checkpoint in hand) and is the
natural follow-up that will exercise the PLE and shared-KV pathways
end-to-end.
**Duration estimate:** 3–4 weeks.
**Branch:** `phase-g4-gemma4`.
**Gate:** see table above.

### Target checkpoint

`gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` — arch=`gemma4`, 30 blocks, embed=2816,
128 experts top-8, expert FFN=704, context=262K, sliding window=1024,
`final_logit_softcapping=30`. Critical metadata to read in
[`temp/gemma4_debug_gguf.txt`](temp/gemma4_debug_gguf.txt) before starting:
`shared_kv_layers=0`, `embedding_length_per_layer_input=0`,
`head_count_kv` is a per-layer array of 30, `sliding_window_pattern` is
a per-layer array, `key_length=512 / value_length=512` (global) vs
`key_length_swa=256 / value_length_swa=256` (sliding),
`rope.freq_base=1e6` (global) / `rope.freq_base_swa=1e4` (local),
`rope.dimension_count=512` / `rope.dimension_count_swa=256`,
`tokenizer.ggml.model='gemma4'`, BOS=2 / EOS=106. **Re-dump the tensor
list** (the existing dump cuts off mid–chat-template) so PR G4.5 can
verify the inventory.

### Structural changes Gemma 4 forces

1. **Per-layer attention shapes — not just per-layer kind.** Sliding and
   global layers in Gemma 4 differ in `head_dim_k`, `head_dim_v`, RoPE
   dimension count, and RoPE base — not only in mask shape. `LayerSpec`
   must carry per-layer `head_dim_k`, `head_dim_v`, `n_kv_heads`,
   `rope_dim`, `rope_base`, `attn_kind`, `window`. Also: this is the first
   model where `n_heads * head_dim ≠ d_model` (16 × 512 = 8192 vs 2816);
   the recipe must not assume that identity.
2. **Per-Layer Embeddings (PLE).** A second, lower-dimensional embedding
   table, looked up per token, projected, and added to the residual stream
   inside every decoder block. This is the first time Qwenium needs a model
   with a *second residual signal*. The block interface so far carries one
   residual tensor; PLE forces a second input. **Inactive in 26B-A4B
   (`embedding_length_per_layer_input=0`)** — the interface ships in G4 but
   is validated by synthetic-config unit tests; the dense 31B follow-up
   will exercise it end-to-end.
3. **Shared KV cache across layers.** The last `num_kv_shared_layers`
   layers do not compute their own K and V projections — they reuse the K
   and V tensors from the last non-shared layer of the same attention type.
   Each shared layer still computes its own Q. This is a fundamental change
   to KV ownership: the cache tensor for layer N may be a *reference* to
   layer M's tensor. **Inactive in 26B-A4B (`shared_kv_layers=0`)** —
   interface ships in G4, validated by a synthetic state-isolation test;
   end-to-end exercise waits for the dense 31B follow-up.
4. **Pruned RoPE (p-RoPE) on global layers only.** Only a fraction of the
   head dimensions receive RoPE rotation; the rest are passed through.
   Local layers use standard RoPE (full rotation). The pruning fraction
   is derived from `rope.dimension_count` vs the per-layer `head_dim_k`.
   This is a new RoPE variant.
5. **Final logit soft-cap is back.** Gemma 3 retired soft-cap in favor of
   QK-norm; Gemma 4 keeps QK-norm *and* re-enables the final-logit cap at
   30.0. The G2.4 op is reused at the LM head; attention soft-cap stays
   off.
6. **Per-layer sliding pattern.** `sliding_window_pattern` ships as a
   per-layer array of 30. Do not hard-code Gemma 3's 5:1 — read the array.
7. **MoE A4B variant** runs in fallback mode during G4. 128 experts, top-8,
   expert FFN=704, **no shared expert** (Qwen 3.6 has one — `MoEConfig`'s
   shared-expert field stays nullable). The fused op lands in modular
   Phase 4, designed against both router shapes simultaneously.
8. **Tokenizer and chat template.** `tokenizer.ggml.model='gemma4'`,
   EOS=106 (not Gemma 1's EOS=1). The IT chat template is a large Jinja
   with tool-calling; G4 implements the basic `<start_of_turn>` /
   `<end_of_turn>` rendering only — **tool-calling rendering is out of
   scope** for this phase.
9. **Sliding window sizes are model-size-dependent:** 512 for E2B/E4B
   (out of scope, edge models), 1024 for 31B and A4B.

### Why this phase is large

PLE and shared KV both touch interfaces that have not yet been stressed.
Doing them in the same phase is justified because they only need to cohere
at the recipe level — but each one needs careful interface design so it
does not become Gemma-specific cruft polluting the block builder.

### PRs

**PR G4.0 — Reference logit harness for G4.** Targets 26B-A4B-It.
Downsample the canary set if a full reference pass is too expensive for
CI; document the budget choice in the script header.

**PR G4.1 — Per-layer attention shapes on `LayerSpec`.** Tests first:
unit test that a `LayerSpec` vector populated from synthetic per-layer
arrays produces the expected `head_dim_k`, `head_dim_v`, `n_kv_heads`,
`rope_dim`, `rope_base`, `attn_kind`, `window` per layer. The recipe's
`from_metadata` factory reads `head_count_kv` (array), `key_length` /
`key_length_swa`, `value_length` / `value_length_swa`,
`rope.dimension_count` / `rope.dimension_count_swa`, `rope.freq_base` /
`rope.freq_base_swa`, and `sliding_window_pattern` (array) from
`raw_kv` and assembles the per-layer vector. Refactor `attention.cpp`
to read these from `LayerSpec` rather than `Hyperparams`. Gate: green;
Qwen and Gemma 1/2/3 paths bit-identical (their per-layer shapes are
uniform, so the spec degrades to the existing scalar config).

**PR G4.2 — Block interface: second residual slot.** Tests first: unit
test a `transformer_block` invoked with a null secondary residual (Qwen,
Gemma 1/2/3 path) and with a non-null secondary residual that is added at
a specified injection point. Verify the secondary path is bit-identical to
not-passing-it when the input is zeros. **Spec the injection point as a
member of the block's interface, not as an internal hardcode.** This is the
critical interface change of G4 — get it right or G4 contaminates the block
forever. Gate: green; all earlier tests bit-identical.

**PR G4.3 — PLE table + projection (interface, inactive in A4B).** Tests
first: unit test PLE lookup on a synthetic 4-token input with a
synthetic 32-dim PLE table; verify the projected output is added to the
residual at the documented injection point. Implement
`build_ple(token_ids, ple_table, ple_proj)` in a new `src/layers/ple.cpp`
(PLE is a layer concept, not a model concept). Gate: oracle test green.
**No canary gate** — `embedding_length_per_layer_input=0` in 26B-A4B
means PLE is a no-op there. The dense 31B follow-up promotes this to a
canary gate.

**PR G4.4 — KV cache reference-mode (interface, inactive in A4B).**
Tests first: a synthetic state-isolation test that allocates KV cache
for a 26-layer model with the last 6 layers configured as "share with
layer 19," runs prefill, runs decode, and verifies (a) layer 25's K and
V tensors at any time alias layer 19's, (b) writes to layer 19's K/V
in decode are visible at layer 25 without copy. Add a `KvSharingSpec`
to `LayerSpec` and refactor the KV cache to support reference-mode
slots (no allocation, just a pointer to the producing layer's slot).
**Do not unify this with the existing KV-vs-state distinction** —
those are different axes. Gate: state-isolation test green;
non-reference KV cache unchanged. **No canary gate** —
`shared_kv_layers=0` in 26B-A4B means the path is dormant. Dense 31B
follow-up promotes this to a canary gate.

**PR G4.5 — p-RoPE op.** Tests first: unit test `build_rope_pruned(x,
base, prune_fraction)` rotates only the first `(1 - prune_fraction) *
head_dim` elements and passes the remainder through. Verify against a
hand-computed oracle. Add `RopeKind { Standard, Pruned }` to
`LayerSpec`; the recipe derives `prune_fraction` per layer from
`rope_dim` vs `head_dim_k`. Gate: oracle test green; standard RoPE
bit-identical.

**PR G4.6 — Final logit soft-cap re-enable.** Tests first: unit test
the LM head with `final_softcap_value=30.0` against a hand-rolled
oracle (reuse G2.4's `build_softcap`). Wire it into the recipe-level
LM head; attention soft-cap stays off (QK-norm replaces it). Gate:
oracle green; null-cap path unchanged.

**PR G4.7 — Tokenizer and basic chat template.** Tests first: encode/
decode round-trip on a curated string set with `tokenizer.ggml.model
='gemma4'`, BOS=2, EOS=106; render a single-turn `<start_of_turn>` /
`<end_of_turn>` conversation and verify special-token IDs against
metadata. **Tool-calling Jinja rendering is out of scope** —
acknowledge it in the template registration so the IT model still
runs basic chat. Gate: round-trip + single-turn rendering green.

**PR G4.8 — Gemma 4 MoE A4B recipe (`src/models/gemma4.cpp`).**
Tests first: integration test against the G4.0 reference logits.
Compose: per-layer attn shapes (G4.1) + sandwich norm (G2) + dual
RoPE local standard / global p-RoPE (G4.5) + per-layer sliding
pattern (read array; do not assume 5:1) + final logit soft-cap
(G4.6) + MoE fallback FFN. PLE (G4.3) and shared-KV (G4.4) are wired
through the recipe but inactive given `embedding_length_per_layer_input
=0` and `shared_kv_layers=0`. Register inventory validator,
`TokenizerConfig`, and `ChatTemplate` in this file (no edits to
`src/loader/`). The A4B router shape (128 experts, top-8, expert
FFN=704, no shared expert) flows through the shared `MoEConfig`
struct. **No edits to existing recipes.** Gate: canary
logit-agreement met on Gemma 4 26B-A4B-It in `QINF_MOE_FALLBACK=ON`.
Performance is explicitly *not* a gate — that lands with the fused
op in modular Phase 4.

**PR G4.9 — Long-context verification.** Tests first: a long-prefill
(≥64K tokens, scaling toward 256K as memory permits) through Gemma 4
26B-A4B with the per-layer sliding/global pattern and global p-RoPE.
Verify KV memory is within the documented budget and that decode
remains correct. Gate: green.

**Deferred — Gemma 4 dense 31B follow-up.** Once a 31B checkpoint is
available, add a recipe-only PR (no new ops) that promotes G4.3 (PLE)
and G4.4 (shared-KV) to canary gates and validates them end-to-end.
This is the moment the architecture's PLE / shared-KV claims become
operationally backed.

### Phase G4 exit criteria

- [ ] Gemma 4 26B-A4B-It meets the logit-agreement gate in
      `QINF_MOE_FALLBACK=ON`.
- [ ] Per-layer attention shapes (`head_dim_k`, `head_dim_v`,
      `n_kv_heads`, `rope_dim`, `rope_base`, `attn_kind`, `window`) are
      first-class on `LayerSpec`. The assumption
      `n_heads * head_dim == d_model` is dead.
- [ ] PLE is a layer concept living in `src/layers/ple.cpp`, validated
      by synthetic-config unit tests. End-to-end exercise is owed to
      the deferred dense 31B PR.
- [ ] KV cache supports reference-mode slots; the mechanism is generic,
      not Gemma-named. Validated by a synthetic state-isolation test;
      end-to-end exercise is owed to the deferred dense 31B PR.
- [ ] p-RoPE is a parameterized variant of the existing RoPE op, not a
      new module.
- [ ] Final logit soft-cap is re-enabled (cap=30) at the LM head; G2.4
      op is reused unchanged.
- [ ] Per-layer sliding pattern is read from
      `sliding_window_pattern`; the 5:1 shortcut from G3 is not
      hard-coded.
- [ ] MoE A4B runs correctly in fallback mode; its router config flows
      through the shared `MoEConfig` struct (with nullable
      shared-expert) so modular Phase 4 can host both Qwen 3.6 and
      Gemma 4 A4B behind one fused op without family-specific branches.
- [ ] Tokenizer (`tokenizer.ggml.model='gemma4'`, BOS=2, EOS=106) and
      basic `<start_of_turn>` / `<end_of_turn>` chat template work;
      tool-calling Jinja is documented out of scope.
- [ ] G3 + G2 + G1 + Qwen tests still pass.

### Out of scope for G4

- Dense 31B end-to-end validation (no checkpoint in hand) — interfaces
  ship, validation deferred to follow-up PR.
- Edge variants E2B and E4B (different sliding window sizes; same
  architecture otherwise — adding them is a config tweak, not a phase).
- Tool-calling chat template rendering (the IT model's Jinja is large;
  basic turn-tagged rendering only).
- Vision encoder (learned 2D pos + multidim RoPE).
- Audio encoder (USM-style conformer).
- Altup (Google explicitly omitted it; we do likewise).

---

## Cross-cutting concerns

### Loader and tokenizer are family-agnostic by contract

After the pre-G2 plumbing PRs landed, no Gemma phase touches `src/loader/`
or `src/core/`:

- **GGUF KVs:** the loader stores every scalar KV in `ModelMetadata::raw_kv`.
  Family-specific fields are read by the recipe's `from_metadata` factory
  via `raw_kv.get_uint32("<gguf-key>")` (and the `_float`/`_bool`/etc.
  variants). The loader does not gain per-family extraction code.
- **Inventory validation:** each recipe registers its own validator
  through the model registry. Adding a Gemma generation means writing a
  validator in the recipe file and registering it — never editing
  `gguf_loader.cpp`.
- **Tokenizer:** the tokenizer is a generic engine consuming a
  `TokenizerConfig` (normalizer kind, byte-fallback, special-token
  table). Each recipe registers its config; the tokenizer file contains
  no architecture branches.
- **Chat templates:** registered alongside the recipe via the
  `ChatTemplate` interface. CLI/server look up by architecture string.

If a Gemma phase finds itself wanting to edit `src/loader/`,
`src/core/`, `cli/chat.cpp`, or `cli/main.cpp`, treat it as an interface
defect and surface it. The CI guard rejects new architecture string
literals in those files.

### Reference harness

- All four phases share `scripts/gen_gemma_ref_logits.py`. It must remain
  deterministic across `transformers` versions. Pin the `transformers`
  version in `myenv/` and document the pin in the script's header.
- Reference logits are versioned by `(gemma_generation, model_name,
  transformers_version, prompt_set_version)`. Regenerating them is a
  deliberate act, not a side effect.

### Quantization compatibility

K-Quants are architecture-agnostic and work for Gemma without changes.
TurboQuant and SnapKV were removed (outside the workload envelope); no
KV-precision or KV-eviction interaction to verify.

### Speculative decoding

Speculative decoding is architecture-agnostic per the modular spec, but
the draft model needs to share the target's tokenizer. For Gemma, the
natural draft is a smaller Gemma of the same generation. We do not block
the Gemma phases on speculative — speculative continues independently.

### Grammar / TokenTrie

Architecture-agnostic. Add a Gemma pretokenized literal cache (PR G1.8)
once and reuse it across G2/G3/G4.

### Chat templates

Each Gemma generation reuses the same `<start_of_turn>` / `<end_of_turn>`
template. Implement once in G1 (PR G1.7) and reuse.

---

## Timeline Sketch

| Phase | Estimate | Cumulative |
|---|---|---|
| G1 | 2–3 weeks | 2–3 weeks |
| G2 | 2 weeks | 4–5 weeks |
| G3 | 1.5 weeks | 5.5–6.5 weeks |
| G4 | 3–4 weeks | 8.5–10.5 weeks |

These estimates assume one engineer. They cover correctness only, not
kernel performance — all Metal kernel work (fused MoE, DeltaNet, attention
review) lives in modular Phase 4, which follows G4 and is now informed by
both families. During the Gemma window MoE runs in
`QINF_MOE_FALLBACK=ON`; that is the accepted cost of the Gemma-first
ordering.

---

## Success Looks Like

**At the end of G1:**
- A non-Qwen family runs on Qwenium end-to-end.
- The loader, tokenizer, and dispatcher are family-generic at the seam,
  not at the leaf.
- The codebase has earned the "models are recipes" claim for the first
  time. Modular Phase 4 can now be designed against a validated
  interface rather than a Qwen-only one.

**At the end of G2:**
- Sandwich norm, sliding-window attention, and soft-cap are all
  parameters on existing modules — no parallel module hierarchy.
- KV cache supports per-layer capacity.
- Gemma 2 27B runs with mixed local/global attention without a perf cliff.

**At the end of G3:**
- QK-norm is one op with two weight-shape modes, not two ops.
- RoPE base frequency is per-layer.
- Adding a Gemma generation is now a config-and-recipe job, not a layer-op
  job. (Validated by G3 being smaller than G2.)

**At the end of G4:**
- PLE exists as a generic layer concept that any future model can adopt.
- KV cache supports reference-mode slots — the next family that wants
  cross-layer KV sharing inherits it for free.
- The MoE A4B router shape flows through the shared `MoEConfig` struct
  alongside Qwen 3.6 A3B, so the fused MoE op landing in modular Phase
  4 hosts both families behind one kernel — designed against two
  topologies from day one, not refactored to fit a second.
- The architecture has hosted *four* generations of a non-Qwen family
  with no interface contamination from any of them. The modular spec's
  load-bearing claim is no longer hopeful — it is operational.

If at any phase the implementation requires a non-trivial change to an
existing layer module file (not a new optional parameter, but actual
edits to logic that other recipes depend on), pause and treat it as an
interface defect in the modular spec. **The plan's success metric is not
"Gemma works"; it is "the architecture hosted Gemma without bending."**
