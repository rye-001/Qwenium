# Gemma Vision — Implementation Plan

**Status:** Implementation plan. Companion to
[`plan-multimodal-eval.md`](plan-multimodal-eval.md), which is the
architecture evaluation and decision substrate. This doc does **not**
re-litigate scope, family choice, or the cross-family deferral — those are
binding constraints (C1–C4) decided in the eval. It states the architectural
shape and the phasing.

**Scope (from eval, sharpened by review):** **Gemma 3 specifically** — not
"Gemma vision" as a family. Verified during review: Gemma 3 uses SigLIP,
**Gemma 3n uses MobileNetV5** (a structurally different vision encoder),
and **Gemma 4V uses its own graph**. Generalizing across Gemma vision
variants would require multiple encoder graphs per variant — exactly the
path-zoo failure mode. Gemma 3n and Gemma 4V are deferred under separate
decisions, with the same kind of reopen criteria the eval doc uses for
Qwen-VL. Scope (a) vision-as-input only. Qwen-VL deferred (per eval).
Audio out of scope.

**Single-tile only — Pan & Scan deliberately not implemented in v1.** Gemma
3 ships with a "Pan & Scan" preprocessing step that adaptively crops
non-square / high-resolution images into multiple 896×896 tiles, each
producing 256 tokens. A single landscape photo can therefore produce
1024+ tokens, not 256. We **scope Pan & Scan out** of the initial
implementation: the engine accepts only images that fit a single 896×896
tile (host-side resize to fit, no multi-tile fan-out). Stated constraint,
not silent omission. Reopen criterion: a named user demonstrates the
single-tile constraint blocks a real workload; see Open Questions.

**Backend:** ggml only. No new library dependency. llama.cpp's
`tools/mtmd/clip.cpp` is read as reference material for the SigLIP graph
shape and Gemma 3 specifics (`mm.input_projection.weight`,
`mm.soft_emb_norm.weight`, 27-layer ViT-So400M, 896×896, patch size 14,
4×4 average pooling on the patch grid → `mm_tokens_per_image=256`). We do
not import their code.

---

## Architectural shape

```
src/
├── vision/                          [NEW — concept-named directory]
│   ├── vision_model.h/cpp           Weights + config object (the mmproj)
│   ├── vision_encoder.h/cpp         Forward pass: bitmap → image embeddings
│   └── vision_loader.h/cpp          mmproj.gguf reader
├── graph_inputs/
│   └── image_embeddings_input.h/cpp [NEW typed input — embedding substitution]
├── loader/gguf_loader.cpp           Detect-and-refuse stays; mmproj recognized
├── models/gemma3.cpp                Two new recipe sites (substitution + mask)
├── core/                            One new prefill orchestrator call site
└── cli/chat.cpp                     Chunk list + media markers
```

**Untouched:** `src/state/`, `src/layers/`, `src/sampling/`, `src/metal/`,
`forward_pass_base.h`, all non-Gemma recipes, `feed_tokens`, `DecodePlan`,
the existing KV cache, grammar, sparse decode.

**The architectural claim (C3 made concrete):** vision lives in `src/vision/`
as a **separate subsystem with its own model object and its own graph**,
connected to the text model only at the embedding boundary. Vision tower's
internal attention / MLP / norms stay *inside* `src/vision/`, **not promoted**
to `src/layers/`. The blueprint enumerates five layer-module types; vision
tower is not a sixth — it is a pre-decoder pipeline outside the layer-module
contract.

This is the same architectural separation llama.cpp consciously chose
(`mtmd` outside `libllama`), expressed in our directory conventions.

---

## Data flow (single image, single turn)

1. User message (text + image) → `ChatTemplate::render()` → string with
   media markers. The render signature is **preserved** — `std::string` in,
   `std::string` out.
2. Tokenizer expands markers per Gemma 3:
   `<start_of_image>` + 256 × `<image_soft_token>` + `<end_of_image>`.
3. Prompt-prep splits the token stream into a **chunk list**: text-token-runs
   and image-chunks (each image-chunk references a bitmap).
4. **Vision encoder** runs once per bitmap, in its own ggml context with
   its own graph. Three stages, not one — the plan must distinguish them
   because each has its own weights in mmproj:
   1. SigLIP ViT forward: 896×896 → conv2d patch embed (patch 14) → 27
      transformer layers → **4096 patch tokens** (64×64) at SigLIP hidden
      dim.
   2. **4×4 average pooling** on the patch grid → 256 tokens.
   3. Projection: `mm.input_projection.weight` (linear) →
      `mm.soft_emb_norm.weight` (norm) → 256 vectors at the text-embd dim.

   Lives in `src/vision/`.
5. **Prefill orchestrator** (one new call site in `core/`) builds the text
   prefill graph; `ImageEmbeddingsInput` carries the image embeddings into
   the graph for substitution at placeholder positions.
6. Text prefill runs: at the recipe's **one** substitution site, embedding
   lookup is replaced by gather-from-input for image-token positions;
   text-token positions go through normal embedding lookup. At the recipe's
   **one** mask site, the attention mask is bidirectional within the image
   span (Gemma 3 specific; recipe-bound, not engine-wide).
7. Decoder runs as normal. From the decoder's perspective, the cache holds
   opaque KV; the fact some positions originated from images is invisible
   to KV cache, attention, sparse decode, grammar.

---

## Phasing (risk-ordered; each phase gated on its own deliverable)

Each phase ships self-consistent: code compiles, existing tests stay green,
the phase's deliverable is gated by a test that did not exist before the
phase. This mirrors the project's standing TDD discipline.

### Phase 1 — Loader detect-and-refuse

Closes the originating MedGemma incident **independent** of the multimodal
build. When the loader sees a GGUF / HF config exposing `vision_config`,
`image_token_id`, `mm_tokens_per_image`, or a `*ForConditionalGeneration`
arch string and **no mmproj is provided**, refuse with a CLAUDE.md-shaped
error (field name, expected value, actual value).

- **Deliverable gate:** unit test with a minimal GGUF metadata fixture
  asserting the loader throws with the expected message shape. No model
  file required.
- Detection logic is a small named function — reused by Phase 2 as the
  same metadata, opposite branch (refuse if absent → recognize if present).
- **Not** a stub for multimodal support; it is the refusal, complete in
  itself.

### Phase 2 — Vision encoder, standalone

Build the SigLIP encoder graph in `src/vision/vision_encoder.cpp` using ggml
primitives, plus the average-pool + projection + soft-emb-norm stages.
Load the mmproj GGUF via `src/vision/vision_loader.cpp`. Run one forward
pass on a fixed test bitmap and verify output embeddings against a
reference.

**Reference harness — pivoted from earlier draft.** The earlier "Python
SigLIP comparison" assumed an existing harness; there isn't one. Pivot to
**capturing llama.cpp's `clip.cpp` output as the reference**, once, for a
fixed test image. Same backend (ggml), same weights, eliminates
implementation-vs-implementation noise that a HuggingFace Python reference
would introduce. The reference is a static captured tensor checked into
the test fixtures; the test runs our encoder and compares.

- **Deliverable gate:** bitwise-identical (or token-stable within
  documented ε, matching the `feed_tokens` precedent) output embeddings
  vs. the captured llama.cpp reference for a fixed bitmap input on Gemma 3
  mmproj.
- This is the highest-risk phase: it is the only place we write a brand-new
  transformer-shape from scratch. If the bit-level reference fails, the
  whole plan stalls here, not after wiring is half-done. Front-loading the
  risk is the point.
- **C3 deliverable lands here, but provisional.** With the vision encoder
  standalone and shipping, the answer to *"VisionTower as sixth
  layer-module type or pre-decoder pipeline?"* is concrete for **one
  recipe (gemma3 + SigLIP).** Per CLAUDE.md's cross-family rule, that is
  "presumed gemma3-shaped until a second recipe proves otherwise." With
  Qwen-VL deferred indefinitely, the cross-family validation stays open.
  The blueprint update
  ([`modular-layer-architecture.md`](modular-layer-architecture.md)) must
  ship in Phase 2 **and** must explicitly mark the answer as provisional
  pending a second multimodal recipe. The "pipeline, not layer-module"
  framing is the working hypothesis grounded by Gemma 3; it is not the
  validated cross-family conclusion.

### Phase 3 — `ImageEmbeddingsInput` + gemma3 embedding-substitution site

Add the typed graph input `ImageEmbeddingsInput` in `src/graph_inputs/`.
Wire it into `models/gemma3.cpp`'s `build_prefill_graph` at the **one**
embedding-lookup site: at image-soft-token positions, gather from the typed
input instead of from the embedding table. **One site, parameterized** —
same locality discipline as the `want_logits` head guard.

- **Deliverable gate:** a differential test that feeds pre-computed image
  embeddings + text tokens directly into the gemma3 prefill graph, and
  asserts the resulting logits match a reference (greedy top-1 stable, or
  bitwise — decision made when the test is written).
- No vision encode yet. Phase 3 isolates the recipe-side wiring from the
  vision-encoder correctness already proven in Phase 2.

### Phase 4 — Image-span bidirectional attention mask

Add the mask-builder site in `models/gemma3.cpp` that makes attention
bidirectional within the image span. Recipe-bound. The mask-shape decision
is a recipe property; it does not lift to `ForwardPassBase`.

**Sharp edge flagged during review:** this is harder than "one site" in
isolation. Gemma 3 has 5:1 local/global sliding-window attention; the
bidirectional image-span mask must compose correctly with the
sliding-window mask on both local and global layers. HuggingFace's own
Gemma 3 reference implementation [had a bug](https://github.com/huggingface/transformers/issues/39389)
where the bidirectional mask did not reach attention forward — i.e., the
reference implementations themselves have stumbled on this composition.
Treat Phase 4 as a real correctness phase, not a one-line guard.

- **Deliverable gate:** the Phase 3 differential test extended to assert
  the mask shape is correct under three configurations: (a) text-only
  prefix attends causally with sliding window, (b) image-span tokens
  attend bidirectionally to each other within the span, (c) text tokens
  after the image span attend causally to the entire image span (image
  tokens are "in the past" once the span closes). All three must hold on
  both local and global layers.
- If composition with sliding window proves intractable in one site,
  pause and re-evaluate before adding a second site — multiple mask
  call-sites in one recipe is a locality-of-reasoning violation and
  needs an explicit owner decision, not a quiet drift.

### Phase 5 — Prefill orchestrator

Add the **one** new call site in `src/core/` that, when the chunk list
contains image chunks, runs the vision encoder first, then text prefill
with the encoded embeddings injected via `ImageEmbeddingsInput`. This is
top-level wiring — recipes do not orchestrate vision; the orchestrator
does.

- **Deliverable gate:** end-to-end test, a single-image single-turn prompt
  through the full pipeline. Output compared against a reference (greedy
  top-1 stable, or token-stable — matches the `feed_tokens` precedent).

### Phase 6 — Chat template + chunk list + CLI integration

Add media marker handling to `ChatTemplate` (string interface preserved;
markers go into the rendered string, get expanded by the tokenizer). Add
the chunk-list construction step in `cli/chat.cpp`. Surface a CLI flag for
passing image files.

- **Deliverable gate:** running `chat` with an image file and a text
  prompt produces a coherent response on Gemma 3 4B IT (multimodal
  reference checkpoint).

### Phase 7 — Multi-image, multi-turn, cache interaction

Stress the seam: multiple images in one prompt, an image in a mid-turn of
a multi-turn conversation, image followed by long text generation, image
plus grammar-constrained output. This is where workload-envelope decisions
from C4 land for real — image-count caps or expanded envelope are
implemented and verified here.

**Image-embedding reuse across turns (added during review).** The naive
implementation re-encodes the same image every turn of a multi-turn
conversation — wasted work. llama.cpp's mtmd API exposes
`mtmd_bitmap_set_id()` precisely so the same image can be encoded once and
reused. We adopt the same pattern: each image-chunk in the chunk list
carries a content-derived ID (hash of pixel data, or caller-supplied);
the orchestrator caches encoded embeddings keyed by ID for the session
lifetime. Cache is per-session, evicted on session end. No cross-session
sharing in v1 (security / isolation surface, deferred).

- **Deliverable gate:** a small suite of multi-image / multi-turn scenarios
  passing on Gemma 3 4B IT. KV cache behavior verified — image tokens
  must remain in-cache and unmodified across turns (no eviction surprises;
  recall SnapKV was deleted, so this is a clean check). Image-embedding
  reuse verified — a fixed image referenced twice in the same session
  encodes exactly once (assert via encoder call count, not just timing).

---

## C3 deliverable (the architectural payoff, honestly scoped)

By the end of **Phase 2**, the blueprint
([`modular-layer-architecture.md`](modular-layer-architecture.md)) is
updated with a **provisional** answer to:

> *"Is the vision tower a sixth layer-module type, or a pre-decoder pipeline
> outside the layer-module contract?"*

Working hypothesis: **pre-decoder pipeline.** Justification: the concrete
shape of `src/vision/` as it ships in Phase 2 — its own model object, its
own graph, its own ggml context, connected only at the embedding
boundary. Layer modules participate in the residual stream of the
decoder; vision tower terminates at the embedding boundary. Different
contract, different module category.

**Honest caveat (from review).** This answer is validated against exactly
**one recipe** (gemma3 + SigLIP). Per CLAUDE.md's cross-family rule, an
interface validated on one family is "presumed family-shaped until a
second proves otherwise." Qwen-VL is deferred indefinitely (eval doc),
which means the cross-family validation of the pipeline-vs-layer-module
answer stays open indefinitely. The blueprint update must explicitly
record:

1. The provisional answer (pipeline) and its Gemma 3 grounding.
2. That this is "presumed Gemma-shaped" until a second multimodal recipe
   proves otherwise.
3. What a second recipe would have to demonstrate to promote the
   provisional answer to validated (concretely: a second vision-bearing
   recipe in a different family with a different vision encoder shape
   integrating cleanly without changes to the pipeline contract).

If Phase 2 ships without this blueprint update — including the
provisional caveat — the work has produced a feature and forfeited the
architectural justification it was approved on. That is the C3 contract.

---

## Workload envelope (C4 made concrete)

The C4 decision (image-count caps vs envelope expansion) is made before
Phase 5 starts coding, not discovered during Phase 7.

**Default proposal:** cap images at **2 per session**, **single-tile each**
(Pan & Scan scoped out per the top-of-doc constraint). Arithmetic: 2 × 256
= 512 image tokens = ~13% of the 4K envelope, leaving ≥3.5K for text —
fits the stated OM-style workload. Sessions wanting more images, or
images that the single-tile constraint cannot accommodate, are refused
fail-loud rather than silently truncated.

**Why single-tile matters for the arithmetic.** The earlier draft of this
section computed "2 × 256 = 512" assuming one image = 256 tokens. With
Pan & Scan, a non-square or high-resolution image is *not* 256 tokens —
it can be 256 × N tiles. The single-tile constraint is what makes the
"2 images" cap arithmetically meaningful. If Pan & Scan is ever
reintroduced, this whole arithmetic recomputes.

This proposal is reviewed and accepted-or-amended at the Phase 5 design
moment; it is not silently inherited.

---

## What this plan does NOT do

- It does not touch `feed_tokens`, the `feed_tokens` token-stable/bitwise
  fork, or speculative decoding. Multimodal does not interact with
  `feed_tokens` (vision is prefill-only).
- It does not promote anything from `src/vision/` into `src/layers/` or
  `ForwardPassBase`. C2 is enforced by directory placement.
- It does not add audio support, even within Gemma. Phase 7 closes the
  vision question; audio is a separate evaluation.
- It does not anticipate Qwen-VL. The reopen criteria for Qwen-VL are in
  the eval doc; nothing in this plan should make that future reopening
  harder.

---

## Open questions for follow-up (not gating this plan)

- **mmproj distribution:** do we recommend users obtain mmproj GGUFs from
  third-party converters (ggml-org HF account), or do we ship our own
  `convert_hf_to_gguf.py`-equivalent? Decision deferred to Phase 6.
- **Vision-tower quantization:** K-quants apply cleanly to ViT-shaped
  tensors in ggml, but we have not yet verified the SigLIP-specific
  weights quantize without quality regression. Phase 2 should include a
  small probe; final answer can land in Phase 7.
- **Image preprocessing on the host:** image decoding (JPEG/PNG → raw RGB
  tensor) is a host-side operation that has no ggml component. Need a
  small image-loading dependency or a hand-rolled minimal decoder. Phase 6.
  Note: with Pan & Scan scoped out, preprocessing is plain decode +
  resize-to-fit (letterbox or center-crop to 896×896), not the multi-tile
  algorithm.
- **Pan & Scan reintroduction.** Scoped out in v1. Reopen criterion: a
  named workload demonstrates the single-tile constraint is blocking
  (e.g., documents with fine text the resize-to-fit destroys, or
  landscape product photos that lose critical detail). At that point: Pan
  & Scan is its own design phase, not a slip into the existing
  preprocessing — the C4 envelope arithmetic recomputes from scratch.
- **Gemma 3n (MobileNetV5) and Gemma 4V vision.** Scoped out in v1.
  Different vision encoders, structurally distinct from SigLIP. Reopen
  criterion: a named user requires the variant *and* the modular layer
  refactor has answered whether multiple vision encoders share the
  pre-decoder-pipeline boundary cleanly (C3 follow-up).
