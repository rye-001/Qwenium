# Multimodal Support — Architecture Evaluation

**Status:** Evaluation only (no implementation). Output of the architecture
pressure test for multimodal support, triggered by the MedGemma 1.5 incident.

**Recommendation: CONDITIONAL SUPPORT — Gemma family only, scope (a)
vision-as-input, used deliberately as the forcing function for an open
architectural question.** Multimodal does not clear the "credible production
target" bar today (the workload envelope is text-heavy), but it does clear
the "forces architecture clarification" bar — *if* the work is scoped
tightly. The four binding constraints below are not aspirations or
preferences; they are the conditions under which this decision holds. If any
one cannot be met, the recommendation reverts to PUNT.

The original evaluation recommended PUNT on two grounds: (1) cross-family
check fails and (2) no named production target. (2) still stands, and is the
reason this is not "support multimodal." (1) is now treated as a *design
constraint*, not a blocker — the work is scoped so that the cross-family rule
is satisfied by recipe-level opt-in rather than by a forward-pass-wide
abstraction.

---

## Binding constraints (load-bearing — read these before any design work)

These four constraints are the decision. They are not the "nice to have"
section. If the implementation drifts from any of them, this evaluation
expires and the work pauses for re-evaluation.

### C1. Scope is Gemma family only. Qwen-VL is consciously deferred.

Qwen-VL uses M-RoPE (multi-axis rotary positional embedding for 2D image
patches); Gemma 3 vision treats image tokens as a 1D sequence under standard
RoPE. The image-injection *pattern* converges across the families (encoder +
projection + placeholder substitution) but the *position-encoding contract*
diverges. Adding Qwen-VL would force M-RoPE into the RoPE module — a
structural change driven by a feature no current text-only recipe needs.

Therefore: **building Gemma-Vision is not a step toward Qwen-VL.** Those are
separate decisions, made under separate conditions. The Qwen-VL question
reopens only when M-RoPE has a non-bending answer (e.g., the modular
refactor lands "RoPE-1D and RoPE-MA as separate modules sharing a base"),
not as a continuation of this work.

### C2. Multimodal is a Gemma-recipe capability, not a forward-pass abstraction.

Text recipes (all Qwen, all Gemma text-only) remain **untouched**. The
forward-pass interface gains no `is_multimodal` axis. Image embeddings enter
through a Gemma-recipe-specific embedding-stage hook; the decoder graph
topology is unchanged. KV cache, attention, RoPE, sparse decode, grammar —
all see opaque token positions, exactly as today.

This keeps us inside the CLAUDE.md cross-family rule. We are not claiming a
multimodal abstraction across families — we are claiming a Gemma-family
capability that opts in at the recipe level. If the implementation finds
itself wanting to lift multimodal logic into `ForwardPassBase`, that is the
signal that this constraint is being violated; pause and reconsider.

### C3. The work answers one architectural question. If it doesn't, it wasted the forcing function.

The blueprint
([`modular-layer-architecture.md`](modular-layer-architecture.md)) enumerates
five layer-module types: Attention, SSM, DeltaNet, MoE, DenseFFN. A vision
tower is structurally different — its own architecture (ViT/SigLIP), runs
once at prefill, produces embeddings consumed by the decoder, doesn't fit
the "build into the per-layer transformer block" mold.

The forcing-function payoff of doing Gemma-Vision now is that it **forces
the answer** to: *"Is VisionTower a sixth layer-module type, or a
pre-decoder pipeline outside the layer-module contract?"* The deliverable
of this work includes a blueprint update with that answer, grounded in the
shipped Gemma 3 vision integration. If the work ships without that update,
we built a feature and forfeited the architectural payoff.

Working hypothesis (not yet decided): **pre-decoder pipeline, not a layer-module
type.** A vision tower doesn't participate in the residual stream the way
layer modules do; it terminates at the embedding boundary. But this is the
question the work has to answer, not assume.

### C4. Workload envelope: explicit shift OR image-count caps. Pick one, write it down.

The current envelope (≤10 slots, ≤4K context, 12 GB-class models on Apple
Silicon) does not accommodate uncapped multimodal use: 4 images × 256
tokens = 1024 tokens = 25% of a 4K context. SigLIP-base adds ~0.8–1.2 GB of
resident memory (fits 32 GB unified memory comfortably; quantizable).

Two options, no third:

- **(i) Cap images per session** (e.g., ≤2 images/session for multimodal
  recipes). Envelope unchanged. Text budget remains 3.5K+.
- **(ii) Expand envelope** to ≤8K context for multimodal sessions, leaving
  text-only sessions at 4K. The envelope contract grows from one row to
  two; both are documented.

Pick at design time, before coding. Do not drift the envelope silently — the
envelope is the contract recent simplifications (TQ + SnapKV deletion) were
made against, and changing it without a stated decision invalidates those
calls retroactively.

---

## Sequencing relative to existing work

This work fits **after** the stabilization backlog's load-bearing items,
not in place of them:

1. **Resolve the `feed_tokens` token-stable vs bitwise fork** — owner
   decision, blocks the third `feed_tokens` consumer.
2. **Push the `feed_tokens` seam through Gemma text** — the cross-family
   falsifier per CLAUDE.md. If the seam bends here, the interface is
   defective and that's the work, not multimodal.
3. **Loader detect-and-refuse for multimodal-only checkpoints** — the
   actual MedGemma lesson. One-day warmup task. Closes the originating
   incident and exercises the "is this checkpoint multimodal" detection
   logic the multimodal work will reuse anyway. **Do this first.**
4. **Gemma-Vision** — this evaluation.
5. **Speculative decoding** — next stated capability after multimodal lands
   or punts again.

(1) and (2) are not deprioritized by multimodal — they remain the gating
items for `feed_tokens`-dependent consumers. Multimodal does not touch
`feed_tokens`; it can proceed in parallel with their resolution, but the
freeze on third-consumer additions to `feed_tokens` still holds.

---

## Per-axis evaluation (the analysis that produced the constraints)

The findings below are the substrate of C1–C4. Read them when revisiting the
decision, not when implementing.

### 1. Module-boundary effect

Adding a vision encoder + projection + interleave stage *before* the
transformer decoder is conceptually clean — the decoder consumes embeddings
opaquely, regardless of source. But "VisionTower" is structurally a sixth
candidate layer-module type (ViT/SigLIP attention, conv patch embedding,
layer norm variants), not a parameter on an existing module.

The blueprint enumerates five and has not stably extracted them yet.
Introducing a sixth candidate at the same time is the source of C3 — the
work must answer whether it actually *is* a sixth type or whether it lives
outside the layer-module contract entirely.

### 2. Scope

- **(a)** Vision-as-input only — image embeddings projected into text
  embedding space at input, decoder unchanged. **This is the chosen scope.**
- **(b)** Vision + interleaved attention — forks attention. Rejected on
  architecture grounds.
- **(c)** Multimodal generation — out of scope for target families.

### 3. Cross-family check

Qwen-VL M-RoPE vs Gemma 3 1D RoPE is the structural divergence. The
constraint C1 (Gemma-only, Qwen-VL consciously deferred) is the response.
This is not a "we'll figure out Qwen-VL later" — it is a "Qwen-VL is a
separate decision under separate conditions."

### 4. State management (scope (a), Gemma)

- **KV cache:** Image tokens append uniformly (256/image). Unchanged. ✓
- **Position encoding:** 1D RoPE applies normally. ✓
- **Sliding-window attention** (Gemma 3): Image tokens count toward the
  window normally. ✓
- **Sparse decode / spec decode / grammar:** Image embeddings only enter at
  prefill; decode-time subsystems see opaque positions. ✓

State-side is clean within C1's scope.

### 5. Tokenization & chat template

`ChatTemplate::render()` returns `std::string`. Preserve the signature: emit
`<image_soft_token>` placeholders, replace with image-embedding positions
in the embedding stage. Localizes the change to one place; does not ripple
to `chat.cpp` / `complete.cpp` / server.

The structured-content API alternative (`render → vector<TemplateToken>`)
is deferred — if a second multimodal forcing function ever requires it, the
signature change can be made then. Don't paint into a corner by adding
template features that assume string-only, but don't preemptively refactor
either.

### 6. Workload envelope

Quantified in C4. SigLIP fits memory; image tokens shift context budget;
choice between caps and envelope expansion is a design-time decision.

### 7. Production-target credibility

No named production user. This is the bar this evaluation does *not* clear.
The decision rests entirely on the forcing-function bar (C3). If C3 is not
actually delivered — i.e., the blueprint question goes unanswered — the
work was unjustified.

### 8. Effort sizing

Scope (a), Gemma-only (C1 satisfied): bucket **M** (1–2 months). Vision
tower module, projection, embedding-stage placeholder substitution,
chat-template handling, Gemma 3 / Gemma 4 recipe wiring, testing, blueprint
update for C3.

---

## Adjacent decision (now sequenced ahead): loader detect-and-refuse

When the GGUF metadata or upstream HF config exposes a `vision_config`
block, or `image_token_id` / `mm_tokens_per_image`-class fields, or an arch
string implying `*ForConditionalGeneration`, the loader refuses with a
CLAUDE.md-shaped message naming the field, the expected value (absent /
null), and the actual value (present / multimodal-only).

Two reasons this leads:

1. Closes the originating MedGemma incident — a user loading a multimodal
   checkpoint should fail loud at load, not at the third token of degenerate
   output.
2. Exercises the "is this checkpoint multimodal" detection logic the
   multimodal work itself will reuse — same metadata, opposite branch.

Small, contained, separate commit/PR. Do this before the Gemma-Vision design
work begins.

---

## Reopen criteria (Qwen-VL specifically — Gemma-Vision is now in scope)

The Qwen-VL multimodal question reopens only when:

1. **M-RoPE has a non-bending answer.** Either the modular refactor lands
   "RoPE-1D and RoPE-MA as separate modules sharing a base" (the
   architecturally clean answer), or Qwen ships a future VL model that
   converges with Gemma's 1D-positioning approach (unlikely; M-RoPE is a
   stated design choice).
2. **A named production target requires Qwen-VL specifically.** Not "VLM in
   general" — Qwen-VL by name, with workload bounded against the envelope.

Both should hold. The first alone is "we *could* support Qwen-VL"; the
second is what makes it worth doing.

---

## Open architectural questions (the ones C3 must close)

C3 makes these load-bearing for this work — they cannot be left open at
ship time:

- **Is RoPE a single module or a family?** Forced when Qwen-VL eventually
  reopens. Not in scope for Gemma-Vision per C1, but flagged here so the
  Gemma-Vision design doesn't inadvertently make the future answer harder.
- **VisionTower: sixth layer-module type, or pre-decoder pipeline outside
  the layer-module contract?** This is the C3 question. Working hypothesis:
  pre-decoder pipeline. Must be answered, with justification, in the
  blueprint update that ships with the Gemma-Vision work.
- **Chat-template signature:** string-with-placeholders today (C5
  finding); structured-content API deferred. Recorded so the deferral is
  conscious.

---

## Notes on llama.cpp

- **Take:** vision encoder is a separate graph from the decoder, sharing
  only the embedding-table boundary. Confirms scope (a) is the right scope.
- **Take:** chat-template placeholder substitution pattern. Keeps call sites
  unchanged.
- **Do NOT take:** the `llava` / `llava_next` / `minicpmv` / `internvl` /
  `mtmd` proliferation of per-VLM special cases. That is exactly the
  path-zoo failure mode CLAUDE.md names. One Gemma-Vision module that hosts
  Gemma 3 and Gemma 4 vision **without bending** — not parallel modules per
  variant.

---

## Decision log

- **2026-05-27 (initial):** Evaluation recommended PUNT. Two grounds:
  cross-family check fails (Qwen-VL M-RoPE vs Gemma 3 1D RoPE), no named
  production target.
- **2026-05-27 (revised):** Owner revisited. Stabilization backlog is in a
  state where bandwidth is available. Recommendation revised to
  **CONDITIONAL SUPPORT, Gemma family only, scope (a)**, with four binding
  constraints (C1–C4) that make the decision architecturally clean. The
  cross-family finding is reclassified from blocker to design constraint
  (C1, C2). The no-named-target finding still stands and is the reason
  the work is justified on the forcing-function bar (C3) rather than on
  production demand. Loader detect-and-refuse is sequenced ahead as a
  one-day warmup that closes the originating incident independent of the
  larger work.
