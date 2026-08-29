# `src/engine/` — the loaded model, and one step over it

Renamed from `src/core/` on 2026-08-29. The old name was concept-free and had
accreted three unrelated jobs; the GGUF value bag and the mmap wrapper went to
`loader/`, the persistence services to `session/`, and what remained is this.

| File | Owns |
|---|---|
| `model.{h,cpp}` | The load path and its product: metadata, tokenizer vocabulary, weights bound per layer, the backend buffer. Releases the file mapping after the weight copy — load-bearing for steady-state RSS, see the header. |
| `decode_plan.{h,cpp}` | The resolved, immutable decision for ONE decode step. Pure logic, no ggml, so its truth table is enumerable in a unit test. |
| `decode_step.{h,cpp}` | Executes one decode step for one slot: grammar peek → forward pass → sample → accept, plus forced-token elision. |
| `decode_graph_cache.{h,cpp}` | The opt-in persistent decode graph (`--persistent-graph`): build+allocate once per KV-width bucket on a dedicated scheduler, reuse across steps. |
| `multimodal_prefill.{h,cpp}` | The single call site bridging the vision subsystem and a text recipe. |

Two of these (`decode_step`, `multimodal_prefill`) are compiled directly into
their consumers rather than into the `qinf-engine` library, because they depend
on `qinf-models` / `qinf-vision` which sit above it. Directory is the concept;
the CMake target is the layering.

This file is a signpost only. The as-built map — dataflows, seams, invariants,
and the honest ledger of soft spots — is [`docs/architecture.md`](../../docs/architecture.md),
and it is the thing to keep current.
