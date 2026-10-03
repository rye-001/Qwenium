# Repo split: Qwenium / lens-server / qemmi-lens (2026-10-03)

User idea: move every lens topic out of the engine repo. The engine keeps a
general HTTP server. Three repos: **Qwenium** (engine), **lens-server**,
**qemmi-lens** (client). Analysed only; nothing changed.

## 1. What the code shows (measured 2026-10-03)

- **Size.** `src/server/` is the largest directory in `src/` (12,185 lines).
  - Lens: `server_lens` 5,541 lines; `image_verdict` 744.
  - About 1,350 of the 3,719 lines in `http_server.cpp` (by section; the five
    lens routes alone are 621).
  - 5 of 27 server flags.
  - `test_server_lens.cpp` 3,056 lines; `tests/perf/attn_provenance.cpp`
    (hunts + drift gate) 20,292.
  - About 61 of 136 docs (by file name).
  - 8 of the last 10 PRs (#35–#42).
- **Coupling.** The lens is a peer of engine internals, not an API client.
  About 1,870 lines of `server_lens.cpp` are drivers (`run_lens_*`, the tapped
  decode). They call 26 `ForwardPassBase` methods, raw `ggml_backend_sched_*`
  and snapshot internals. `server_lens.h` itself is pure std.
- **Churn across the seam.** Engine files outside `src/server/` changed by lens
  PRs: #39 943 lines / 19 files, #40 986 / 20, #41 224 / 6, #42 504 / 16. On
  2026-10-02 the "where" probe found two engine faults (decode mask, IMROPE).
- **Client.** qemmi-lens calls only the five lens routes, never an OpenAI
  route. Its contract is the versioned `qemmi-lens/vN` format plus the flag
  name `--attention-lens` in its error hints.
- **Visibility.** The Qwenium repo is public and MIT (GitHub API 200 without
  login). Qemmi-lens is not public (404).

## 2. Pros

1. **Lean general server.** About 1,350 lines and 5 flags leave. So do the
   lens-only locks: lens ⊥ `--speculative`, lens ⊥ quantized `--kv-type`,
   per-model flash in prefill, the lens layer cut inside the general model-load
   path, no OpenAI traffic on slot 0 during an extraction.
2. **The engine repo reads as an engine.** Lens policy leaves §5, §6 and §11 of
   `architecture.md`; the harness and lens docs leave too.
3. **Product data sits with the product.** Lens-format versions (v2–v4) change
   through engine PRs today; the only consumer is qemmi-lens. Calibration rows
   are Qwen-only by measurement; the engine is cross-family by rule.
4. **License and visibility become a choice** for new lens work. Limit: PRs
   #26–#42 are already public.
5. **The build enforces the dependency direction**, but only after the seam is
   narrowed. Through a submodule every engine header stays reachable.

## 3. Cons

1. **Most lens features become two-repo changes** (4 of the last 5 lens PRs
   changed 6–20 engine files). Null-question calibration adds a driver pass,
   exactly on the seam.
2. **Engine internals become a public API** while still being refactored.
   Today an interface change and its lens effect land in one PR, checked by
   `test_server_lens` (175) and LENSDUMP (95 reports byte-identical).
3. **The decision gates leave the engine repo** (see §4).
4. **lens-server is a full server, not a thin layer.** Every verb needs the
   model in-process (extract decodes, the image verdict decodes "where" boxes).
   It needs model load, tokenizer and templates, image preparation, snapshots,
   HTTP and error mapping, the lock discipline. It cannot hand generation to a
   separate `qwenium-server` on the same model: the server copies weights (no
   `--mmap-weights`), so a second process doubles weight memory.
5. **Build work.** `CMAKE_SOURCE_DIR` 20× in 10 CMakeLists,
   `PROJECT_SOURCE_DIR` 0× — not subproject-safe. No `install(EXPORT)`, so the
   route is submodule + `add_subdirectory`. Patched ggml (b10964) comes through
   the engine build.
6. **Claude's memory is per working directory.** Lens memories stay with this
   repo unless moved.

Not claimed (no evidence): faster builds or tests, deployment independence,
team scaling.

Problems to fix either way: the image verdict is a qemmi-lens product endpoint
living in `src/server/`; the lens drivers run ggml's scheduler directly.

## 4. Lens rules: two kinds

The "receipts constraints" in `architecture.md` §11.

| Kind | Example | Checked by | After a split |
|---|---|---|---|
| Mechanical: rows exist and are correct | `kq_soft.<il>` on every model type, real softmax rows, tap off changes nothing, flash never with a decode tap | 14 `ForwardPassTapTest` cases × 5 model types (3 Qwen, 2 Gemma), hard errors in `mark_attention_taps` / `DecodePolicy` | stays in the engine |
| Decision: an engine change must not move a lens result | flash prefill, F32 KV default, no nondeterministic kernels, single-slot determinism, MoE warm cache | `lens_drift_gate.sh` (BANDDRIFT), LENSDUMP — need calibrated heads, corpus, margins | lens data leaves |

Worked example: flash attention in prefill passed every engine test (decode
stays materialized, dense models token-identical). Only the drift gate could
judge it: 9B drift 0.00084 vs margin 0.00126 PASS; 27B 0.00054 vs 0.00099
PASS; 35B 0.0242 vs 0.0237 **FAIL** (one extraction in 15 changes). Hence the
per-model `flash_prefill_ok`.

The decision gates are manual today too: a GPU run of a few minutes per arm,
not in ctest.

## 5. The rule: client gates (user, 2026-10-03)

The engine serves many clients; the lens is one. Clients pay for the engine's
evolution, so they have a say in it. **An engine change that can affect a
client lands only when that client's gate passes, or the client records an
explicit OK.** Otherwise the engine is free to change. The reverse dependency
is intended.

- The engine owns the list of client-affecting change classes (§11): attention
  numerics, default KV type, prefill shape, kernels, ggml bumps, MoE routing.
- Each client owns its gate and its data. Lens: drift gate (chunk, flash,
  warm) + LENSDUMP; heads, corpus and margins stay in lens-server (private).
  One entry point, e.g. `gate.sh <engine checkout>`.
- The engine side keeps only the runner and the client list, and runs every
  registered gate before such a PR lands. `flash_prefill_ok` is the existing
  example of a recorded OK.

Three conditions:
1. Gates check decisions, not bytes — a byte gate would freeze the engine.
2. One command per client — the engine runs it without knowing the client.
3. No gate, no say — a client without a gate accepts what the engine does.

## 6. Recommendation

- **Step A, in this repo, reversible.** Replace the 26 methods and raw ggml
  calls with a small readout API in `engine/`: prefill or decode with a readout
  spec (taps, layer cut, attention mode, routing capture/replay, logits), plus
  slot snapshot/restore. Give the lens and the image verdict their own target
  and binary; `qwenium-server` stops linking them. Gates: LENSDUMP
  byte-identical, `test_server_lens`, server tests unchanged. A new `src/lens/`
  needs approval.
- **Step B, mechanical.** Fix the CMake paths; move the target, tests,
  harness, docs and memories to lens-server; engine as a pinned submodule; add
  the client-gate runner (§5).
- **When to do Step B:** after two lens features in a row land with no engine
  change outside the readout API.
- **Order:** run the null-question calibration probe first (harness, like the
  earlier hunts).
