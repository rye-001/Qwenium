# Consolidation defect ledger — 2026-09-02

Findings from driving real traffic through the server (qemmi-ops planner calls,
`/v1/extract`, concurrency and overload) with `--token-log` on. Method: log
defects, do not fix opportunistically; measure before believing; run a control
before attributing a cause.

**Status key.** Entries marked FIXED landed in the same session. Entries marked
OPEN are recorded, not scheduled. Two entries (the D3 correction and D5) record
things *I* got wrong and are kept deliberately — the reasoning that produced them
is the useful part.

**Fixed this session:** D8 (lens left slot 0 dirty, corrupting the next request),
the missing try/catch on the stateless prefill path, slot cleanup on prefill
failure, and the `--flash-attn`-after-prefill ordering defect.

**Still open, ranked:** D1 (`<think>` leaks into `content`/`text` — the channel
filter is Gemma-4-only), the blanket HTTP 413 mapping for every `error_message`,
D9 (two servers can both bind `:8080` via SO_REUSEPORT), D6 (stale
`/health` `queue_depth`), D7 (interleaved stdout under load), D2 (thought channel
spends the whole `max_tokens`).

**Not a defect:** D3's headline was wrong — `/v1/completions` reaches
`--prefix-cache` today via `system_prompt` (measured ~10x). The gap is in the
client, which does not send it.

---

# Consolidation defect ledger — opened 2026-09-02

Rule: log, don't fix opportunistically. Fix at the end, ranked.
Columns: id | severity | what | trigger | status

| id | sev | defect | how it surfaced | status |
|---|---|---|---|---|
| D1 | med | Qwen `<think>…</think>` scaffold returned VERBATIM in `content` (chat) and `text` (completions). `ChannelFilter` only knows Gemma 4's `<\|channel>` framing and is "inert for non-Gemma-4 models" (http_server.cpp:1276). Every client must strip it itself; no `reasoning_content` split. | first 3 manual requests, Qwen 3.6 | OPEN |
| D2 | med | Thought channel spends the whole `max_tokens`: chat req with max_tokens=30 returned ONLY `<think>…` and finish_reason `"length"` — zero visible answer, clean status. Known (§12), reproduced on the first real chat request. | manual req C | OPEN (known) |
| D3 | HIGH | **Neither warm-KV mechanism can serve the actual client workload.** Measured corpus: prompt:generated = **5.6:1**, and consecutive qemmi-ops planner prompts share **798/930 leading tokens (86%)**; a synthetic agent-shaped triple shares ~99.5%. Yet `--chat-prefix-cache` logs `MISS ... prefilled 1938 tokens cold` on every one, and wall time is identical with and without it (5.51/5.44/5.39s vs 5.60/5.47/5.45s). Cause is NOT the documented thinking-model excuse (§6 "~0 hits on thinking models — scaffold stripped on re-render"): this is `/v1/completions` with a raw prompt, no chat re-render, byte-identical prefix. Real cause: `warm_prefix_pick` requires `std::equal(res.begin(), res.end(), tokens.begin())` — the whole RESIDENT stream must be a prefix of the NEW request (append-only). An agent workload is fixed-prefix + **varying tail**, so it can never hit. The other mechanism, `--prefix-cache`, keys off a chat **system message** (`render_system_prefix`), and `/v1/completions` has none — so a plain OpenAI completions client cannot reach it at all. | corpus analysis then A/B measurement | OPEN |
| D4 | note | Why D3 is structural on this model, not an oversight: reusing the *longest common* prefix needs a KV **rewind** to the divergence point, and §9 says rewind is safe only for pure-attention models — Qwen 3.6 is a hybrid whose recurrent state has overwrite semantics and cannot rewind. So LCP-reuse is blocked on hybrids but WOULD be available on a Gemma recipe. Any fix must be recipe-gated. | derived from D3 | OPEN |

## D3 measurement — `--prefix-cache` WORKS; the workload just can't reach it

Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL, `/v1/chat/completions`, ~1920-token system
message + a short varying user turn, `max_tokens=8`, `temperature=0`.
Wall time per request:

| run | req0 | req1 | req2 | req3 |
|---|---:|---:|---:|---:|
| control, no `--prefix-cache` | 5.58s | 5.51s | 5.53s | 5.50s |
| `--prefix-cache` (cold dir) | 5.80s (MISS, stores) | **0.60s** | **0.60s** | **0.53s** |
| `--prefix-cache`, after a server RESTART | **0.66s** | 0.54s | 0.53s | 0.53s |

* **~9.5x on warm requests** (5.51 → 0.58 mean). Server log confirms the
  mechanism, not a timing artefact: `[prefix-cache] HIT: skipped prefill of 1920
  system tokens`.
* The control is **flat** (5.58/5.51/5.53/5.50), so none of the win is warm-up.
* Cold cost of populating the cache is ~0.25s, paid once.
* **It survives a restart**: a freshly started server HITS on its very first
  request. The "across restarts" claim in §6 holds.

So the mechanism is sound and the payoff on an agent workload is large. The gap
is purely reachability: `cacheable_prefix_text` is populated only on the chat
route, from the system message (`http_server.cpp:1249`, `render_system_prefix`),
and `/v1/completions` — what qemmi-ops uses — has no system message.

## D3 CORRECTION (2026-09-02) — the reachability half of D3 was WRONG

`/v1/completions` has accepted a top-level **`system_prompt`** field all along
(http_server.cpp ~1275). It renders `render_system_user_turn`, sets
`skip_template`, and populates `cacheable_prefix_text` when `--prefix-cache` is
wired. **Measured: 5.49s cold → 0.55s / 0.54s warm**, `[prefix-cache] HIT`.

So "a plain OpenAI completions client cannot reach `--prefix-cache` at all" was
false. I had grepped `cacheable_prefix_text =`, found hits at 1249 and 1446, and
attributed BOTH to the chat route; 1249 is the completions route's system_prompt
branch. The engine gap does not exist.

**What survives from D3:** `--chat-prefix-cache` genuinely cannot serve this
shape (append-only vs varying tail) — that measurement stands. And qemmi-ops
embeds its fixed block in `prompt` and sends no `system_prompt`, so it gets no
reuse today. But that is a CLIENT gap, not an engine one.

## D5 — the `cache_prefix` field I added is not transparent (my defect)

It works (5.61s → 0.47s, ~11.9x) but silently changes what the model sees:
with the field, `run_cached_text_prefill` encodes `req.prompt` RAW, while the
normal completions path wraps it via `set_tokenize`'s `wrap_user_turn`. Same
request, different prompt — visible as `prompt_tokens` 1938 (no field) vs 1925
(with field). An optimization flag that changes the prompt is exactly the class
of defect this session has been hunting. Making it transparent needs wrap-aware
prefix construction (the analog of `render_system_prefix`), for a marginal gain
over `system_prompt`, which is transparent by construction.

## Multi-slot concurrency pass (2026-09-02) — mostly GREEN

Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL, `-s 4 -c 8192`, `/v1/completions`, temperature 0.

**Correctness — PASS.** 4 concurrent requests with distinguishable answers:
peak `active_slots=4` (real batching), and **no cross-talk** — every answer
contained its own tag and no foreign tag. Serial 8.3s vs concurrent 6.60s = 1.26x
at B=4, consistent with §1's 1.59x ceiling for this recipe.

**Determinism — PASS, and it confirms §11 exactly.** Concurrent run 2 vs run 3:
**identical on all 4**. Concurrent vs serial: 3 of 4 identical, BRAVO diverges at
char 200/217 (deep in the `<think>` span, near the max_tokens cut). So batched
decode is deterministic run-to-run at a fixed batch shape, and the batch-shape
fork is real but reproducible — "byte-replay claims hold at B=1", as §11 says.
*Nuance worth recording:* §11 calls the batching gate "token-stable", unqualified.
At 60 generated tokens it broke on 1 of 4 prompts. Token-stability degrades with
generation length; the shipped gates check far shorter runs.

**Token log under batching — PASS.** 4 records, 4 distinct slot ids (0,1,2,3),
4 distinct generated streams, correct counts. No cross-slot contamination.

**Overload — PASS.** 12 concurrent on 4 slots: all HTTP 200, peak
`active_slots=4` (respects `-s`), peak `queue_depth=5`, 4.0s total. Queueing works.

**Cancellation mid-generation — PASS.** Raw socket, hung up after 3s:
`[server-stop] slot=0 reason=cancelled where=cancelled gen=86` — slot released,
server kept serving, and the token log honestly recorded
`finish_reason:"cancelled"` (the endpoint would have flattened it; see D2's
sibling, the known `chat_finish_reason` issue).

| id | sev | defect | how it surfaced | status |
|---|---|---|---|---|
| D6 | low | `/health`'s `queue_depth` can go stale and stay wrong: observed **stuck at 1 for 15s+ with an empty queue, 0 active slots, and a subsequent request served normally**. It self-healed only when later traffic forced the pop loop to run again (back to 0 after 8 concurrent). Cause: `stats_.queue_depth` is written only after a successful `try_pop` inside `assign_requests_to_slots` (inference_server.h:579) and after a push (:504), so any removal path that bypasses that loop leaves the mirror stale. **Admission is NOT affected** — the 503 gate uses the real `request_queue_.size()` (:498) — so this is a monitoring lie, not a DoS. Appeared immediately after a cancellation. | cancellation test | OPEN |
| D7 | low | Server stdout interleaves between the inference thread and HTTP threads, producing corrupted log lines, e.g. `[server-stop] slot=1 reason=stop=========Qwenium Response===========`. Unsynchronized `std::cout`. Makes logs unreliable exactly when they matter (under load). | concurrency runs | OPEN |

## Attention-lens under load (2026-09-02)

Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL, `-s 4`, `--attention-lens`.

**PASS — exclusivity works.** 3 concurrent `/v1/extract` calls serialize behind
the model lock (2.22 / 4.39 / 6.50s, ~3x the 2.1s solo time) and every report is
**byte-identical to the serial baseline**. No corruption, no interleaving.

**PASS — the documented warning is over-cautious, in the safe direction.** §6 says
"do not drive concurrent OpenAI traffic on slot 0 while extracting". Measured: an
extract run concurrently with 4 `/v1/completions` produced a report
**byte-identical** to the same extract run alone. The model lock does cover it.

**PASS — extraction quality.** All 4 fields correct (`customer`,
`order_date`, `quantity`, `total`), `badge:"grounded"`, `body_mass` 0.857 vs the
0.538 threshold, 8 citations each with byte spans landing on the right document
regions. Absent-by-omission works: an unmentioned concept returns
`value:null, badge:"absent", tier:null, citations:[]`.

| id | sev | defect | how it surfaced | status |
|---|---|---|---|---|
| D8 | HIGH (receipts) | **Running `/v1/extract` changes the output of a later, identical `/v1/completions`.** Discriminating control, one server, greedy, B=1 throughout: baseline vs baseline after an intervening DIFFERENT completion → **identical**; baseline vs baseline after an intervening EXTRACT → **differs**, diverging at char 87/177 ("...wavelengths of sunlight, such as blue..." vs "...-wavelength blue light from the sun..."). Reproduced on two server instances; 4 repeated completions with no extract are 4/4 identical, so this is not run-to-run noise. This contradicts §11's byte-reproducible-greedy-decode-at-B=1 claim and §12's "the tap seam is byte-inert when disarmed" — the unit gate `TapOffByteIdentical` proves the *graph* is identical at one step in one process, which is a weaker property than what a server needs. **Mechanism NOT established.** Leading hypothesis, by precedent: the tapped graph marks extra outputs, so galloc re-plans across alternating shapes on the shared scheduler — the same class as `server-image-multirequest-bug.md`. Other candidates: DecodePolicy or per-step arming state not fully reset after `set_attention_taps({})`. Needs a dedicated probe. | lens load test | OPEN |

## D8 probe (2026-09-02) — engine EXONERATED, cause is the server layer

**Shape of the defect, established first.** Only the FIRST completion after an
extract differs; every later one returns to baseline. Two distinct outputs,
perfectly reproducible: `base0, base1, afterX1_b, afterX1_c, afterX3_b` all hash
`8bf7ffd8`; `afterX1_a, afterX3_a` both hash `d3254183`. So the perturbation is
**transient, exactly one request deep** — which rules out a persistent galloc
plan change.

**Controls run:**
* intervening ordinary completion (short) → next completion **identical**
* intervening LONG completion (~800-token prompt, 200 decode tokens) → next
  completion **identical**. So it is lens-specific, NOT sequence-length driven.
  (This control was missing from the original D8 entry; the first version of that
  entry compared only against a short completion.)

**Three engine-level gates added to `tests/unit/test_forward_pass_base.cpp`, all
PASS on qwen35moe — the engine does not reproduce D8:**
1. `TapArmedThenDisarmedIsInertForNextDecode` — arm, tapped pass on a different
   prompt, disarm, re-run the original decode: logits byte-identical. **The tap
   seam is not the cause.**
2. `ExtraUntappedPassIsInertForNextDecode` — control, same sequence untapped.
3. `LensShapedUntappedPassIsInertForNextDecode` — a 120-token prefill plus 24
   single-slot decode steps with growing n_kv (the lens driver's graph shape),
   untapped, on the shared scheduler: still byte-identical. **Graph-shape
   alternation / galloc re-planning is not the cause either**, which was the
   leading hypothesis by precedent (server-image-multirequest-bug.md).

**Hypotheses refuted along the way:** (a) stale KV left in engine slot 0 —
refuted, `simple_kv_cache::clear_slot` resets `positions[slot]=0` and
`inference_server.h:916` clears before any pos-0 prefill; (b) persistent galloc
re-plan — refuted by the one-deep shape; (c) the tap seam — refuted by gate 1;
(d) unusual graph shape — refuted by gate 3.

**Still open: the exact site.** It is in the server integration layer, in state
the lens driver touches that a normal request does not — `extract_lens_json`
drives `fp` directly (its own `clear_slot`/`set_cache_pos`/`run_prefill`/decode
loop) while the `InferenceServer`'s own `Slot` bookkeeping for slot 0 never
learns the slot was used. Remaining candidates: per-slot sampler state
(`slot_samplers_`), the request/slot lifecycle around `model_mutex_`, or
`ForwardPassBase` per-step arming that `clear_slot` does not cover. Next
instrument: log engine `get_cache_pos(0)` and slot bookkeeping either side of an
extract — the mismatch should be visible directly rather than inferred.

## D8 — MECHANISM FOUND AND FIXED (2026-09-02)

One env-gated log line at the boundary settled it in a single run, after four
hypotheses had died to guesswork. Trace (`QINF_D8_TRACE=1`, since removed):

    normal request:
      prefill:enter slot=0 start_pos=0 n_tokens=24 cache_pos=0     <- clean
      clear_slot    slot=0 cache_pos_before=56                     <- cleared on RELEASE
    after an extract:
      extract:enter slot=0 cache_pos=0
      extract:exit  slot=0 cache_pos=138        <- lens leaves 138 on slot 0
      prefill:enter slot=0 start_pos=0 n_tokens=24 cache_pos=138   <- NOT CLEARED
      prefill:after_advance          cache_pos=162                 <- 138+24, not 24

**Cause.** `run_lens_tapped_decode` drives slot 0 directly and leaves it at
`cache_pos = prompt+generated`. The `InferenceServer` slot lifecycle never learns
slot 0 was used, so no release fires for it. The next request prefills at
`start_pos=0` but `advance_cache()` **adds**, so it decodes with `n_kv=162` for a
24-token sequence and attends over 138 rows of stale lens KV. One request deep
because that request's own release then clears the slot (`cache_pos_before=201`).

**Fix.** `fp->clear_slot(0)` beside the existing `set_attention_taps({})` in
`server_lens.cpp` — symmetric with the disarm, restoring the intent its own
comment already stated ("leave the engine byte-inert for the next request").

**Verified.** All five completions around two extracts now hash `8bf7ffd8`
(previously `d3254183` for the first post-extract one); `extract:exit
cache_pos=0`; extract reports unchanged and stable. `ctest -j1`: **840/840**.

**Why the engine gates could not have caught it** (and why they are still worth
keeping): `decode_once` in the test calls `clear_slot`/`set_cache_pos` itself, so
the test始 always starts from a clean slot — structurally unable to observe a
leaked one. The server's `run_prefill` does not clear. Same shape as every other
instrument defect this session.

**Refuted en route, for the record:** the tap seam (gate 1), graph-shape/galloc
re-planning (gate 3), sequence length (long-completion control), and — wrongly,
by me — the stale-slot hypothesis itself, dismissed on `inference_server.h:916`'s
`if (prefill_pos == 0) clear_slot_(...)`. The trace shows that line is a
different path; on the normal path the slot is clean only because the PREVIOUS
request's release cleared it.

| id | sev | defect | how it surfaced | status |
|---|---|---|---|---|
| D9 | med | **Two server instances can both bind :8080** (`SO_REUSEPORT`), so a second start silently splits traffic instead of failing with "address in use". Observed live: two PIDs in LISTEN on 8080 at once, which confounded one D8 run until spotted. An operator restarting without stopping the old process gets a silent split brain with two divergent KV states. | D8 probe | OPEN |
