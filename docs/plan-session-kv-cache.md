# Plan: Session-Level KV Cache Reuse

## Goal

Eliminate redundant prefill on multi-turn HTTP conversations. Turn N+1 should prefill
only the new user message, reusing the KV state of all prior turns.

## Current State (as of 2026-04-24)

`src/server/http_server.cpp` already has:
- `MAX_SLOTS = 10` KV slots, per-slot `cache_pos` tracking.
- Slot 0 reserved as `TEMPLATE_SLOT` for a single system-prompt prefix cache
  (hash-keyed, cloned into working slot via `clone_slot`).
- Primitives: `clear_slot`, `clone_slot(from, to, len)`, `get_cache_pos`,
  `build_prefill_graph(tokens, start_pos, slot_id)`, `advance_cache`.

What is missing: a session layer on top. `/v1/completions` is stateless — every
request picks a fresh working slot and re-prefills the full history.

## Design

### 1. Session identity

Add `session_id` (opaque string, client-supplied) to the request body. Absent →
stateless behavior (current code path, unchanged).

### 2. SessionManager

New component, owned by `QweniumServerIntegration`. Maps `session_id → SessionEntry`:

```
struct SessionEntry {
  int slot_id;
  int cache_pos;                    // authoritative; may diverge from forward_pass if we snapshot
  std::vector<int32_t> token_history;  // for validation / re-prefill on eviction
  size_t system_prompt_hash;        // which prefix this session was built on
  std::chrono::steady_clock::time_point last_used;
};
```

Operations:
- `acquire(session_id, system_prompt) -> slot_id, start_pos`
  - Hit: bump `last_used`, return existing `slot_id` and `cache_pos` as `start_pos`.
  - Miss: allocate a slot (LRU evict if needed), clone from template slot, return.
- `commit(session_id, new_tokens_len)` — advance `cache_pos` after prefill+decode.
- `release(session_id)` — explicit close (optional; LRU handles the rest).

### 3. Slot allocation & eviction

- Slot 0 pinned as `TEMPLATE_SLOT` (never evicted).
- Slots 1..N-1 are session slots, LRU-evicted when a new session needs one.
- Eviction calls `clear_slot(slot_id)` and drops the `SessionEntry`.
- Future: multi-template support (radix tree of prefixes) — out of scope for v1.

### 4. Request flow change

In `/v1/completions`:
1. Parse `session_id` (optional) and `system_prompt`.
2. If `session_id` present:
   - `slot_id, start_pos = sessions.acquire(session_id, system_prompt)`.
   - Tokenize only the new user turn (not full history).
   - Submit inference request with `slot_id` and `start_pos`.
3. After decode completes: `sessions.commit(session_id, prefill_len + generated_len)`.

`InferenceRequest` needs a `session_id` field and plumbing through to the
scheduler so the right slot is used (today the slot is picked inside
`InferenceServer`; this needs an override path for pinned sessions).

### 5. Thread safety

`SessionManager` guarded by its own mutex — do not reuse `model_mutex_` (would
serialize HTTP handlers against inference). The session mutex is only held for
map lookup / LRU bookkeeping, not across the inference call.

## Interaction with existing features

- **System-prompt cache**: unchanged. `SessionEntry.system_prompt_hash` records
  which template the session was forked from; if a session's system prompt
  changes mid-conversation, invalidate the session (rare; log a warning).
- **SnapKV**: removed (outside workload envelope). No interaction.
- **DeltaNet / SSM recurrent state (Qwen3.5, Qwen3.6)**: recurrent state has
  overwrite semantics. A session slot must snapshot the recurrent state at
  each turn boundary, not just the KV cache. `ForwardPassBase` needs a
  `snapshot_recurrent_state(slot_id)` / `restore_recurrent_state(slot_id)`
  pair, or an equivalent slot-copy primitive that handles both cache types.
  **This is the load-bearing constraint for the hybrid models — do not ship
  session caching for qwen35/qwen35moe without it.**
- **Speculative decoding**: orthogonal; session layer sits above.

## Phasing

1. **Phase 1 — pure-transformer only (Qwen3 / Qwen2.5).**
   SessionManager + LRU + request plumbing. Gate by architecture: reject
   `session_id` on qwen35/qwen35moe with a clear error.
2. **Phase 2 — recurrent state snapshot.**
   Add snapshot/restore to `ForwardPassBase`, implement for DeltaNet and SSM.
   Enable sessions for hybrid architectures.
3. **Phase 3 — multi-prefix template cache.**
   Replace single `TEMPLATE_SLOT` with a radix tree of shared prefixes (à la
   SGLang). Only worth it if multiple distinct system prompts are in active use.
## Non-goals (v1)

- Cross-process session persistence (disk-backed cache).
- Session migration between server instances.
- Automatic session_id generation / cookies — client supplies the ID.
