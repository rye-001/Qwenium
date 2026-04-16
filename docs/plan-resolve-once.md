# Resolve-Once Optimization Plan

**Goal:** Eliminate redundant recursive grammar expansions in `consume_token()` — the single largest remaining bottleneck in grammar-constrained decoding.  
**Impact:** `get_valid_tokens()` currently averages **110-130ms/call**, of which ~110ms is `consume_token` calls on LITERAL candidates. This optimization should reduce that to **~1-5ms**.

---

## The Problem (from profiling)

```
[gvt-detail] stacks=2  lit_states=1  cc_states=1  lit_groups=1
             candidates=424  consume_calls=424
             classify+cc avg=0.82ms   ← SOLVED (trie + bitset)
             literal avg=110ms        ← THIS IS THE BOTTLENECK
```

When the grammar expects a short literal like `" "` (space), the trie returns ~424 space-prefixed tokens as candidates. For **each** candidate, `consume_token` does:

1. Compares chars with the literal (fast — 1 char for `" "`)
2. Calls `advance_after_terminal()` (enters the expensive path)
3. `advance_after_terminal` calls `resolve_and_consume()`
4. `resolve_and_consume` calls `explore_alternatives()` which recursively expands the **next grammar element** (e.g., `function_name` → 5 alternatives, `identifier` → 9 alternatives, `expression` → 5 alternatives...)

**The same grammar expansion runs 424 times.** Every candidate consumes `" "` identically — they only differ in remaining characters (offset 1+). The `explore_alternatives` call at step 4 produces the exact same set of resolved terminal states every time.

At ~260μs per `consume_token` call × 424 candidates = **110ms**.

---

## The Fix: Pre-Resolve, Then Match

### Core Insight

For a given LITERAL state `(pos, char_idx, continuations)`, all candidate tokens that **fully consume** the literal reach the **same** set of follow-on grammar states. The grammar expansion after the literal depends only on the grammar structure, not on the token being consumed.

We can:
1. Resolve the follow-on states **once** per LITERAL group
2. Match remaining token characters against those pre-resolved states

### Before (Current)

```
For each candidate token:
    consume_token(token, pos, char_idx, conts)
        → if token fully consumes literal:
            advance_after_terminal(token, offset, pos, conts)     ← EXPENSIVE
                → resolve_and_consume(token, offset, next_pos, conts)
                    → explore_alternatives(...)                    ← REDUNDANT ×424
                        → for each resolved terminal:
                            consume_token(token, offset, ...)      ← recursive
```

### After (Proposed)

```
// Phase 1: Pre-resolve (ONCE per literal group)
resolved_terminals = explore_alternatives(next_pos_after_literal, conts)

// Phase 2: Classify candidates (FAST)
For each candidate token:
    if token.length <= remaining_literal:
        // Partial consumption — just compare chars, push state (no recursion)
        check_partial(token, literal, char_idx)
    else:
        // Full consumption + overshoot — match remaining chars against pre-resolved terminals
        remaining = token.substr(literal_remaining_length)
        for each resolved terminal:
            consume_token(remaining, 0, terminal.pos, terminal.char_idx, terminal.conts)
```

The key difference: `explore_alternatives` runs **once** instead of 424 times.

---

## Implementation Plan

### Phase 1: Split consume_token Logic

**File:** `grammar_vocab.cpp`

Refactor `consume_token` to separate two concerns:
- **Character matching** (cheap): compare token chars with literal chars
- **Grammar resolution** (expensive): what comes after the literal

New helper functions:

```cpp
// Try to match token_str[offset:] against literal[char_idx:].
// Returns: {matched_length, is_full_literal_consumed}
// Does NOT do grammar resolution — just character comparison.
static MatchResult match_literal(const std::string& token_str, size_t offset,
                                  const std::string& literal, size_t char_idx);

// Pre-resolve: given a terminal position, compute all reachable grammar states
// after that terminal is fully consumed. This is the expensive part that we
// want to do only once.
static void resolve_after_terminal(const Impl* impl,
                                    const Elem* terminal_pos,
                                    const std::vector<const Elem*>& conts,
                                    std::vector<GrammarState>& resolved);
```

### Phase 2: Modify the LITERAL Loop in get_valid_tokens

**File:** `grammar_vocab.cpp` → the grouped trie path

```cpp
for (const auto& [key, states] : literal_groups) {
    // ... trie query gives candidates ...

    for (const auto* st : states) {
        const std::string& lit = literal_values[st->pos->value];
        size_t remaining_len = lit.size() - st->char_idx;

        // PRE-RESOLVE: what grammar states follow after this literal?
        // (Expensive, but done ONCE per state, not per candidate)
        std::vector<GrammarState> after_terminals;
        resolve_after_terminal(impl, st->pos, st->continuations, after_terminals);

        for (int32_t tok_id : candidates) {
            if (valid_mask[tok_id]) continue;
            const std::string& tok = vocab[tok_id];

            // Fast path: partial consumption (token shorter than remaining literal)
            if (tok.size() <= remaining_len) {
                if (tok == lit.substr(st->char_idx, tok.size())) {
                    valid_mask[tok_id] = true;  // partial match, valid
                }
                continue;
            }

            // Full consumption: token consumes all of remaining literal
            if (lit.compare(st->char_idx, remaining_len, tok, 0, remaining_len) != 0)
                continue;  // doesn't match the literal

            // Now match the REMAINING token chars (tok[remaining_len:])
            // against the pre-resolved terminals. No grammar expansion needed.
            size_t tok_offset = remaining_len;
            for (const auto& resolved : after_terminals) {
                std::vector<GrammarState> result;
                consume_token(impl, tok, tok_offset,
                              resolved.pos, resolved.char_idx,
                              resolved.continuations, result);
                if (!result.empty()) {
                    valid_mask[tok_id] = true;
                    break;
                }
            }
        }
    }
}
```

### Phase 3: Handle Edge Cases

1. **Multi-element tokens:** A token like `" getAccounts("` spans `" "` → `"getAccounts"` → `"("`. After consuming the first literal, the remaining `"getAccounts("` must still recurse through grammar elements. The pre-resolved terminals handle the first follow-on element; further elements recurse normally. The win is eliminating the first (most expensive) expansion.

2. **Token exactly matches literal:** `tok.size() == remaining_len` — no remaining chars. The token is valid iff the grammar can advance (accepting state or more elements). Check `after_terminals` for END states.

3. **CHAR_CLASS follow-ons:** If the element after the literal is a CHAR_CLASS (not a LITERAL), the pre-resolved terminals will include CHAR_CLASS states. The remaining-chars matching handles this naturally through `consume_token` on the resolved state.

4. **Continuations with OP_STAR/OP_OPT:** The resolve_after_terminal handles these through `explore_alternatives` which already expands optional/star groups correctly.

---

## Testing Strategy

### Unit Tests

1. **Equivalence test:** Compare results of resolve-once path vs. current per-candidate path for every grammar position in the existing test suite. Must produce identical valid token sets.

2. **Edge case tests:**
   - Single-char literal `" "` with 400+ space-prefixed candidates
   - Multi-element token spanning 3+ grammar elements
   - Token that exactly matches remaining literal (no overshoot)
   - CHAR_CLASS immediately following a LITERAL
   - Nested groups with OP_STAR after literal

### Performance Validation

Before/after profiling on the same test prompts:
- `get_valid_tokens` avg should drop from ~120ms to ~5ms
- `literal avg` in `[gvt-detail]` should drop from ~110ms to ~1-3ms
- Grammar-constrained `sample` time should approach forward-pass time

---

## Performance Expectations

| Scenario | Current | After |
|----------|---------|-------|
| `" "` literal, 424 candidates | 424 × resolve_and_consume = 110ms | 1 × resolve + 424 × char compare = ~2ms |
| `"getAccounts"` literal, 5 candidates | 5 × resolve = ~1.3ms | 1 × resolve + 5 × match = ~0.3ms |
| Total `get_valid_tokens` | ~120ms | ~3-5ms |
| Total `sample` per token | ~170ms | ~10-15ms |
| Grammar overhead vs forward pass | ~1.1x | ~0.1x |

---

## File Inventory

| File | Action |
|------|--------|
| `src/sampling/grammar_vocab.cpp` | Refactor: add `resolve_after_terminal()`, `match_literal()`, modify LITERAL loop |
| `src/sampling/grammar_vocab.h` | No changes needed (internal refactor) |
| `tests/unit/test_token_trie.cpp` | Add resolve-once equivalence tests |

---

## Risk Assessment

- **Correctness risk: MEDIUM.** The grammar state machine is complex. Pre-resolving must handle all the same edge cases as the current recursive path (ALT, END, RULE_REF, OPEN/CLOSE, OP_STAR/PLUS/OPT). Extensive equivalence testing is critical.
- **Performance risk: LOW.** The change only affects the LITERAL path in get_valid_tokens. CHAR_CLASS and accept_token are unaffected. Brute-force fallback preserved.
- **Complexity risk: LOW.** The refactoring adds ~50 lines and removes redundant work. The existing `explore_alternatives` and `consume_token` functions are reused — we're just calling them in a different order.

---

## Open Questions

1. **Cache resolved terminals across calls?** If the grammar state doesn't change between calls (same literal, same char_idx, same continuations), the resolved terminals are identical. A small LRU cache keyed on `(pos, char_idx, conts_hash)` could avoid re-resolving. Worth profiling after the basic fix.

2. **Batch trie candidates by consumption type?** Instead of per-candidate branching, pre-sort candidates into "partial" (shorter than remaining literal) and "overshoot" (longer) groups. Partial group needs only char comparison; overshoot group shares the pre-resolved terminals. This enables SIMD-friendly batch processing.

3. **Recursive overshoot depth?** If a token spans 3+ grammar elements (e.g., `" getAccounts("`), the resolve-once optimization helps for the first element boundary but not subsequent ones. In practice, most tokens span at most 2 elements, so one level of pre-resolution captures most of the win.

---

## Addendum: Skip-When-Easy (Cherry-on-Top)

**Idea:** At grammar positions where >99% of vocab is valid (negated CHAR_CLASS like `[^"]`), skip grammar enforcement entirely and let the model sample freely. Validate after with `accept_token` — if the rare invalid token is picked, do one constrained re-sample.

**Why it's NOT an alternative to resolve-once:** The 110ms bottleneck is in LITERAL states where <1% of vocab is valid (e.g., `" "` followed by specific identifiers). You can't skip enforcement there — the model would pick invalid tokens constantly. Skip-when-easy only helps CHAR_CLASS negated states, which are already 0.82ms after the precomputed-bucket optimization.

**But it's still worth doing** as a ~5-line follow-up after resolve-once:

```cpp
// In get_valid_tokens, for CHAR_CLASS states:
if (cc.second && cc.first.size() <= 5) {
    // Negated class with few exclusions — >99% of vocab valid.
    // Skip enforcement: return "all valid" and let accept_token catch the rare miss.
    // Saves 0.8ms per call + eliminates mask_logits cost for this step.
    return {};  // empty = "skip grammar this step" (needs sampler cooperation)
}
```

**Saves:** ~0.8ms (CHAR_CLASS) + ~1.7ms (mask_logits) = ~2.5ms per call at those positions. Minor vs. the 110ms resolve-once target, but free perf once the plumbing exists. Could also serve as a foundation for a more aggressive speculative grammar approach in the future.
