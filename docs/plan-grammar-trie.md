# Grammar/Token Trie Plan

**Goal:** Replace the O(vocab_size) brute-force scan in `get_valid_tokens()` with a trie-based lookup that only visits matching tokens.  
**Impact:** Grammar-constrained decoding overhead measured at 2-4x (e.g. 35s→13s without grammar). This is currently the single largest bottleneck.

---

## The Problem (30-second summary)

Every decode step, `GrammarVocab::get_valid_tokens()` iterates **all ~152K vocab tokens**, checking each against the current grammar state via string comparison. For LITERAL states it does:

1. O(1) first-char pre-filter
2. Full `consume_token()` string walk for matches

This runs every single token generated. With 152K tokens and potentially multiple active grammar states, it dominates sampling latency.

---

## How the Trie Fixes It

A trie indexes all 152K vocab token strings by their characters. Instead of scanning every token, we walk the trie from the grammar's expected prefix and collect only tokens that match.

```
Grammar expects: "pri" (mid-literal "price")

Brute force:  scan 152K tokens, ~10 match     → O(vocab_size)
Trie walk:    p → r → i → collect subtree      → O(matching_tokens)
```

Typical matching set is 10-100 tokens, so this is 1000-10000x fewer lookups.

---

## Architecture

### Data Structure

```
TokenTrie
├── root TrieNode
│   ├── children: array<TrieNode*, 256>  (or unordered_map for sparse)
│   └── token_ids: vector<int32_t>       (tokens that ARE this exact string)
│
└── Built once from vocab at startup, immutable thereafter
```

Each node at depth `d` represents a character prefix of length `d`. Leaf/interior nodes store `token_ids` for tokens whose decoded string exactly matches that prefix path.

**Memory estimate:** ~152K tokens, average ~4 chars each ≈ ~600K nodes. At ~40 bytes/node (map + vector) ≈ ~24 MB. Acceptable.

### Key Operations

```cpp
class TokenTrie {
public:
    // Build trie from vocabulary (called once at sampler init)
    void build(const std::vector<std::string>& vocab);

    // Collect all token IDs whose string starts with `prefix`
    // Used for LITERAL mid-match: prefix = remaining literal chars
    void collect_by_prefix(const std::string& prefix, size_t offset,
                           std::vector<int32_t>& out) const;

    // Collect all token IDs whose first char is in the set
    // Used for CHAR_CLASS states
    void collect_by_char_class(const std::set<char>& chars, bool negated,
                               std::vector<int32_t>& out) const;

    // Collect all token IDs whose string starts with exactly `c`
    // Fast path for single-char prefix filter
    void collect_by_first_char(char c, std::vector<int32_t>& out) const;
};
```

---

## Integration Plan

### Phase 1: Build the Trie

**File:** `src/sampling/token-trie.h`, `src/sampling/token-trie.cpp`

1. Define `TrieNode` and `TokenTrie` classes
2. `build()` inserts each `vocab[token_id]` into the trie character by character
3. Each node accumulates token IDs for tokens that terminate at that node
4. Unit test: build from a small vocab, verify prefix collection matches brute force

### Phase 2: LITERAL Fast Path

**File:** Modify `grammar_vocab.cpp` → `get_valid_tokens()`

Current LITERAL path:
```cpp
// For each state with LITERAL at char_idx:
const std::string& lit = impl->literal_values[pos->value];
char expected_first = lit[char_idx];
for (size_t i = 0; i < vocab.size(); ++i) {
    if (vocab[i][0] == expected_first) {
        // consume_token() full validation
    }
}
```

New LITERAL path:
```cpp
// Get remaining literal suffix from char_idx onward
std::string_view remaining = std::string_view(lit).substr(char_idx);

// Collect candidate tokens whose string is a prefix of `remaining`
// OR tokens that `remaining` is a prefix of (token extends past literal end)
std::vector<int32_t> candidates;
trie_->collect_literal_candidates(remaining, candidates);

// Only run consume_token() on the small candidate set
for (int32_t tok_id : candidates) {
    consume_token(impl, vocab[tok_id], 0, pos, char_idx, conts, result);
}
```

**Subtlety:** A token can:
- Match a prefix of the remaining literal (e.g. token `"pr"` against literal `"price"`)
- Match the entire remaining literal and continue into the next grammar element
- Extend past the literal (e.g. token `"price_"` when literal is `"price"`)

The trie must collect **all tokens on the path** from root to the remaining literal's end, **plus** all tokens in the subtree below that end. This handles all three cases.

### Phase 3: CHAR_CLASS Fast Path

Current: scan full vocab checking `cc.first.count(vocab[i][0])`.

New: for each char in the class set, collect tokens from `trie_->children[c]` subtree. For negated classes, iterate all root children *except* those in the set.

### Phase 4: Multi-State Optimization

When multiple grammar states are active (alternatives, recursion), merge their trie queries:
- Collect candidate sets per state
- Union into a single `valid_set`
- Avoids redundant trie walks for overlapping prefixes

---

## Handling Edge Cases

### Partial Literal Consumption
Token `"pr"` against literal `"price"` → consumes 2 chars, grammar state advances `char_idx` to 2. Next step, trie query starts from `remaining = "ice"`. Works naturally.

### Multi-Literal Tokens
Token `":{" ` could span a LITERAL `":"` and start of the next LITERAL `"{"`. The trie collects `":"` and `":{"` as candidates. `consume_token()` handles the grammar state transitions — the trie just narrows the candidate set.

### Empty/Special Tokens
Skip empty strings and special tokens (`<|endoftext|>`, `<|im_end|>`) during trie build. Handle EOS separately in the sampler (already done).

---

## Testing Strategy

### Unit Tests (`tests/unit/test_token_trie.cpp`)

1. **Build correctness:** Small vocab, verify all tokens retrievable by exact prefix
2. **Prefix collection:** Query `"pri"` → get tokens starting with `"pri"`, tokens that are prefixes of `"pri"` (`"p"`, `"pr"`), and tokens extending past (`"price"`, `"print"`)
3. **Char class collection:** `[a-z]` collects all tokens starting with lowercase
4. **Negated class:** `[^0-9]` excludes digit-starting tokens
5. **Empty token handling:** Empty strings don't crash the trie
6. **Equivalence test:** For a real vocab + grammar, verify trie-based `get_valid_tokens()` returns identical results to brute-force

### Integration Tests

1. Run existing grammar tests (order-management, JSON schema) — output must be identical
2. Benchmark: measure tokens/sec with and without trie on the grammar-constrained paths

---

## Performance Expectations

| Scenario | Brute Force | Trie |
|----------|------------|------|
| LITERAL mid-match | 152K checks | ~10-50 candidates |
| CHAR_CLASS `[0-9]` | 152K checks | ~5K candidates (digits) |
| CHAR_CLASS `[^"]` | 152K checks | ~147K candidates (negated — less gain) |
| Multiple states (3) | 3 × 152K | 3 × ~50 candidates |

Negated char classes are the weak spot — most tokens match. Consider: for negated classes with large match sets, fall back to the brute-force scan with an inverted check (mask out the excluded tokens). The trie helps most for the common case: LITERAL matching during structured output generation (JSON keys, enum values, etc.).

---

## File Inventory

| File | Action |
|------|--------|
| `src/sampling/token-trie.h` | New — TrieNode + TokenTrie class |
| `src/sampling/token-trie.cpp` | New — build + collect implementations |
| `src/sampling/grammar_vocab.h` | Add `TokenTrie*` member |
| `src/sampling/grammar_vocab.cpp` | Replace vocab scan loops with trie queries |
| `src/sampling/sampling.cpp` | Pass trie to grammar (or build in sampler init) |
| `tests/unit/test_token_trie.cpp` | New — trie unit tests |
| `tests/CMakeLists.txt` | Add trie test target |

---

## Open Questions

1. **Trie node representation:** `array<256>` (fast, ~10 MB overhead) vs `unordered_map` (compact, slower)? Profile both. Given 152K tokens and ~600K nodes, array might be worth the memory.

2. **Mutual-recursion bug:** The existing `get_valid_tokens()` has a documented state leak bug with mutual recursion. Fix it before or after the trie migration? Probably before — the trie shouldn't change correctness, only performance.

3. **Thread safety:** If the server uses one trie across threads, it must be immutable after build (it is — the trie is read-only after construction).
