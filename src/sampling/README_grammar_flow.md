# Qwen3 Inference: Grammar & Sampling Architecture Flow

This document outlines the end-to-end execution flow of the Token-Agnostic Grammar Engine (Vocab-Scan) implemented in the Qwen3 inference system.

## 1. Setup & Pre-Decoding (`src/cli/chat.cpp`)
At startup, `chat.cpp` extracts the entire vocabulary from the `Tokenizer`. It decodes every single token ID into its actual UTF-8 string representation (e.g., Token ID `713` → string `" const"`). This is stored in a continuous vector (`decoded_vocab`).

**Architectural Note:** There are no offline, pre-compiled lookup tables mapping GBNF keywords to specific token IDs.

## 2. Model Forward Pass (`src/cli/chat.cpp`)
The application tokenizes the user's input, runs the neural network forward pass (`forward_pass->build_prefill_graph`, etc.), and produces an array of `logits` representing the raw probability distribution for the next ~152k tokens in the vocabulary.

## 3. The Sampling Pipeline (`src/sampling/sampling.cpp`)
Instead of directly picking the highest probability, `chat.cpp` hands the raw `logits` and the `decoded_vocab` to the `Sampler`.
The Sampler immediately enforces the structural rules by calling `apply_grammar_constraints(logits, decoded_vocab)`:

1.  **Ask the Grammar:** It calls `grammar_->get_valid_tokens(decoded_vocab)` to get a definitive list of syntactically legal token IDs.
2.  **Mask Invalid Tokens:** Any token ID *not* in the valid list has its logit overwritten to `-infinity`.
3.  **Handle EOS:** If the grammar state machine reports `is_accepting_state()`, the sampler explicitly looks up the EOS and `<|im_end|>` tokens dynamically from the `decoded_vocab` and unmasks them, allowing the model to naturally choose when to finish the turn.
4.  **Math:** Finally, standard Temperature, Top-K, and Repetition Penalty algorithms are applied only to the surviving, structurally valid logits to select the final `next_token_id`.

## 4. The Live Vocab-Scan (`src/sampling/grammar_vocab.cpp`)
When the sampler asks `get_valid_tokens` for the legal IDs, the engine performs the "Option B" real-time character evaluation:

1.  **State Inspection:** The engine looks at its current position in the GBNF tree (e.g., expecting `"const"`, currently at index `0`).
2.  **Full Vocabulary Loop:** It iterates through the entire `decoded_vocab` (all ~152k strings).
3.  **Fast Prefix Filter:** It performs an O(1) check: does the token's first character match the next character the grammar expects?
4.  **Deep Consumption:** If the first character matches, it calls `consume_token()` to evaluate the entire string of the token against the active grammar rules, character by character.
5.  **Token Agnosticism:** If a token string perfectly satisfies the current grammatical expectation (e.g., `"con"`, `"c"`, or `" const"`), its Token ID is marked as valid.

## 5. Advancement (`src/cli/chat.cpp`)
Once the Sampler has chosen the `next_token_id`, it is printed to the screen and appended to the KV cache context. 

Finally, `chat.cpp` calls `grammar->accept_token(next_token_id, decoded_vocab)`. The grammar engine officially updates its internal stack, consuming the characters of that token and advancing its pointers (e.g., if `"con"` was chosen, it advances its state to expect `"st"` on the next loop).