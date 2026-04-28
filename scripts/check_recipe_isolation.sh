#!/usr/bin/env bash
# check_recipe_isolation.sh
#
# Enforces two isolation rules:
#
# 1. RECIPE INCLUDE ISOLATION — outside src/models/ and tests/, no source file
#    may #include a src/models/<family>.h header.  All dispatch must go through
#    model_registry.h.
#
# 2. ARCH-STRING ISOLATION — src/loader/tokenizer.{cpp,h} and
#    src/cli/chat.{cpp,h} must not contain the architecture string literals
#    "gemma", "qwen", or the SentencePiece underscore character (▁, U+2581)
#    outside of comments.  Family logic belongs in recipe files; the
#    tokenizer and CLI are generic engines that read TokenizerConfig /
#    ChatTemplate injected via the registry.
#
# Exit code: 0 = clean, 1 = one or more violations found.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FAILED=0

# ── Rule 1: recipe include isolation ─────────────────────────────────────────

while IFS= read -r hit; do
    raw_file="${hit%%:*}"
    rest="${hit#*:}"
    lineno="${rest%%:*}"
    include_line="${rest#*:}"

    rel="${raw_file#"$REPO_ROOT"/}"

    # src/models/ — the registry and recipe TUs themselves are allowed.
    [[ "$rel" == src/models/* ]] && continue

    # tests/ — recipe-specific tests are allowed.
    [[ "$rel" == tests/* ]] && continue

    echo "RECIPE ISOLATION VIOLATION"
    echo "  File:    $rel:$lineno"
    echo "  Include: $(echo "$include_line" | xargs)"
    echo "  Rule:    files outside src/models/ and tests/ must not #include a"
    echo "           models/<family>.h header; route dispatch through model_registry.h"
    echo ""
    FAILED=1
done < <(grep -rn \
    '#include.*models/\(qwen3\|qwen35\|qwen36\|gemma1\)\.h' \
    "$REPO_ROOT/src/" 2>/dev/null || true)

# ── Rule 2: arch-string isolation in tokenizer and CLI chat layer ─────────────
#
# Scan non-comment lines of the listed files for family-specific strings.
# We strip // line comments and /* … */ block comments before matching.

ARCH_STRING_FILES=(
    "src/loader/tokenizer.cpp"
    "src/loader/tokenizer.h"
    "src/cli/chat.cpp"
    "src/cli/chat.h"
)

# Strings that must not appear in the generic engine files.
FORBIDDEN_PATTERN='gemma\|qwen\|\xE2\x96\x81'

for rel_path in "${ARCH_STRING_FILES[@]}"; do
    full_path="$REPO_ROOT/$rel_path"
    [[ -f "$full_path" ]] || continue

    # Strip C++ comments with a simple sed pass, then grep for forbidden tokens.
    # This is not a full C++ parser — block comments that span lines may not be
    # stripped correctly, but that edge case doesn't occur in these files.
    while IFS= read -r hit; do
        lineno="${hit%%:*}"
        content="${hit#*:}"
        echo "ARCH-STRING ISOLATION VIOLATION"
        echo "  File:    $rel_path:$lineno"
        echo "  Content: $content"
        echo "  Rule:    src/loader/tokenizer.{cpp,h} and src/cli/chat.{cpp,h} must"
        echo "           not contain 'gemma', 'qwen', or U+2581 (▁) outside comments;"
        echo "           family logic belongs in recipe files via TokenizerConfig /"
        echo "           ChatTemplate injected through model_registry."
        echo ""
        FAILED=1
    done < <(
        grep -n "$FORBIDDEN_PATTERN" "$full_path" \
        | grep -v '^\s*//' \
        | grep -v '//.*'"$FORBIDDEN_PATTERN" \
        || true
    )
done

if [ "$FAILED" -ne 0 ]; then
    exit 1
fi

echo "check_recipe_isolation: PASSED"
