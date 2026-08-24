#!/usr/bin/env bash
#
# apply-all.sh — apply every patches/*.patch in lexical order to the ggml
# source tree. Invoked by CMake's FetchContent PATCH_COMMAND from the
# fetched ggml source directory (cwd).
#
# Three states this has to tell apart, and only three:
#
#   pristine tree     → the patch applies. Apply it.
#   already patched   → the patch does not apply, but its own added marker
#                       lines are all present. Skip it.
#   source moved      → the patch neither applies nor is present. FAIL LOUD.
#
# The third case is the one that matters. It is what a ggml pin bump looks
# like before patches/ has been re-ported, and skipping it silently yields an
# engine quietly missing its fused DeltaNet ops — a silent fallback at a module
# boundary, which CLAUDE.md forbids.
#
# Why marker-presence and not `git apply --reverse --check`: the patches are a
# sequential series, and a later patch inserts text directly adjacent to an
# earlier one's additions. That rewrites the earlier patch's trailing context,
# so 0002 does not reverse-apply cleanly once 0004 has landed — even though it
# is fully applied. `git apply` is atomic per patch, so a patch is either
# wholly present or wholly absent; checking a few distinctive added lines per
# touched file is therefore sound as well as fast.
#
# Args:
#   $1  absolute path to the qwen-inference patches/ directory
#
# Exit codes:
#   0  success (every patch is applied — freshly, or already present)
#   1  hard failure: a patch neither applies nor is already present (context
#      mismatch after a ggml bump), or apply failed after --check passed

set -euo pipefail

PATCH_DIR="${1:?usage: apply-all.sh <patches-dir>}"
shopt -s nullglob

# Emit up to MARKERS_PER_FILE distinctive added lines per touched file, as
# "<path>\t<line>". Trivial lines (braces, blanks, short fragments) are skipped
# so the markers actually identify this patch.
#
# Only PURE-INSERTION hunks yield markers. A hunk that also removes lines is a
# modification, and modified lines are not stable identifiers: 0002 rewrites
# the GGML_OP_COUNT static_assert to 102, then 0004 rewrites it again to 103,
# so 0002's version of that line is absent from a fully-patched tree even
# though 0002 is applied. Hunks are buffered so a trailing "-" still
# disqualifies the "+" lines above it.
MARKERS_PER_FILE=3
patch_markers() {
    awk -v want="$MARKERS_PER_FILE" '
        function flush_hunk(   i, line, t) {
            if (!dirty) {
                for (i = 0; i < nadd; i++) {
                    if (n >= want) break
                    line = add[i]
                    t = line
                    gsub(/[[:space:]]/, "", t)
                    if (length(t) < 12) continue
                    printf "%s\t%s\n", file, line
                    n++
                }
            }
            nadd = 0; dirty = 0
        }
        /^\+\+\+ b\// { flush_hunk(); file = substr($0, 7); n = 0; next }
        /^\+\+\+ /    { flush_hunk(); file = ""; next }
        /^@@/          { flush_hunk(); next }
        /^--- /        { next }
        /^\+/ { if (file != "") add[nadd++] = substr($0, 2); next }
        /^-/   { if (file != "") dirty = 1; next }
        END { flush_hunk() }
    ' "$1"
}

patch_already_applied() {
    local p="$1" absent
    absent="$(patch_markers "$p" | while IFS="$(printf '\t')" read -r file line; do
        if [ ! -f "$file" ] || ! grep -Fxq -- "$line" "$file"; then
            printf 'x'
        fi
    done)"
    [ -z "$absent" ]
}

applied=0
skipped=0

for p in "$PATCH_DIR"/*.patch; do
    name="$(basename "$p")"
    if git apply --check "$p" 2>/dev/null; then
        if git apply "$p"; then
            echo "qinf: applied $name"
            applied=$((applied + 1))
        else
            echo "qinf: ERROR — $name passed --check but failed to apply" >&2
            exit 1
        fi
    elif patch_already_applied "$p"; then
        echo "qinf: skipping $name (already applied)"
        skipped=$((skipped + 1))
    else
        echo "qinf: ERROR — $name neither applies nor is already present." >&2
        echo "qinf:   expected: patch context present (to apply), or the patch's added lines already in the tree (to skip)" >&2
        echo "qinf:   actual:   neither — the ggml source has moved under this patch" >&2
        echo "qinf:   fix:      re-port patches/ against the pinned ggml revision (see docs/note-ggml-upgrade-b10582.md)" >&2
        exit 1
    fi
done

echo "qinf: patch summary — applied=$applied skipped=$skipped"
exit 0
