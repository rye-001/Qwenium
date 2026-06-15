#!/usr/bin/env bash
# l2_xprocess_smoke.sh — Phase 3.5 cross-process L2 round-trip + build_path_tag
# sufficiency (docs/plan-session-snapshot.md). Converts the in-process L2 green
# into the actual portability promise.
#
# Process A (save): build state, snapshot the L2 blob (KV + recurrent) + the
#   control-only L1 blob to disk, and dump its OWN live continuation (the GATE 1
#   reference). Process B (load): a fresh process reloads the blobs, resumes by
#   memcpy (L2) and by feed_tokens (L1), and dumps its results.
#
#   GATE 1 (fidelity) : A.live  vs  B.L2-resume   — byte (logits) + token-equal
#   GATE 2 (substit.) : B.L2    vs  B.L1(feed)    — token-stable
#   build_path_tag    : B re-derives the header from its OWN forward pass; the
#                       blob loads only because the path matches. With $OTHER_BIN
#                       (a different-backend build, e.g. CPU vs Metal), the same
#                       blob MUST be refused fail-loud — proving the tag is
#                       SUFFICIENT, not merely that mismatches are refused.
#
# Run on Metal (the load-bearing case) AND CPU:
#   BIN=build-metal/bin/test-l2-session-roundtrip \
#   OTHER_BIN=build/bin/test-l2-session-roundtrip \
#   MODEL=models/Qwen3.5-0.8B-BF16.gguf tests/smoke/l2_xprocess_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BIN="${BIN:-build/bin/test-l2-session-roundtrip}"
OTHER_BIN="${OTHER_BIN:-}"     # optional: a DIFFERENT-backend build for the refusal proof
MODEL="${MODEL:-models/Qwen3.5-0.8B-BF16.gguf}"

[[ -x "$BIN" ]]   || { echo "FAIL: missing harness '$BIN'"; exit 1; }
[[ -e "$MODEL" ]] || { echo "FAIL: missing model '$MODEL'"; exit 1; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
L2="$TMP/l2.bin"; L1="$TMP/l1.bin"; REF="$TMP/ref"; OUT="$TMP/out"

echo "=== Process A (save) — $BIN ==="
"$BIN" "$MODEL" save "$L2" "$L1" "$REF" 2>/dev/null | grep -E "^\[save\]" || true

echo "=== Process B (load, fresh process) — $BIN ==="
"$BIN" "$MODEL" load "$L2" "$L1" "$OUT" 2>/dev/null | grep -E "^\[load\]" || true

rc=0
# GATE 1 — cross-process fidelity. Byte on the logits is the strong claim; on
# Metal it may fall to token-stable (mm-vs-mv / cross-process galloc) — report
# which holds, fail only if even the token sequence diverges.
if cmp -s "$REF.logits" "$OUT.l2.logits"; then
  echo "GATE 1 (A.live vs B.L2): PASS — BYTE-identical logits + tokens"
elif diff -q "$REF.tok" "$OUT.l2.tok" >/dev/null; then
  echo "GATE 1 (A.live vs B.L2): PASS (token-stable; logits NOT byte — report)"
else
  echo "GATE 1 (A.live vs B.L2): FAIL — token sequence diverged across processes"
  paste "$REF.tok" "$OUT.l2.tok" | head; rc=3
fi

# GATE 2 — substitutability (token-stable) inside process B.
if diff -q "$OUT.l2.tok" "$OUT.l1.tok" >/dev/null; then
  echo "GATE 2 (B.L2 vs B.L1 feed_tokens): PASS — token-stable"
else
  echo "GATE 2 (B.L2 vs B.L1 feed_tokens): FAIL — STOP, do not ship L2"
  paste "$OUT.l2.tok" "$OUT.l1.tok" | head; rc=3
fi

# build_path_tag sufficiency: a different-backend build must REFUSE this blob.
if [[ -n "$OTHER_BIN" && -x "$OTHER_BIN" ]]; then
  echo "=== build_path_tag refusal — load A's blob with $OTHER_BIN ==="
  if OUT2="$("$OTHER_BIN" "$MODEL" load "$L2" "$L1" "$TMP/other" 2>&1)"; then :; fi
  if echo "$OUT2" | grep -qiE "refused/failed.*build_path_tag"; then
    echo "build_path_tag: PASS — cross-backend blob refused fail-loud"
    echo "$OUT2" | grep -iE "build_path_tag" | head -1
  else
    echo "build_path_tag: FAIL — cross-backend blob was NOT refused (tag insufficient)"
    echo "$OUT2" | tail -3; rc=4
  fi
fi

[[ $rc -eq 0 ]] && echo "RESULT: PASS" || echo "RESULT: FAIL"
exit $rc
