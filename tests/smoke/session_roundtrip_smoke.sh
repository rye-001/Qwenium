#!/usr/bin/env bash
# session_roundtrip_smoke.sh — Phase 2 L1 portable-session gate (cross-process).
#
# Proves a session saved to disk and reloaded in a FRESH process continues
# token-stably (docs/plan-session-snapshot.md, Phase 2). Two runs of the CLI:
#   1. SAVE: generate N tokens, snapshot the L1 state at --save-session-at K,
#      keep generating (so the same run yields the continuous tail).
#   2. LOAD: a second process loads the snapshot, feed_tokens the stored span,
#      and continues. Its token log must equal the SAVE run's tail from token K.
# Then a cross-version load (a different model) must be refused fail-loud.
#
# Run against a Qwen recipe AND a Gemma recipe (CLAUDE.md cross-family rule):
#   MODEL=models/Qwen3.5-0.8B-BF16.gguf tests/smoke/session_roundtrip_smoke.sh
#   MODEL=models/gemma-3-1b-it-BF16.gguf \
#     OTHER_MODEL=models/Qwen3.5-0.8B-BF16.gguf tests/smoke/session_roundtrip_smoke.sh
#
# CPU build (the bitwise-on-CPU tier; Metal divergence is DISABLED). Needs GGUFs,
# so it is NOT a CI unit test.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CLI="${CLI:-build/bin/qwenium}"
MODEL="${MODEL:-models/Qwen3.5-0.8B-BF16.gguf}"
OTHER_MODEL="${OTHER_MODEL:-}"     # optional: a DIFFERENT-arch model for the refusal check
PROMPT="${PROMPT:-Once upon a time}"
TEMP="${TEMP:-0.8}"
N="${N:-12}"
K="${K:-4}"                        # snapshot after K generated tokens

[[ -x "$CLI" ]]  || { echo "FAIL: missing CLI '$CLI' (build target qwenium-cli)"; exit 1; }
[[ -e "$MODEL" ]] || { echo "FAIL: missing model '$MODEL'"; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
SESS="$TMP/session.bin"; SAVE_LOG="$TMP/save.log"; LOAD_LOG="$TMP/load.log"

echo "=== SAVE: $MODEL (snapshot at gen #$K, generate $N) ==="
"$CLI" "$MODEL" -p "$PROMPT" -t "$TEMP" -n "$N" \
  --save-session "$SESS" --save-session-at "$K" --log-tokens-to "$SAVE_LOG" \
  >/dev/null 2>&1

echo "=== LOAD: fresh process resumes the snapshot ==="
"$CLI" "$MODEL" -t "$TEMP" -n "$N" \
  --load-session "$SESS" --log-tokens-to "$LOAD_LOG" \
  >/dev/null 2>&1

# The LOAD log is the continuation from token K onward; it must match the SAVE
# run's tail from line K (line K = the snapshotted current token).
if diff <(tail -n +"$K" "$SAVE_LOG") "$LOAD_LOG" >/dev/null; then
  echo "PASS: cross-process continuation is token-stable ($(wc -l < "$LOAD_LOG" | tr -d ' ') tokens)"
else
  echo "FAIL: cross-process continuation DIVERGED"
  echo "--- save tail (from line $K) ---"; tail -n +"$K" "$SAVE_LOG"
  echo "--- load log ---"; cat "$LOAD_LOG"
  exit 3
fi

# Cross-version refusal: loading the snapshot under a different-arch model must
# fail loud (header arch_id / shape mismatch), never silently re-prefill.
if [[ -n "$OTHER_MODEL" && -e "$OTHER_MODEL" ]]; then
  echo "=== CROSS-VERSION: load into $OTHER_MODEL (must refuse fail-loud) ==="
  if OUT="$("$CLI" "$OTHER_MODEL" -t "$TEMP" -n 2 --load-session "$SESS" 2>&1)"; then
    : # run returned 0 — check it still refused
  fi
  if echo "$OUT" | grep -qiE "refusing to load.*compat_header"; then
    echo "PASS: cross-version load refused fail-loud"
    echo "$OUT" | grep -iE "refusing to load" | head -1
  else
    echo "FAIL: cross-version load was NOT refused fail-loud"
    exit 4
  fi
fi

echo "RESULT: PASS"
