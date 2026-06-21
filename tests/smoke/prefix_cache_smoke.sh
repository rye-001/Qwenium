#!/usr/bin/env bash
# prefix_cache_smoke.sh — coherence + result-transparency smoke for the opt-in
# warm-prefix KV cache (--prefix-cache; Phase 4, docs/plan-session-snapshot.md).
#
# Two fresh CLI processes share one cache dir and the SAME system prompt:
#   Run 1 (cold): MISS  -> prefill the system prompt, store its slot-0 KV blob.
#   Run 2 (warm): HIT   -> load the blob, SKIP the system-prompt prefill.
#
# Gates (both must hold):
#   1. SIGNALS — run 1 logs "[prefix-cache] MISS" + writes one .snap blob; run 2
#      logs "[prefix-cache] HIT". The skip actually fired across processes.
#   2. RESULT-TRANSPARENT — at temperature 0 (greedy) the warm answer is
#      BYTE-IDENTICAL to the cold answer. The cached system KV is a byte-exact
#      substitute for re-prefilling it, so skipping the prefill changes nothing
#      the user sees (the F9 non-negotiable: the cache is transparent, not a
#      result-altering shortcut). A non-empty, identical answer = PASS.
#
# Heavy-ish: loads a text model on Metal twice. Defaults to gemma-3-1b-it.
#
# Usage:
#   tests/smoke/prefix_cache_smoke.sh
#   MODEL=models/Qwen3.5-0.8B-BF16.gguf tests/smoke/prefix_cache_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CLI="${CLI:-build-metal/bin/qwenium}"
MODEL="${MODEL:-models/gemma-3-1b-it-BF16.gguf}"
PROMPT="${PROMPT:-Name three primary colors.}"

for f in "$CLI" "$MODEL"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

WORK="$(mktemp -d /tmp/qinf_prefix_smoke.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
CACHE_DIR="$WORK/cache"
SYS_FILE="$WORK/system.txt"
printf 'You are a terse assistant. Always answer in one short sentence.' >"$SYS_FILE"
echo "Cache dir: $CACHE_DIR"

run_chat() {  # $1 = label; writes the assistant answer to stdout, full log to file
  local label="$1" log="$WORK/$1.log"
  printf '%s\n\nexit\n' "$PROMPT" | \
    "$CLI" "$MODEL" --chat -t 0 -n 80 \
      --system-prompt "$SYS_FILE" --prefix-cache "$CACHE_DIR" >"$log" 2>&1 || {
        echo "FAIL: $label CLI exited non-zero" >&2; sed -n '1,40p' "$log" >&2; exit 1; }
  awk '!p { i=index($0,"Assistant: "); if (i){ print substr($0,i+11); p=1; next } }
       p { if ($0 ~ /^\[Timing\]/) exit; print }' "$log"
}

echo "=== Run 1 (cold: MISS, prefill + store) ==="
ANS1="$(run_chat run1)"
echo "$ANS1"
grep -q "\[prefix-cache\] MISS" "$WORK/run1.log" || { echo "FAIL: run 1 did not log a MISS"; exit 1; }
BLOBS=( "$CACHE_DIR"/prefix-*.snap )
[[ -e "${BLOBS[0]}" ]] || { echo "FAIL: run 1 wrote no .snap blob"; exit 1; }
echo "Cached blob: ${BLOBS[0]} ($(stat -f '%z B' "${BLOBS[0]}"))"

echo "=== Run 2 (warm: HIT, skip prefill) ==="
ANS2="$(run_chat run2)"
echo "$ANS2"
grep -q "\[prefix-cache\] HIT" "$WORK/run2.log" || { echo "FAIL: run 2 did not log a HIT (cache miss?)"; exit 1; }

# Gate 2: non-empty + byte-identical answers.
[[ -n "$(printf '%s' "$ANS1" | tr -d '[:space:]')" ]] || { echo "FAIL: run 1 answer is empty"; exit 1; }
if [[ "$ANS1" != "$ANS2" ]]; then
  echo "FAIL: warm answer differs from cold (cache is NOT result-transparent)"
  echo "--- cold ---"; printf '%s\n' "$ANS1"
  echo "--- warm ---"; printf '%s\n' "$ANS2"
  exit 1
fi
echo "[transparency] warm answer byte-identical to cold -> PASS"
echo "================ SMOKE PASS ================"
