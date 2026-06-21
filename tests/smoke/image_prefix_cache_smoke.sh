#!/usr/bin/env bash
# image_prefix_cache_smoke.sh — coherence + result-transparency + multirequest
# smoke for the opt-in image-prefix KV cache (--image-prefix-cache; vision V2,
# docs/plan-session-snapshot.md).
#
# The image-prefix blob is the slot's KV up to and including the image span,
# keyed by (preceding context + image content_id) — NOT the question. A later
# process with the SAME image + context loads the blob and skips BOTH the ViT
# encode AND the image-position prefill, then prefills only its own question.
#
# Three fresh CLI processes share one cache dir + the SAME image:
#   Run 1 (cold, question Q1): MISS -> encode + prefill + store the image-prefix.
#   Run 2 (warm, question Q1): HIT  -> skip ViT + image prefill; SAME question.
#   Run 3 (warm, question Q2): HIT  -> different question off the cached image.
#
# Gates (all must hold):
#   1. SIGNALS — run 1 logs "[image-prefix-cache] MISS" + writes one prefix-*.snap
#      blob; runs 2 and 3 log "[image-prefix-cache] HIT". The skip fired across
#      processes, and the key is question-independent (run 3's different question
#      still hits).
#   2. RESULT-TRANSPARENT — at temperature 0, run 2's answer (warm, Q1) is
#      BYTE-IDENTICAL to run 1's (cold, Q1). The cached image-prefix KV is a
#      byte-exact substitute for [encode + image prefill], so skipping them
#      changes nothing the user sees (the F9 non-negotiable; validated byte by
#      test-image-prefix-roundtrip).
#   3. MULTIREQUEST SIDESTEP — run 2 IS request #2 in a fresh process: it loads
#      the KV bytes into a fresh slot and prefills only text (the path proven
#      immune), and gates 2+coherence prove it does NOT degenerate the way the
#      live-graph-reuse path did (server-image-multirequest-bug.md). Run 3 then
#      asserts the key is QUESTION-INDEPENDENT (a DIFFERENT question still HITs
#      the cached image-prefix). Run 3's coherence is INFORMATIONAL only: whether
#      an arbitrary second question yields coherent prose depends on the model's
#      image-path prompt-robustness (Gemma4's path is fragile for some prompts
#      even WITHOUT the cache), which is orthogonal to the V2 cache.
#
# Heavy: loads a multi-GB model on Metal three times. Run ONE model at a time.
#
# Usage (defaults = the gemma3-siglip medgemma combo from the V1 vision gate):
#   tests/smoke/image_prefix_cache_smoke.sh
#   MODEL=models/gemma-4-12B-it-Q8_0.gguf \
#   MMPROJ=models/mmproj-gemma-4-12B-it-Q8_0.gguf \
#   IMAGE=IMG_4210.jpg  tests/smoke/image_prefix_cache_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CLI="${CLI:-build-metal/bin/qwenium}"
MODEL="${MODEL:-models/medgemma-1.5-4b-it-BF16.gguf}"
MMPROJ="${MMPROJ:-models/mmproj-BF16.gguf}"
IMAGE="${IMAGE:-IMG_4210.jpg}"
PROMPT1="${PROMPT1:-Describe this image in one or two sentences.}"
PROMPT2="${PROMPT2:-What is the main color shown in this image?}"
CTX="${CTX:-4096}"

for f in "$CLI" "$MODEL" "$MMPROJ" "$IMAGE"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

CACHE_DIR="$(mktemp -d /tmp/qinf_imgprefix_cache.XXXXXX)"
trap 'rm -rf "$CACHE_DIR"' EXIT
echo "Cache dir: $CACHE_DIR"

run_chat() {  # $1 = label, $2 = prompt; writes the assistant answer to stdout
  local label="$1" prompt="$2" log="$CACHE_DIR/$1.log"
  printf '%s\n\nexit\n' "$prompt" | \
    "$CLI" "$MODEL" --chat -t 0 -n 200 \
      --image "$IMAGE" --mmproj "$MMPROJ" \
      --image-prefix-cache "$CACHE_DIR" >"$log" 2>&1 || {
        echo "FAIL: $label CLI exited non-zero" >&2; sed -n '1,40p' "$log" >&2; exit 1; }
  awk '!p { i=index($0,"Assistant: "); if (i){ print substr($0,i+11); p=1; next } }
       p { if ($0 ~ /^\[Timing\]/) exit; print }' "$log"
}

judge() {  # stdin = answer text; fail loud on the degenerate soup signature
  python3 -c '
import re, sys
txt = sys.stdin.read()
letters = sum(c.isalpha() for c in txt)
words = re.findall(r"[A-Za-z]{3,}", txt)
underscores = txt.count("_")
total = max(len(txt), 1)
bad = (len(words) < 8) or (letters/total < 0.4) or (underscores/total > 0.15)
print("[coherence] words=%d letters=%.0f%% underscores=%.0f%% -> %s"
      % (len(words), 100*letters/total, 100*underscores/total,
         "FAIL (degenerate)" if bad else "PASS"))
sys.exit(1 if bad else 0)'
}

echo "=== Run 1 (cold cache, Q1: MISS -> encode + prefill + store) ==="
ANS1="$(run_chat run1 "$PROMPT1")"
echo "$ANS1"
echo "$ANS1" | judge
grep -q "\[image-prefix-cache\] MISS" "$CACHE_DIR/run1.log" || { echo "FAIL: run 1 did not log a MISS"; exit 1; }

BLOBS=( "$CACHE_DIR"/prefix-*.snap )
[[ -e "${BLOBS[0]}" ]] || { echo "FAIL: run 1 wrote no prefix-*.snap blob"; exit 1; }
[[ ${#BLOBS[@]} -eq 1 ]] || { echo "FAIL: expected 1 blob, got ${#BLOBS[@]}"; exit 1; }
echo "Cached blob: ${BLOBS[0]} ($(stat -f '%z B' "${BLOBS[0]}"))"

echo "=== Run 2 (warm cache, Q1: HIT -> skip ViT + image prefill) ==="
ANS2="$(run_chat run2 "$PROMPT1")"
echo "$ANS2"
echo "$ANS2" | judge
grep -q "\[image-prefix-cache\] HIT" "$CACHE_DIR/run2.log" || { echo "FAIL: run 2 did not log a HIT (cache miss?)"; exit 1; }

# Gate 2: result-transparent — same question, warm == cold, byte-identical.
[[ -n "$(printf '%s' "$ANS1" | tr -d '[:space:]')" ]] || { echo "FAIL: run 1 answer is empty"; exit 1; }
if [[ "$ANS1" != "$ANS2" ]]; then
  echo "FAIL: warm answer differs from cold (image-prefix cache is NOT result-transparent)"
  echo "--- cold ---"; printf '%s\n' "$ANS1"
  echo "--- warm ---"; printf '%s\n' "$ANS2"
  exit 1
fi
echo "[transparency] warm answer byte-identical to cold -> PASS"

echo "=== Run 3 (warm cache, Q2 DIFFERENT: HIT -> key question-independence) ==="
ANS3="$(run_chat run3 "$PROMPT2")"
echo "$ANS3"
# Coherence here is INFORMATIONAL (model image-path prompt-robustness, not V2).
echo "$ANS3" | judge || echo "[note] run 3 answer not coherent — a model image-path quirk for this prompt, NOT a cache bug (see gate 2)"
# HARD gate: a different question still HITs (the key excludes the question).
grep -q "\[image-prefix-cache\] HIT" "$CACHE_DIR/run3.log" || { echo "FAIL: run 3 did not HIT (key should be question-independent)"; exit 1; }
echo "[key] a different question HIT the same image-prefix blob -> PASS"

echo "================ SMOKE PASS ================"
