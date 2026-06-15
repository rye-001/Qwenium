#!/usr/bin/env bash
# image_embed_cache_smoke.sh — coherence + ViT-skip smoke for the opt-in
# persistent image-embedding cache (--image-embed-cache; vision V1,
# docs/plan-session-snapshot.md).
#
# Two fresh CLI processes share one on-disk cache dir for the SAME image:
#   Run 1 (cold dir): MISS -> SigLIP/gemma4uv encode -> persist an .embd blob.
#   Run 2 (warm dir): HIT  -> load the blob, SKIP the ViT pass.
#
# Gates (both must hold):
#   1. COHERENCE — each run's answer is image-grounded prose, not the
#      degenerate token-soup the image path has historically produced
#      (server-image-multirequest-bug.md). Same auto-judge as
#      server_image_smoke.sh: enough real words, mostly letters, few '_'.
#   2. VIT-SKIP (the felt win) — run 1 creates exactly one .embd blob; run 2
#      does NOT rewrite it (get_or_encode only store()s on a miss, so a hit
#      leaves the blob's mtime/size untouched). An unchanged blob after a fresh
#      process proves the cross-process disk hit fired and the encode was
#      skipped — no diagnostic print needed in the silent, opportunistic store.
#
# Heavy: loads a multi-GB model on the Metal device. Run ONE model at a time.
#
# Usage (defaults = the gemma3-siglip medgemma combo from the V1 vision gate):
#   tests/smoke/image_embed_cache_smoke.sh
#   MODEL=models/gemma-4-12B-it-Q8_0.gguf \
#   MMPROJ=models/mmproj-gemma-4-12B-it-Q8_0.gguf \
#   IMAGE=IMG_4210.jpg  tests/smoke/image_embed_cache_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CLI="${CLI:-build-metal/bin/qwenium}"
MODEL="${MODEL:-models/medgemma-1.5-4b-it-BF16.gguf}"
MMPROJ="${MMPROJ:-models/mmproj-BF16.gguf}"
IMAGE="${IMAGE:-IMG_4210.jpg}"
PROMPT="${PROMPT:-Describe this image in one or two sentences.}"
CTX="${CTX:-4096}"

for f in "$CLI" "$MODEL" "$MMPROJ" "$IMAGE"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

CACHE_DIR="$(mktemp -d /tmp/qinf_embed_cache.XXXXXX)"
trap 'rm -rf "$CACHE_DIR"' EXIT
echo "Cache dir: $CACHE_DIR"

run_chat() {  # $1 = label, writes the assistant answer to stdout (log on stderr)
  local label="$1" log="$CACHE_DIR/$1.log"
  printf '%s\n\nexit\n' "$PROMPT" | \
    "$CLI" "$MODEL" --chat -t 0 -n 200 \
      --image "$IMAGE" --mmproj "$MMPROJ" \
      --image-embed-cache "$CACHE_DIR" >"$log" 2>&1 || {
        echo "FAIL: $label CLI exited non-zero" >&2; sed -n '1,40p' "$log" >&2; exit 1; }
  # The CLI prints "User: Assistant: <text>\n[Timing] ...". The "Assistant: "
  # marker is mid-line (after the "User: " prompt), so match it anywhere and
  # print from just past it until the [Timing] line.
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

echo "=== Run 1 (cold cache: encode + persist) ==="
ANS1="$(run_chat run1)"
echo "$ANS1"
echo "$ANS1" | judge

BLOBS=( "$CACHE_DIR"/img-*.embd )
[[ -e "${BLOBS[0]}" ]] || { echo "FAIL: run 1 wrote no .embd blob"; exit 1; }
[[ ${#BLOBS[@]} -eq 1 ]] || { echo "FAIL: expected 1 blob, got ${#BLOBS[@]}"; exit 1; }
BLOB="${BLOBS[0]}"
# Capture an identity fingerprint (size + mtime) that a re-store would change.
SIG1="$(stat -f '%z %m' "$BLOB")"
echo "Cached blob: $BLOB ($SIG1)"

echo "=== Run 2 (warm cache: load + ViT skip) ==="
ANS2="$(run_chat run2)"
echo "$ANS2"
echo "$ANS2" | judge

SIG2="$(stat -f '%z %m' "$BLOB")"
if [[ "$SIG1" != "$SIG2" ]]; then
  echo "FAIL: blob changed across run 2 ($SIG1 -> $SIG2) — cache MISSED, ViT was NOT skipped"
  exit 1
fi
echo "[vit-skip] blob unchanged ($SIG2) — run 2 was a disk HIT, ViT pass skipped -> PASS"
echo "================ SMOKE PASS ================"
