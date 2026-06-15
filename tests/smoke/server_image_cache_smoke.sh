#!/usr/bin/env bash
# server_image_cache_smoke.sh — V1 + V2 image-cache smoke for the HTTP server
# (--image-embed-cache / --image-prefix-cache; docs/plan-session-snapshot.md).
#
# One server, one fixed image, THREE /v1/chat/completions requests:
#   R1 (cold): V1 MISS (ViT encode + persist) + V2 MISS (capture image-prefix KV
#              + store). Coherent, image-grounded answer.
#   R2 (same question): V2 HIT — loads the image-prefix KV, skips ViT + image
#              prefill, prefills only the question text. Must be BYTE-identical to
#              R1 (result-transparent) at temperature 0.
#   R3 (DIFFERENT question, same image): V2 HIT again — the key is over the
#              context BEFORE the image (image is rendered before the question),
#              so it is question-independent. This is the decisive "same image,
#              many questions" win AND the sidestep of the image-multirequest bug
#              (requests #2..N load KV bytes into the slot + prefill text only —
#              the path proven immune; server-image-multirequest-bug.md §5).
#
# Gates (all must hold):
#   1. SIGNALS  — server log shows exactly one "[image-prefix-cache] MISS" (R1)
#                 then "[image-prefix-cache] HIT" for R2 and R3.
#   2. V1 SKIP  — one .embd blob, byte-unchanged after R1 (get_or_encode stores
#                 only on a miss → an untouched blob proves the ViT was skipped).
#   3. V2 SKIP  — one .snap blob, byte-unchanged after R1 (store only on a miss).
#   4. TRANSPARENT — R2 answer byte-identical to R1 (the cached KV is an exact
#                 substitute for re-encoding + re-prefilling the image).
#   5. COHERENT — R1 and R3 are image-grounded prose, not the degenerate soup the
#                 image-multirequest bug produces for request #2+ WITHOUT V2.
#
# Heavy: loads a multi-GB model on Metal. Run ONE model at a time. ~25 s/turn.
#
# Usage (defaults = medgemma gemma3-siglip — the family the multirequest bug was
# observed on, so R3 is a real sidestep test):
#   tests/smoke/server_image_cache_smoke.sh
#   MODEL=models/gemma-4-12B-it-Q8_0.gguf \
#   MMPROJ=models/mmproj-gemma-4-12B-it-Q8_0.gguf \
#   IMAGE=IMG_4210.jpg  tests/smoke/server_image_cache_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SERVER="${SERVER:-build-metal/bin/http_server}"
MODEL="${MODEL:-models/medgemma-1.5-4b-it-BF16.gguf}"
MMPROJ="${MMPROJ:-models/mmproj-BF16.gguf}"
IMAGE="${IMAGE:-IMG_4210.jpg}"
PROMPT1="${PROMPT1:-Describe this image in one or two sentences.}"
PROMPT3="${PROMPT3:-What are the main colors in this image?}"
PORT="${PORT:-18097}"
CTX="${CTX:-4096}"

for f in "$SERVER" "$MODEL" "$MMPROJ" "$IMAGE"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

case "$IMAGE" in
  *.png) MIME=image/png ;; *.jpg|*.jpeg) MIME=image/jpeg ;;
  *.gif) MIME=image/gif ;; *.bmp) MIME=image/bmp ;; *) MIME=image/jpeg ;;
esac

WORK="$(mktemp -d /tmp/qinf_srv_cache_smoke.XXXXXX)"
EMBED_DIR="$WORK/embed"
PREFIX_DIR="$WORK/prefix"
SRV_LOG="$WORK/server.log"
SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT
echo "Work dir: $WORK"

echo "Starting server: $SERVER -m $MODEL -j $MMPROJ -c $CTX -s 1 -p $PORT \\"
echo "    --image-embed-cache $EMBED_DIR --image-prefix-cache $PREFIX_DIR"
"$SERVER" -m "$MODEL" -j "$MMPROJ" -c "$CTX" -s 1 -p "$PORT" \
  --image-embed-cache "$EMBED_DIR" --image-prefix-cache "$PREFIX_DIR" \
  >"$SRV_LOG" 2>&1 &
SERVER_PID=$!

echo -n "Waiting for server to come up"
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo " up."; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo " FAIL: server died"; sed -n '1,40p' "$SRV_LOG"; exit 1; fi
  echo -n "."; sleep 1
done

# Build a chat request file with the image as a base64 data URI + a given prompt.
make_req() {  # $1 = prompt, $2 = out file
  python3 - "$IMAGE" "$MIME" "$1" "$2" <<'PY'
import base64, json, sys
path, mime, prompt, out = sys.argv[1:5]
with open(path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
with open(out, "w") as f:
    json.dump({"model": "gemma-local", "max_tokens": 200, "temperature": 0.0,
               "stream": False,
               "messages": [{"role": "user", "content": [
                   {"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}]}, f)
PY
}

post() {  # $1 = req file -> prints assistant content
  curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' --data @"$1" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
}

judge() {  # stdin = answer; fail loud on the degenerate soup signature
  python3 -c '
import re, sys
txt = sys.stdin.read()
letters = sum(c.isalpha() for c in txt); words = re.findall(r"[A-Za-z]{3,}", txt)
total = max(len(txt), 1); under = txt.count("_")
bad = (len(words) < 8) or (letters/total < 0.4) or (under/total > 0.15)
print("[coherence] words=%d letters=%.0f%% underscores=%.0f%% -> %s"
      % (len(words), 100*letters/total, 100*under/total, "FAIL (degenerate)" if bad else "PASS"))
sys.exit(1 if bad else 0)'
}

make_req "$PROMPT1" "$WORK/r1.json"
make_req "$PROMPT1" "$WORK/r2.json"
make_req "$PROMPT3" "$WORK/r3.json"

echo "=== R1 (cold: V1+V2 MISS) ==="; ANS1="$(post "$WORK/r1.json")"; echo "$ANS1"; echo "$ANS1" | judge
EMBED_BLOB=( "$EMBED_DIR"/*.embd ); PREFIX_BLOB=( "$PREFIX_DIR"/*.snap )
[[ -e "${EMBED_BLOB[0]:-}" ]] || { echo "FAIL: R1 wrote no .embd blob (V1)"; exit 1; }
[[ -e "${PREFIX_BLOB[0]:-}" ]] || { echo "FAIL: R1 wrote no .snap blob (V2)"; exit 1; }
ESIG1="$(stat -f '%z %m' "${EMBED_BLOB[0]}")"; PSIG1="$(stat -f '%z %m' "${PREFIX_BLOB[0]}")"
echo "V1 blob: ${EMBED_BLOB[0]} ($ESIG1)"; echo "V2 blob: ${PREFIX_BLOB[0]} ($PSIG1)"

echo "=== R2 (same question: V2 HIT, transparency) ==="; ANS2="$(post "$WORK/r2.json")"; echo "$ANS2"
echo "=== R3 (different question, same image: V2 HIT, sidestep) ==="; ANS3="$(post "$WORK/r3.json")"; echo "$ANS3"; echo "$ANS3" | judge

# Gate 1: signals (one MISS, then HITs).
MISS_N="$(grep -c "\[image-prefix-cache\] MISS" "$SRV_LOG" || true)"
HIT_N="$(grep -c "\[image-prefix-cache\] HIT" "$SRV_LOG" || true)"
echo "[signals] MISS=$MISS_N HIT=$HIT_N (expect MISS=1, HIT=2)"
[[ "$MISS_N" -eq 1 && "$HIT_N" -eq 2 ]] || { echo "FAIL: expected 1 MISS + 2 HIT"; exit 1; }

# Gates 2 & 3: blobs byte-unchanged after R1 (store only fires on a miss).
ESIG2="$(stat -f '%z %m' "${EMBED_BLOB[0]}")"; PSIG2="$(stat -f '%z %m' "${PREFIX_BLOB[0]}")"
[[ "$ESIG1" == "$ESIG2" ]] || { echo "FAIL: V1 .embd changed ($ESIG1 -> $ESIG2) — ViT not skipped"; exit 1; }
[[ "$PSIG1" == "$PSIG2" ]] || { echo "FAIL: V2 .snap changed ($PSIG1 -> $PSIG2) — image prefill not skipped"; exit 1; }
echo "[vit-skip]   V1 blob unchanged -> PASS"
echo "[v2-skip]    V2 blob unchanged -> PASS"

# Gate 4: result-transparency (same question → byte-identical answer).
[[ -n "$(printf '%s' "$ANS1" | tr -d '[:space:]')" ]] || { echo "FAIL: R1 answer empty"; exit 1; }
if [[ "$ANS1" != "$ANS2" ]]; then
  echo "FAIL: R2 (V2 HIT) differs from R1 — cache is NOT result-transparent"
  echo "--- R1 ---"; printf '%s\n' "$ANS1"; echo "--- R2 ---"; printf '%s\n' "$ANS2"; exit 1
fi
echo "[transparency] R2 byte-identical to R1 -> PASS"
echo "================ SERVER IMAGE-CACHE SMOKE PASS ================"
