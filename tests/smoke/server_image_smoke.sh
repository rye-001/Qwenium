#!/usr/bin/env bash
# server_image_smoke.sh — coherence smoke for image input on /v1/chat/completions.
#
# Reuses the gemma4uv coherence smoke (docs/plan-gemma4-vision-impl.md) but over
# HTTP instead of the CLI: starts the server with a Gemma vision projector, POSTs
# one OpenAI image_url content part (a base64 data URI), and prints the model's
# answer. A coherent, image-grounded description = PASS (same bar as the CLI
# coherence smoke; there is no byte-exact reference for the generation).
#
# Heavy: loads a ~12 GB model on the Metal device. Run ONE model at a time
# (never two qwenium/http_server processes concurrently). ~25 s/turn.
#
# Usage:
#   tests/smoke/server_image_smoke.sh                 # defaults below
#   MODEL=models/gemma-4-12B-it-Q8_0.gguf \
#   MMPROJ=models/mmproj-gemma-4-12B-it-Q8_0.gguf \
#   IMAGE=IMG_4210.jpg PROMPT="Describe this image." \
#   tests/smoke/server_image_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SERVER="${SERVER:-build-metal/bin/http_server}"
MODEL="${MODEL:-models/gemma-4-12B-it-Q8_0.gguf}"
MMPROJ="${MMPROJ:-models/mmproj-gemma-4-12B-it-Q8_0.gguf}"
IMAGE="${IMAGE:-IMG_4210.jpg}"
PROMPT="${PROMPT:-Describe this image in one or two sentences.}"
PORT="${PORT:-18099}"
CTX="${CTX:-4096}"

for f in "$SERVER" "$MODEL" "$MMPROJ" "$IMAGE"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

# Detect the image MIME from its extension (stb decodes from the bytes anyway).
case "$IMAGE" in
  *.png) MIME=image/png ;; *.jpg|*.jpeg) MIME=image/jpeg ;;
  *.gif) MIME=image/gif ;; *.bmp) MIME=image/bmp ;;
  *) MIME=image/jpeg ;;
esac

echo "Starting server: $SERVER -m $MODEL -j $MMPROJ -c $CTX -s 1 -p $PORT"
"$SERVER" -m "$MODEL" -j "$MMPROJ" -c "$CTX" -s 1 -p "$PORT" >/tmp/qinf_img_smoke_server.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# Wait for the model to load (GET /health returns 200).
echo -n "Waiting for server to come up"
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo " up."; break; fi
  echo -n "."; sleep 1
done

# Build the OpenAI chat request with the image as a base64 data URI into a temp
# FILE (python3 keeps the large base64 + JSON escaping clean). The body is sent
# via curl --data @file — a multi-MB base64 image exceeds the command-line
# ARG_MAX if passed inline.
REQ_FILE="$(mktemp /tmp/qinf_img_smoke_req.XXXXXX.json)"
trap 'kill "$SERVER_PID" 2>/dev/null || true; rm -f "$REQ_FILE"' EXIT
python3 - "$IMAGE" "$MIME" "$PROMPT" "$REQ_FILE" <<'PY'
import base64, json, sys
path, mime, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
with open(out, "w") as f:
    json.dump({
        "model": "gemma-local",
        "max_tokens": 200,
        "temperature": 0.0,
        "stream": False,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}],
    }, f)
PY

echo "POST /v1/chat/completions (image: $IMAGE, $MIME) ..."
RESP="$(curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' --data @"$REQ_FILE")"

echo "=== Response ==="
# Auto-judge coherence: the failure mode this smoke guards (single-batch image
# prefill on Metal) is DEGENERATE token-soup — runs of '_', repeated single
# tokens, or near-zero alphabetic content. A coherent description is mostly
# words. Fail loud (exit 1) on the soup signature so the smoke is a real gate,
# not an eyeball check. The chunked-prefill path (prefill_multimodal) is what
# keeps this green; see [[project_gemma4_vision]].
echo "$RESP" | python3 -c '
import json, re, sys
d = json.load(sys.stdin)
txt = d["choices"][0]["message"]["content"]
print(txt)
print("\n[finish_reason]", d["choices"][0]["finish_reason"], "[usage]", d.get("usage"))
letters = sum(c.isalpha() for c in txt)
words = re.findall(r"[A-Za-z]{3,}", txt)
underscores = txt.count("_")
total = max(len(txt), 1)
# Soup heuristics: too few real words, too few letters, or underscore-dominated.
bad = (len(words) < 8) or (letters / total < 0.4) or (underscores / total > 0.15)
print("\n[coherence] words=%d letters=%.0f%% underscores=%.0f%% -> %s"
      % (len(words), 100*letters/total, 100*underscores/total,
         "FAIL (degenerate)" if bad else "PASS"))
sys.exit(1 if bad else 0)
'
echo "================"
