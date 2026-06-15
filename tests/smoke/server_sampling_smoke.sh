#!/usr/bin/env bash
# server_sampling_smoke.sh — per-slot sampler smoke (Part A: temperature/seed).
#
# Before this change the server used one shared GreedySampler and IGNORED the
# request's temperature. Now each slot builds its own sampler from the request:
# temperature 0 => greedy (deterministic), temperature > 0 => TemperatureSampler
# with optional reproducible `seed`. This smoke pins that contract:
#
#   1. GREEDY DETERMINISTIC — temperature 0, same request twice => byte-identical.
#   2. TEMPERATURE APPLIED   — temperature 0.8 != the greedy output (the param is
#                              no longer silently ignored).
#   3. SEED REPRODUCIBLE     — temperature 0.8 + same seed, twice => byte-identical.
#   4. SEED MATTERS          — temperature 0.8 with a different seed => differs.
#
# It does NOT pin exact text (model-dependent); it pins the determinism/seeding
# behavior. Heavy: loads a model on Metal.
#
# Usage (defaults = Qwen3.5-0.8B; any text model works):
#   tests/smoke/server_sampling_smoke.sh
#   MODEL=models/gemma-3-1b-it-BF16.gguf tests/smoke/server_sampling_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SERVER="${SERVER:-build-metal/bin/http_server}"
MODEL="${MODEL:-models/Qwen3.5-0.8B-BF16.gguf}"
PROMPT="${PROMPT:-Write one sentence about the ocean.}"
PORT="${PORT:-18096}"
CTX="${CTX:-4096}"

for f in "$SERVER" "$MODEL"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

WORK="$(mktemp -d /tmp/qinf_srv_sampling_smoke.XXXXXX)"
SRV_LOG="$WORK/server.log"
SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT

echo "Starting server: $SERVER -m $MODEL -c $CTX -s 2 -p $PORT"
"$SERVER" -m "$MODEL" -c "$CTX" -s 2 -p "$PORT" >"$SRV_LOG" 2>&1 &
SERVER_PID=$!

echo -n "Waiting for server to come up"
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo " up."; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo " FAIL: server died"; sed -n '1,40p' "$SRV_LOG"; exit 1; fi
  echo -n "."; sleep 1
done

ask() {  # $1 = json body -> prints completion text
  curl -fsS "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    --data "$1" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])'
}
body() {  # $1 = temperature, $2 = seed (or "none")
  python3 -c '
import json, sys
prompt, temp, seed = sys.argv[1], float(sys.argv[2]), sys.argv[3]
b = {"prompt": prompt, "max_tokens": 40, "temperature": temp}
if seed != "none": b["seed"] = int(seed)
print(json.dumps(b))' "$PROMPT" "$1" "$2"
}

G1="$(ask "$(body 0.0 none)")"
G2="$(ask "$(body 0.0 none)")"
S1a="$(ask "$(body 0.8 1234)")"
S1b="$(ask "$(body 0.8 1234)")"
S2="$(ask "$(body 0.8 9999)")"

pass=0
check() { if eval "$2"; then echo "[$1] -> PASS"; else echo "[$1] -> FAIL"; pass=1; fi; }

[[ -n "$(printf '%s' "$G1" | tr -d '[:space:]')" ]] || { echo "FAIL: greedy output empty"; exit 1; }
check "greedy-deterministic" '[[ "$G1" == "$G2" ]]'
check "temperature-applied"  '[[ "$S1a" != "$G1" ]]'
check "seed-reproducible"    '[[ "$S1a" == "$S1b" ]]'
check "seed-matters"         '[[ "$S1a" != "$S2" ]]'

[[ "$pass" -eq 0 ]] || { echo "================ SERVER SAMPLING SMOKE FAIL ================"; exit 1; }
echo "================ SERVER SAMPLING SMOKE PASS ================"
