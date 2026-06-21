#!/usr/bin/env bash
# server_grammar_smoke.sh — per-slot grammar / constrained-output smoke (Part B).
#
# A request may carry a GBNF `grammar`; the slot's sampler then masks logits to
# grammar-valid tokens each step and generation stops when the grammar reaches an
# accepting state. This smoke pins that contract on /v1/completions:
#
#   1. ENUM CONSTRAINT — grammar root ::= ("yes"|"no"|"maybe") forces the output
#                        to be EXACTLY one of those, finish_reason == "stop".
#   2. JSON SHAPE      — a small JSON-object grammar yields parseable JSON of the
#                        required shape, finish_reason == "stop".
#   3. STOP AT ACCEPT  — the enum answer is a SINGLE word (the grammar terminates
#                        the moment it is satisfiable), not a runaway to max_tokens.
#   4. BAD GBNF        — an unparseable grammar returns an HTTP error naming the
#                        'grammar' parameter and the server stays alive (fail-loud,
#                        no crash).
#   5. NON-GRAMMAR     — a request without a grammar is still coherent prose
#                        (the grammar path doesn't disturb the default path).
#
# Heavy: loads a model on Metal. The constraint is model-independent (the mask
# forces conformance), so this runs cross-family unchanged.
#
# Usage (defaults = Qwen3.5-0.8B):
#   tests/smoke/server_grammar_smoke.sh
#   MODEL=models/gemma-3-1b-it-BF16.gguf tests/smoke/server_grammar_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SERVER="${SERVER:-build-metal/bin/http_server}"
MODEL="${MODEL:-models/Qwen3.5-0.8B-BF16.gguf}"
PORT="${PORT:-18094}"
CTX="${CTX:-4096}"

for f in "$SERVER" "$MODEL"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

WORK="$(mktemp -d /tmp/qinf_srv_grammar_smoke.XXXXXX)"
SRV_LOG="$WORK/server.log"
SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT

echo "Starting server: $SERVER -m $MODEL -c $CTX -s 1 -p $PORT"
"$SERVER" -m "$MODEL" -c "$CTX" -s 1 -p "$PORT" >"$SRV_LOG" 2>&1 &
SERVER_PID=$!

echo -n "Waiting for server to come up"
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo " up."; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo " FAIL: server died"; sed -n '1,40p' "$SRV_LOG"; exit 1; fi
  echo -n "."; sleep 1
done

URL="http://127.0.0.1:$PORT/v1/completions"
pass=0
check() { if eval "$2"; then echo "[$1] -> PASS"; else echo "[$1] -> FAIL"; pass=1; fi; }

# --- 1 & 3: enum grammar -> exactly one word, finish=stop ---
ENUM_REQ="$(python3 -c 'import json;print(json.dumps({"prompt":"Is the sky blue on a clear day? Answer:","max_tokens":20,"temperature":0.0,"grammar":"root ::= (\"yes\" | \"no\" | \"maybe\")"}))')"
read -r ENUM_TEXT ENUM_FIN ENUM_TOKS < <(curl -fsS "$URL" -H 'Content-Type: application/json' --data "$ENUM_REQ" \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);c=d["choices"][0];print(c["text"].strip(),c["finish_reason"],d["usage"]["completion_tokens"])')
echo "  enum -> text='$ENUM_TEXT' finish=$ENUM_FIN tokens=$ENUM_TOKS"
check "enum-constraint" '[[ "$ENUM_TEXT" == "yes" || "$ENUM_TEXT" == "no" || "$ENUM_TEXT" == "maybe" ]]'
check "enum-finish-stop" '[[ "$ENUM_FIN" == "stop" ]]'
check "stop-at-accept"  '[[ "$ENUM_TOKS" -le 3 ]]'

# --- 2: JSON-object grammar -> parseable JSON of the right shape ---
JSON_REQ="$(python3 -c '
import json
gbnf = "root ::= \"{\\\"ok\\\":\" bool \"}\"\nbool ::= \"true\" | \"false\""
print(json.dumps({"prompt":"Return JSON {\"ok\":true} if 2>1 else {\"ok\":false}.","max_tokens":30,"temperature":0.0,"grammar":gbnf}))')"
JSON_OUT="$(curl -fsS "$URL" -H 'Content-Type: application/json' --data "$JSON_REQ" \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["text"])')"
echo "  json -> '$JSON_OUT'"
check "json-shape" 'python3 -c "import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if isinstance(d,dict) and isinstance(d.get(\"ok\"),bool) else 1)" "$JSON_OUT"'

# --- 4: bad GBNF -> HTTP error naming the param, server stays alive ---
BAD_CODE="$(curl -sS -o "$WORK/bad.json" -w '%{http_code}' "$URL" -H 'Content-Type: application/json' \
  --data '{"prompt":"hi","max_tokens":10,"grammar":"root ::= not valid ((("}')"
echo "  bad-grammar -> HTTP $BAD_CODE, body=$(cat "$WORK/bad.json")"
check "bad-grammar-rejected" '[[ "$BAD_CODE" -ge 400 ]] && grep -q "grammar" "$WORK/bad.json"'
check "server-alive"         'kill -0 "$SERVER_PID" 2>/dev/null'

# --- 5: non-grammar request still coherent ---
PLAIN="$(curl -fsS "$URL" -H 'Content-Type: application/json' \
  --data '{"prompt":"Name one ocean.","max_tokens":40,"temperature":0.0}' \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["text"])')"
check "non-grammar-coherent" '[[ -n "$(printf "%s" "$PLAIN" | tr -d "[:space:]")" ]]'

[[ "$pass" -eq 0 ]] || { echo "================ SERVER GRAMMAR SMOKE FAIL ================"; exit 1; }
echo "================ SERVER GRAMMAR SMOKE PASS ================"
