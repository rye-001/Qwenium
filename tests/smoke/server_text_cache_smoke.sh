#!/usr/bin/env bash
# server_text_cache_smoke.sh — text prefix-cache smoke for the HTTP server
# (--prefix-cache; docs/handoff-server-state.md §1 decision A follow-on).
#
# One server, one fixed system prompt, THREE /v1/completions requests:
#   R1 (cold): MISS — the system-prompt block is prefilled + its KV captured and
#              stored. Coherent answer to question Q1.
#   R2 (SAME system, DIFFERENT question Q2): HIT — the cached system KV is
#              restored, its prefill skipped, only the Q2 suffix is prefilled.
#              The key is over the system block BEFORE the question, so it is
#              question-independent — the "same system prompt, many questions"
#              win (the F9 use case, done transparently).
#   R3 (SAME system, SAME question Q1 as R1): HIT again. Must be BYTE-IDENTICAL
#              to R1 at temperature 0 — the cached system KV is an exact
#              substitute for re-prefilling it (result-transparent / F9-clean).
#
# Gates (all must hold):
#   1. SIGNALS  — exactly one "[prefix-cache] MISS" (R1) then "[prefix-cache] HIT"
#                 for R2 and R3.
#   2. SKIP     — one .snap blob, byte-unchanged after R1 (store only on a miss →
#                 an untouched blob proves the system prefill was skipped).
#   3. TRANSPARENT — R3 answer byte-identical to R1 (cached KV == re-prefill).
#   4. COHERENT — R1 and R2 are real prose, not degenerate token soup.
#
# Heavy: loads a model on Metal. Run ONE model at a time.
#
# The model's recipe MUST expose its KV cache(s) for snapshotting (L2 support):
# Qwen3.5 (qwen35) and Gemma 3/4 do; the plain qwen3 transformer does NOT, and
# --prefix-cache refuses it fail-loud. Pick a snapshot-capable model.
#
# Usage (defaults = Qwen3.5-0.8B; override for cross-family Gemma):
#   tests/smoke/server_text_cache_smoke.sh
#   MODEL=models/gemma-3-1b-it-BF16.gguf tests/smoke/server_text_cache_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SERVER="${SERVER:-build-metal/bin/http_server}"
MODEL="${MODEL:-models/Qwen3.5-0.8B-BF16.gguf}"
SYSTEM="${SYSTEM:-You are a terse assistant for a warehouse order-management system. Answer in one short sentence. Never apologize. Always state units explicitly.}"
Q1="${Q1:-How many pallets fit on a standard truck?}"
Q2="${Q2:-What does SKU stand for?}"
PORT="${PORT:-18098}"
CTX="${CTX:-4096}"

for f in "$SERVER" "$MODEL"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

WORK="$(mktemp -d /tmp/qinf_srv_text_cache_smoke.XXXXXX)"
PREFIX_DIR="$WORK/prefix"
SRV_LOG="$WORK/server.log"
SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT
echo "Work dir: $WORK"

echo "Starting server: $SERVER -m $MODEL -c $CTX -s 1 -p $PORT --prefix-cache $PREFIX_DIR"
"$SERVER" -m "$MODEL" -c "$CTX" -s 1 -p "$PORT" --prefix-cache "$PREFIX_DIR" \
  >"$SRV_LOG" 2>&1 &
SERVER_PID=$!

echo -n "Waiting for server to come up"
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo " up."; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo " FAIL: server died"; sed -n '1,40p' "$SRV_LOG"; exit 1; fi
  echo -n "."; sleep 1
done

make_req() {  # $1 = user prompt, $2 = out file
  python3 - "$SYSTEM" "$1" "$2" <<'PY'
import json, sys
system, prompt, out = sys.argv[1:4]
with open(out, "w") as f:
    json.dump({"model": "local", "max_tokens": 64, "temperature": 0.0,
               "stream": False, "system_prompt": system, "prompt": prompt}, f)
PY
}

post() {  # $1 = req file -> prints completion text
  curl -fsS "http://127.0.0.1:$PORT/v1/completions" \
    -H 'Content-Type: application/json' --data @"$1" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])'
}

judge() {  # stdin = answer; fail loud on degenerate soup
  python3 -c '
import re, sys
txt = sys.stdin.read()
letters = sum(c.isalpha() for c in txt); words = re.findall(r"[A-Za-z]{3,}", txt)
total = max(len(txt), 1); under = txt.count("_")
# Anti-soup: a terse correct reply ("Approximately 250 pallets.") is fine — the
# letter-ratio / underscore checks catch the degenerate token soup, not length.
bad = (len(words) < 2) or (letters/total < 0.4) or (under/total > 0.15)
print("[coherence] words=%d letters=%.0f%% underscores=%.0f%% -> %s"
      % (len(words), 100*letters/total, 100*under/total, "FAIL (degenerate)" if bad else "PASS"))
sys.exit(1 if bad else 0)'
}

make_req "$Q1" "$WORK/r1.json"
make_req "$Q2" "$WORK/r2.json"
make_req "$Q1" "$WORK/r3.json"

echo "=== R1 (cold: MISS) ==="; ANS1="$(post "$WORK/r1.json")"; echo "$ANS1"; echo "$ANS1" | judge
PREFIX_BLOB=( "$PREFIX_DIR"/*.snap )
[[ -e "${PREFIX_BLOB[0]:-}" ]] || { echo "FAIL: R1 wrote no .snap blob"; exit 1; }
PSIG1="$(stat -f '%z %m' "${PREFIX_BLOB[0]}")"
echo "prefix blob: ${PREFIX_BLOB[0]} ($PSIG1)"

echo "=== R2 (same system, different question: HIT) ==="; ANS2="$(post "$WORK/r2.json")"; echo "$ANS2"; echo "$ANS2" | judge
echo "=== R3 (same system, same question: HIT, transparency) ==="; ANS3="$(post "$WORK/r3.json")"; echo "$ANS3"

# Gate 1: signals (one MISS, then HITs).
MISS_N="$(grep -c "\[prefix-cache\] MISS" "$SRV_LOG" || true)"
HIT_N="$(grep -c "\[prefix-cache\] HIT" "$SRV_LOG" || true)"
echo "[signals] MISS=$MISS_N HIT=$HIT_N (expect MISS=1, HIT=2)"
[[ "$MISS_N" -eq 1 && "$HIT_N" -eq 2 ]] || { echo "FAIL: expected 1 MISS + 2 HIT"; exit 1; }

# Gate 2: blob byte-unchanged after R1 (store only fires on a miss).
PSIG2="$(stat -f '%z %m' "${PREFIX_BLOB[0]}")"
[[ "$PSIG1" == "$PSIG2" ]] || { echo "FAIL: .snap changed ($PSIG1 -> $PSIG2) — system prefill not skipped"; exit 1; }
echo "[skip]         prefix blob unchanged -> PASS"

# Gate 3: result-transparency (same system+question → byte-identical answer).
[[ -n "$(printf '%s' "$ANS1" | tr -d '[:space:]')" ]] || { echo "FAIL: R1 answer empty"; exit 1; }
if [[ "$ANS1" != "$ANS3" ]]; then
  echo "FAIL: R3 (HIT) differs from R1 (cold) — cache is NOT result-transparent"
  echo "--- R1 ---"; printf '%s\n' "$ANS1"; echo "--- R3 ---"; printf '%s\n' "$ANS3"; exit 1
fi
echo "[transparency] R3 byte-identical to R1 -> PASS"
echo "================ SERVER TEXT-CACHE SMOKE PASS ================"
