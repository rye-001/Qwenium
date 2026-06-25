#!/usr/bin/env bash
# chat_latency_vs_length.sh — the COMMIT-GATE measurement for --chat-prefix-cache
# (docs/plan-chat-prefix-cache.md "Commit gate: latency vs length").
#
# NOT a pass/fail gate. It produces the number the commit decision hinges on:
# does per-turn latency stay flat with the cache ON while growing ~linearly with
# conversation length when OFF — and is the gap at the ~4K envelope ceiling big
# enough to justify the slot-lifecycle complexity?
#
# A conversation is grown turn by turn (each turn appends a fixed filler block,
# so the re-sent history gets longer), run twice against the same model:
#   OFF: every turn re-prefills the whole conversation from 0  -> latency ~ length
#   ON : every turn prefills only the new turn (warm prefix)   -> latency ~ flat
# max_tokens is tiny and fixed so DECODE time is constant; the per-turn growth is
# the PREFILL cost, which is exactly what the cache removes. We report each turn's
# prompt-token count (from usage) and wall time, OFF vs ON, plus the ratio at the
# longest turn.
#
# Heavy: loads a model on Metal, one at a time (OFF then ON, sequential).
#
# Usage (defaults = Qwen3.5-0.8B; override for the Gemma leg):
#   tests/smoke/chat_latency_vs_length.sh
#   MODEL=models/gemma-3-1b-it-BF16.gguf TURNS=10 tests/smoke/chat_latency_vs_length.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SERVER="${SERVER:-build-metal/bin/http_server}"
MODEL="${MODEL:-models/Qwen3.5-0.8B-BF16.gguf}"
PORT="${PORT:-18097}"
CTX="${CTX:-4096}"
TURNS="${TURNS:-8}"        # number of conversation turns
FILLER_WORDS="${FILLER_WORDS:-300}"  # words added to history each turn (~tokens)
MAXTOK="${MAXTOK:-8}"      # tiny + fixed: hold decode cost constant, isolate prefill

for f in "$SERVER" "$MODEL"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

WORK="$(mktemp -d /tmp/qinf_chat_latency.XXXXXX)"
SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT
echo "model: $MODEL   turns: $TURNS   filler/turn: ~$FILLER_WORDS words   max_tokens: $MAXTOK"

start_server() {  # $1 = extra args, $2 = log
  "$SERVER" -m "$MODEL" -c "$CTX" -s 1 -p "$PORT" $1 >"$2" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "FAIL: server died"; sed -n '1,40p' "$2"; exit 1; }
    sleep 1
  done; echo "FAIL: server never came up"; exit 1
}
stop_server() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }

# Run the growing conversation; emit "turn prompt_tokens wall_ms" per line to $1.
run_measured() {  # $1 = out csv
  python3 - "$PORT" "$TURNS" "$FILLER_WORDS" "$MAXTOK" "$1" <<'PY'
import json, sys, time, urllib.request
port, turns, fillw, maxtok, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
# Deterministic filler so OFF and ON send byte-identical prompts.
filler = " ".join("lorem%d" % (i % 97) for i in range(fillw))
msgs, rows = [], []
for t in range(1, turns + 1):
    msgs.append({"role": "user",
                 "content": f"[block {t}] {filler}\nReply with just: ack {t}."})
    body = json.dumps({"model": "local", "messages": msgs,
                       "max_tokens": maxtok, "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    ptok = resp.get("usage", {}).get("prompt_tokens", -1)
    ans = resp["choices"][0]["message"]["content"]
    msgs.append({"role": "assistant", "content": ans})
    rows.append((t, ptok, wall_ms))
    print(f"  turn {t:2d}  prompt_tokens={ptok:5d}  wall={wall_ms:8.1f} ms")
with open(out, "w") as f:
    for t, p, w in rows:
        f.write(f"{t} {p} {w:.1f}\n")
PY
}

echo "=== OFF (no cache) ==="
start_server "" "$WORK/off.log"
run_measured "$WORK/off.csv"
stop_server

echo "=== ON (--chat-prefix-cache) ==="
start_server "--chat-prefix-cache" "$WORK/on.log"
run_measured "$WORK/on.csv"
stop_server

echo
echo "=== latency vs length (wall ms per turn) ==="
paste "$WORK/off.csv" "$WORK/on.csv" | python3 -c '
import sys
print("%4s %11s %10s %10s %8s" % ("turn", "prompt_tok", "OFF ms", "ON ms", "speedup"))
last = None
for line in sys.stdin:
    a = line.split()
    turn, ptok, off, on = a[0], a[1], float(a[2]), float(a[5])
    sp = off / on if on > 0 else 0.0
    print("%4s %11s %10.1f %10.1f %7.2fx" % (turn, ptok, off, on, sp))
    last = (ptok, off, on, sp)
if last:
    print()
    print("At the longest turn (~%s prompt tokens): OFF=%.0fms  ON=%.0fms  -> %.2fx"
          % (last[0], last[1], last[2], last[3]))
    print("Commit decision: is that gap, at the ~4K envelope ceiling, worth the slot-lifecycle complexity?")
'
echo "================ MEASUREMENT DONE ================"
