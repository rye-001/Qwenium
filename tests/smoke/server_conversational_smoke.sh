#!/usr/bin/env bash
# server_conversational_smoke.sh — the correctness GATE for the warm conversational
# server (--conversational; docs/plan-warm-conversational-server.md).
#
# Unlike --chat-prefix-cache (transparent, warm==cold), this mode is NOT
# transparent to a cold stateless prefill: it retains the assistant reasoning
# scaffold across turns (chat.cpp-grade), so warm != cold by construction. The
# invariant that REPLACES warm==cold is:
#
#     warm-continued  ==  rebuilt-from-log
#
# i.e. continuing a conversation against its resident KV produces the same answer
# as continuing it after the KV was evicted and rebuilt from the saved token log.
# Both append the SAME delta ([closer] + new turn) to the SAME-content KV, so they
# must agree (token-stable + ceiling; the warm KV is built incrementally, the
# rebuilt KV in one prefill, so a borderline final token may FP-flip on Metal —
# the standard gate, strict byte DISABLED; see docs/plan-chat-prefix-cache.md and
# project_metal_mm_mv_fork).
#
# The trick (non-compounding, single continuation each): turn 1 is a cold CREATE,
# byte-identical in both conversations. Then conversation X's turn 2 is served
# WARM (its slot still resident); conversation Y's identical turn 2 is forced
# through a REBUILD by creating a filler conversation between Y's turns which, at
# -s 1, evicts Y's KV (its log survives). Comparing the two turn-2 answers proves
# the append step, without multi-turn FP drift desyncing the comparison.
#
# Gates (all must hold):
#   1. WARM == REBUILD   — token-stable + ceiling on the continued turn. (Plus a
#                          sanity check that the two cold turn-1 creates match.)
#   2. HANDLE-ABSENT == STATELESS — a no-conversation_id request on the
#                          --conversational server matches the same request on a
#                          plain server (invariant 1: the mode never perturbs the
#                          stateless path).
#   3. RECOVER           — an unknown conversation_id is rejected fail-loud with a
#                          "resend the full history" message (not a silent cold run).
#   4. COHERENT          — the warm answers are real prose.
#
# Cross-family rule: run on a HYBRID Qwen (Qwen3.5 — the recurrent-state falsifier;
# the delta append must continue overwrite-semantics state correctly) AND a Gemma.
# Gemma 4 is the pure-attention thinking falsifier; Gemma 3 the clean control.
#
# Heavy: loads a model on Metal, one server at a time.
#
# Usage (defaults = Qwen3.5-0.8B):
#   tests/smoke/server_conversational_smoke.sh
#   MODEL=models/gemma-4-12B-it-Q8_0.gguf tests/smoke/server_conversational_smoke.sh
#   MODEL=models/gemma-3-1b-it-BF16.gguf ENABLE_THINKING=1 MAXTOK=64 tests/smoke/server_conversational_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SERVER="${SERVER:-build-metal/bin/http_server}"
MODEL="${MODEL:-models/Qwen3.5-0.8B-BF16.gguf}"
PORT="${PORT:-18097}"
CTX="${CTX:-4096}"
MAXTOK="${MAXTOK:-64}"
# Disable the <think> scaffold by default so turns COMPLETE (end on a stop token)
# within MAXTOK — a turn cut off by length leaves no clean closer and is not
# delta-continuable (FailNotContinuable), which the gate needs to avoid.
ENABLE_THINKING="${ENABLE_THINKING:-0}"
# Token-stable ceiling: warm and rebuild must agree for >= this many leading chars
# on the continued turn. A real KV/log bug diverges within a few tokens; an FP
# ceiling flip is a single late token.
CEILING="${CEILING:-24}"

# Prompts chosen to COMPLETE cleanly (end on a stop token) so the turn is
# delta-continuable. Thinking models can degenerate on some short phrasings; these
# are known-good for Qwen3.5 with thinking off. The filler need not be continuable
# (it only evicts), but keep it short.
Q1="${Q1:-What is the capital of France? One word.}"
Q2="${Q2:-And the capital of Japan? One word.}"
QF="${QF:-What is one plus one? One word.}"

for f in "$SERVER" "$MODEL"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

WORK="$(mktemp -d /tmp/qinf_conv_smoke.XXXXXX)"
SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT
echo "Work dir: $WORK   model: $MODEL"

start_server() {  # $1 = extra args, $2 = log file
  "$SERVER" -m "$MODEL" -c "$CTX" -s 1 -p "$PORT" $1 >"$2" 2>&1 &
  SERVER_PID=$!
  echo -n "  waiting for server"
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo " up."; return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo " FAIL: server died"; sed -n '1,40p' "$2"; exit 1; fi
    echo -n "."; sleep 1
  done
  echo " FAIL: server never came up"; exit 1
}
stop_server() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }

# ── Phase A: the --conversational server. Drives the warm-vs-rebuild sequence,
#    the recover probe, and the handle-absent capture; writes each to $WORK. ─────
echo "=== Phase A: --conversational server (warm/rebuild/recover/handle-absent) ==="
start_server "--conversational" "$WORK/conv.log"
python3 - "$PORT" "$WORK" "$MAXTOK" "$ENABLE_THINKING" "$Q1" "$Q2" "$QF" <<'PY'
import json, sys, urllib.request, urllib.error
port, outdir, maxtok, think = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
Q1, Q2, QF = sys.argv[5], sys.argv[6], sys.argv[7]

def turn(content, cid=None):
    body = {"model": "local", "messages": [{"role": "user", "content": content}],
            "max_tokens": maxtok, "temperature": 0.0, "stream": False,
            "enable_thinking": think}
    if cid is not None:
        body["conversation_id"] = cid
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        d = json.load(e)
    if "error" in d:
        msg = d["error"] if isinstance(d["error"], str) else json.dumps(d["error"])
        return None, None, None, msg
    ch = d["choices"][0]
    return ch["message"]["content"], ch["finish_reason"], d.get("conversation_id"), None

def w(name, txt):
    open(f"{outdir}/{name}", "w").write("" if txt is None else txt)

# Step 1: create X (cold). Must COMPLETE so it is delta-continuable.
a1x, fr1, xid, err = turn(Q1, "new")
assert err is None and xid, f"create X failed: {err}"
w("a1x.txt", a1x)
if fr1 != "stop":
    print(f"FAIL: turn 1 did not complete (finish_reason={fr1!r}); the turn was cut "
          f"off so it has no clean closer. Raise MAXTOK (now {maxtok}) or set "
          f"ENABLE_THINKING=0.")
    sys.exit(2)
print(f"  step1 create X -> {a1x!r} (finish={fr1}, cid={xid})")

# Step 2: continue X (WARM — X is still resident on the single slot).
a2warm, fr2, _, err = turn(Q2, xid)
assert err is None, f"warm continue failed: {err}"
w("a2warm.txt", a2warm)
print(f"  step2 continue X WARM -> {a2warm!r} (finish={fr2})")

# Step 3: create Y with the SAME Q1 (cold; evicts X under -s 1). Same prompt +
# greedy => a1y must be byte-identical to a1x, so Y's log == X's resident.
a1y, _, yid, err = turn(Q1, "new")
assert err is None and yid, f"create Y failed: {err}"
w("a1y.txt", a1y)

# Step 4: filler create (evicts Y; Y's log survives in the registry).
turn(QF, "new")

# Step 5: continue Y (REBUILD — Y's KV was evicted, rebuilt from its log).
a2reb, _, _, err = turn(Q2, yid)
assert err is None, f"rebuild continue failed: {err}"
w("a2reb.txt", a2reb)
print(f"  step5 continue Y REBUILD -> {a2reb!r}")

# Gate 3 probe: an unknown conversation_id must be rejected fail-loud.
_, _, _, rerr = turn(Q2, "conv_bogus_does_not_exist")
w("recover.txt", rerr)
print(f"  recover probe -> {rerr!r}")

# Gate 2 capture: a request with NO conversation_id (handle-absent).
a_noid, _, _, err = turn(Q1)  # no conversation_id => stateless path
assert err is None, f"handle-absent request failed: {err}"
w("noid.txt", a_noid)

# Gate 5: clearability — DELETE a conversation, then a continue against it must
# fail loud (clearing == eviction == safe; invariant 2).
_, _, zid, err = turn(Q1, "new")
assert err is None and zid, f"create Z failed: {err}"
dreq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/conversations/{zid}",
                              method="DELETE")
with urllib.request.urlopen(dreq, timeout=30) as r:
    dz = json.load(r)
_, _, _, aerr = turn(Q2, zid)  # continue a now-deleted conversation
ok = (dz.get("cleared") == 1) and bool(aerr) and ("unknown conversation" in (aerr or ""))
w("clear_ok.txt", "1" if ok else "0")
w("clear_detail.txt", f"delete={dz} after-continue-error={aerr!r}")
print(f"  clearability: delete={dz} -> continue {aerr!r}")
PY
stop_server

# ── Phase B: a PLAIN server (no --conversational), the handle-absent reference. ──
echo "=== Phase B: plain server (stateless reference for handle-absent) ==="
start_server "" "$WORK/plain.log"
python3 - "$PORT" "$WORK" "$MAXTOK" "$ENABLE_THINKING" "$Q1" <<'PY'
import json, sys, urllib.request
port, outdir, maxtok, think, Q1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1", sys.argv[5]
body = {"model": "local", "messages": [{"role": "user", "content": Q1}],
        "max_tokens": maxtok, "temperature": 0.0, "stream": False, "enable_thinking": think}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    d = json.load(r)
open(f"{outdir}/noid_plain.txt", "w").write(d["choices"][0]["message"]["content"])
PY
stop_server

# ── token-stable + ceiling comparator (shared; strict byte DISABLED) ────────────
cmp_stable() {  # $1 = file A, $2 = file B, $3 = ceiling; echoes "OK COMMON MN KIND"
  python3 - "$1" "$2" "$3" <<'PY'
import sys
a = open(sys.argv[1]).read(); b = open(sys.argv[2]).read(); ceil = int(sys.argv[3])
n = 0
for x, y in zip(a, b):
    if x != y: break
    n += 1
ident = (a == b); mn = min(len(a), len(b))
ok = ident or n == mn or n >= ceil
kind = "byte-identical" if ident else ("prefix-consistent" if n == mn else "token-stable")
print(f"{int(ok)} {n} {mn} {kind}")
PY
}

# Gate 1: warm == rebuild (the core invariant), plus the cold-create sanity match.
echo "=== Gate 1: warm-continued == rebuilt-from-log (>= $CEILING common; strict byte DISABLED) ==="
read -r OK COMMON MN KIND <<<"$(cmp_stable "$WORK/a2warm.txt" "$WORK/a2reb.txt" "$CEILING")"
if [[ "$OK" -ne 1 ]]; then
  echo "FAIL: warm vs rebuild DISAGREE after only $COMMON chars (< ceiling $CEILING)"
  echo "--- WARM ---";    cat "$WORK/a2warm.txt"; echo
  echo "--- REBUILD ---"; cat "$WORK/a2reb.txt";  echo
  exit 1
fi
echo "  warm==rebuild: $KIND ($COMMON common chars) -> PASS"
if ! cmp -s "$WORK/a1x.txt" "$WORK/a1y.txt"; then
  echo "FAIL: the two cold creates (a1x, a1y) differ — turn-1 is not deterministic, the"
  echo "      warm/rebuild comparison is not on equal footing"
  echo "--- a1x ---"; cat "$WORK/a1x.txt"; echo; echo "--- a1y ---"; cat "$WORK/a1y.txt"; echo
  exit 1
fi
echo "  cold-create sanity: a1x == a1y (byte-identical) -> PASS"

# Gate 2: handle-absent == stateless reference.
echo "=== Gate 2: handle-absent == stateless (>= $CEILING common) ==="
read -r OK2 C2 MN2 KIND2 <<<"$(cmp_stable "$WORK/noid.txt" "$WORK/noid_plain.txt" "$CEILING")"
if [[ "$OK2" -ne 1 ]]; then
  echo "FAIL: a no-conversation_id request perturbs the stateless path (diverges after $C2 chars)"
  echo "--- conversational/no-id ---"; cat "$WORK/noid.txt"; echo
  echo "--- plain server ---";          cat "$WORK/noid_plain.txt"; echo
  exit 1
fi
echo "  handle-absent==stateless: $KIND2 ($C2 common chars) -> PASS"

# Gate 3: recover is fail-loud with a useful message.
echo "=== Gate 3: recover (unknown conversation_id fails loud) ==="
if grep -qi "unknown conversation" "$WORK/recover.txt" && grep -qi "resend" "$WORK/recover.txt"; then
  echo "  recover: $(cat "$WORK/recover.txt") -> PASS"
else
  echo "FAIL: unknown conversation_id did not fail loud with a resend hint"
  echo "  got: $(cat "$WORK/recover.txt")"; exit 1
fi

# Gate 4: coherence on the warm answers.
echo "=== Gate 4: coherence ==="
for f in "$WORK/a1x.txt" "$WORK/a2warm.txt"; do
  python3 -c '
import re, sys
txt = open(sys.argv[1]).read().strip()
if len(txt) < 16:
    print("  [coherence] %r terse (%d chars) -> PASS" % (txt, len(txt))); sys.exit(0)
letters = sum(c.isalpha() for c in txt); words = re.findall(r"[A-Za-z]{2,}", txt)
total = max(len(txt), 1); under = txt.count("_")
bad = (len(words) < 2) or (letters/total < 0.4) or (under/total > 0.15)
print("  [coherence] words=%d letters=%.0f%% -> %s"
      % (len(words), 100*letters/total, "FAIL (degenerate)" if bad else "PASS"))
sys.exit(1 if bad else 0)' "$f"
done

# Gate 5: clearability (DELETE then continue fails loud).
echo "=== Gate 5: clearability (DELETE then continue fails loud) ==="
if [[ "$(cat "$WORK/clear_ok.txt")" == "1" ]]; then
  echo "  clearable + safe (cleared=1, post-delete continue fails loud) -> PASS"
else
  echo "FAIL: clearability — $(cat "$WORK/clear_detail.txt")"; exit 1
fi

echo "================ SERVER CONVERSATIONAL SMOKE PASS ================"
