#!/usr/bin/env bash
# server_routing_replay_smoke.sh — the end-to-end GATE for MoE routing capture,
# fingerprinting and REPLAY (docs/plan-lens-only-engine.md §3, Slices 2a-2c).
#
# A MoE router's top-k is an argmax with no margin, so the same document routed
# under a ~1e-4 perturbation selects different experts: 1-6% of selections flip
# and one extraction in fifteen changes. That is what makes an MoE receipt
# reproducible per configuration and not ACROSS configurations. Replay removes
# the discrete channel by pinning the recorded selections, which puts the MoE in
# the band dense models already occupy (measured: worst 1.447e-02 pinned against
# the dense 9B's own 1.384e-02).
#
# MOE ONLY, and that is not a limitation to work around: a dense model has no
# selection to fingerprint, emits no `routing` at all, and gate 1 below asserts
# exactly that when you point this at one.
#
# WHICH MODELS THIS CAN ACTUALLY RUN ON. Fewer than you would expect, and not
# for a reason this gate controls: --attention-lens refuses any model without a
# calibrated lens entry, so the only MoE it will start on today is
# Qwen3.6-35B-A3B. Gemma 4-26B-A4B is a mixture-of-experts model that exercises
# the OTHER family's MoE path, and this script cannot reach it — the server
# refuses at load with "expected a model with a calibrated lens entry".
#
# So the CROSS-FAMILY coverage for routing replay is NOT here. It lives in the
# probe: PINREPLAY drives the same seam directly, with no lens calibration
# needed, and passes on Gemma 4 with the replay control bitwise-identical 15/15
# (docs/plan-lens-only-engine.md §3). If the calibration table ever gains a
# Gemma entry, point this script at it — until then, do not read a green run
# here as cross-family evidence.
#
# SIX GATES:
#   0. FIRST      — the first extract after start records the same routing as
#                   the next (a stale memory plan once corrupted it; fixed 2026-09-27).
#   1. SHAPE      — MoE extract emits routing{digest, per_layer, layers, top_k};
#                   per_layer has one entry per layer. A DENSE model emits none.
#   2. OPT-IN     — `include_routing_trace` carries the selections; without it
#                   the report has a fingerprint and no payload.
#   3. REPLAY     — verify with the trace reports replayed:true; without it,
#                   replayed:false.
#   4. EFFECT     — the replayed verify's digest DIFFERS from the un-replayed
#                   one. Without this the gate would pass on a no-op that
#                   flipped a flag and pinned nothing.
#   5. STABLE     — two replays of the same trace agree exactly.
#   6. REFUSAL    — an EDITED extraction plus the trace is REFUSED (HTTP 400).
#                   This is the one that matters. /v1/verify exists to check a
#                   SUPPLIED extraction, and an edited one tokenizes
#                   differently, so position p no longer holds the token whose
#                   experts were recorded there. Replaying anyway would pin one
#                   token's routing onto another and emit a confident, wrong
#                   receipt. The trace is bound to its token ids so the server
#                   can tell; this gate proves it still does.
#
# Heavy: loads a model on Metal. Run ONE model at a time.
#
# Usage:
#   tests/smoke/server_routing_replay_smoke.sh
#   MODEL=models/Qwen3.8-9B-Q8_0.gguf tests/smoke/server_routing_replay_smoke.sh  # dense: gate 1 only
# (a Gemma MoE will NOT start here — see "which models" above)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SERVER="${SERVER:-build-metal/bin/qwenium-server}"
MODEL="${MODEL:-models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf}"
PORT="${PORT:-18107}"
CTX="${CTX:-4096}"

for f in "$SERVER" "$MODEL"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

WORK="$(mktemp -d /tmp/qinf_routing_smoke.XXXXXX)"
SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT
echo "Work dir: $WORK   model: $MODEL"

"$SERVER" -m "$MODEL" -c "$CTX" -s 1 -p "$PORT" --attention-lens >"$WORK/server.log" 2>&1 &
SERVER_PID=$!
echo -n "Waiting for server"
for _ in $(seq 1 300); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo " up."; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo " FAIL: server died"; sed -n '1,40p' "$WORK/server.log"; exit 1; fi
  echo -n "."; sleep 1
done

PORT="$PORT" WORK="$WORK" python3 - <<'PY'
import json, os, sys, urllib.request, urllib.error

PORT = os.environ["PORT"]
DOC = ("Von: einkauf@bergblick.example\nBetreff: Bestellung KW42\n\n"
       "Bitte bestellen Sie fuer die Bergblick Handels GmbH 80 Stueck des "
       "Wanderstock Alu Pro zu je 24,90 EUR. Bestelldatum 2025-10-13.\n")
KEYS = ["customer", "quantity", "unit_price", "order_date"]
fails = []

def post(path, obj):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 json.dumps(obj).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=900)), None
    except urllib.error.HTTPError as e:
        return None, (e.code, e.read().decode())

def check(ok, gate, detail):
    print(f"[{gate}] {'PASS' if ok else 'FAIL'} — {detail}")
    if not ok:
        fails.append(gate)

# The FIRST request after start used to record a different routing digest
# from every later one. It was not a first-build effect: the captured prompt
# prefill inherited the memory plan planted at start, in which `moe_idx` was
# scratch, so the recorded trace was wrong (the model's output was not). Fixed
# 2026-09-27 by ForwardPassBase::alloc_readback_graph; gate 0 keeps it fixed.
first, err = post("/v1/extract", {"document": DOC, "key_vocabulary": KEYS, "max_tokens": 220})
if err:
    print(f"[setup] FAIL — extract returned HTTP {err[0]}: {err[1][:200]}"); sys.exit(1)
plain, err = post("/v1/extract", {"document": DOC, "key_vocabulary": KEYS, "max_tokens": 220})
if err:
    print(f"[setup] FAIL — extract returned HTTP {err[0]}: {err[1][:200]}"); sys.exit(1)

# ── Gate 0: FIRST — the first request after start records what later ones do ─
if first.get("routing") is not None:
    check(first["routing"]["digest"] == plain["routing"]["digest"], "first",
          f"first-request routing {first['routing']['digest']} vs steady state {plain['routing']['digest']}")

# ── Gate 1: SHAPE ────────────────────────────────────────────────────────────
rt = plain.get("routing")
if rt is None:
    check(True, "shape", "no routing — a DENSE model has no selection to fingerprint")
    print("\nDense model: gates 2-6 need a mixture-of-experts recipe. Nothing else to test.")
    print("================ ROUTING REPLAY SMOKE PASS (dense) ================")
    sys.exit(0)

check(rt.get("digest", "") != "" and rt.get("layers", 0) > 0
      and len(rt.get("per_layer", [])) == rt.get("layers")
      and rt.get("top_k", 0) > 0,
      "shape", f"digest {rt.get('digest')}, {rt.get('layers')} layers, "
               f"{len(rt.get('per_layer', []))} per-layer digests, top_k {rt.get('top_k')}")

# ── Gate 2: OPT-IN ───────────────────────────────────────────────────────────
check("trace" not in plain["routing"] or not plain["routing"]["trace"],
      "opt-in", "no trace carried when it was not asked for")
withtrace, err = post("/v1/extract", {"document": DOC, "key_vocabulary": KEYS,
                                      "max_tokens": 220, "include_routing_trace": True})
if err:
    print(f"[opt-in] FAIL — HTTP {err[0]}"); sys.exit(1)
trace = withtrace["routing"].get("trace", "")
check(len(trace) > 1000, "opt-in", f"trace carried on request: {len(trace)} chars")

extraction = json.dumps({f["key"]: f["value"] for f in withtrace["fields"]}, ensure_ascii=False)
base = {"document": DOC, "key_vocabulary": KEYS, "extraction": extraction}

# ── Gates 3-5: REPLAY, EFFECT, STABLE ────────────────────────────────────────
v_plain, e1 = post("/v1/verify", base)
v_rep,   e2 = post("/v1/verify", {**base, "routing": withtrace["routing"]})
v_rep2,  e3 = post("/v1/verify", {**base, "routing": withtrace["routing"]})
for e in (e1, e2, e3):
    if e:
        print(f"[replay] FAIL — verify returned HTTP {e[0]}: {e[1][:200]}"); sys.exit(1)

check(v_plain["routing"]["replayed"] is False and v_rep["routing"]["replayed"] is True,
      "replay", f"plain replayed={v_plain['routing']['replayed']}, "
                f"with trace replayed={v_rep['routing']['replayed']}")

check(v_plain["routing"]["digest"] != v_rep["routing"]["digest"],
      "effect", f"replay changed the routing ({v_plain['routing']['digest']} -> "
                f"{v_rep['routing']['digest']}) rather than flipping a flag")

check(v_rep["routing"]["digest"] == v_rep2["routing"]["digest"],
      "stable", f"two replays agree: {v_rep['routing']['digest']}")

# ── Gate 6: REFUSAL ──────────────────────────────────────────────────────────
edited = json.dumps({**{f["key"]: f["value"] for f in withtrace["fields"]},
                     "customer": "WRONG GmbH"}, ensure_ascii=False)
bad, err = post("/v1/verify", {**base, "extraction": edited,
                               "routing": withtrace["routing"]})
if bad is not None:
    check(False, "refusal", "an EDITED extraction was REPLAYED — one token's experts "
                            "pinned onto another, and the receipt would not say so")
else:
    code, msg = err
    check(code == 400 and "routing.trace" in msg,
          "refusal", f"edited extraction refused with HTTP {code}")

print()
if fails:
    print(f"================ ROUTING REPLAY SMOKE FAIL: {', '.join(fails)} ================")
    sys.exit(1)
print("================ ROUTING REPLAY SMOKE PASS ================")
PY
