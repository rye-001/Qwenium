#!/usr/bin/env bash
# server_locate_smoke.sh — the end-to-end GATE for POST /v1/locate
# (../qemmi-lens/docs/plan-locate-and-cut.md §3).
#
# Locate is the THIRD and WEAKEST lens claim. Extract says where the model
# looked while WRITING values; verify says where it attends while READING values
# it was handed; locate says only where these KEY TOKENS look. Nothing is
# generated and nothing is audited, so most of what this script gates is that
# the response cannot be mistaken for one of the other two.
#
# Runs on ANY lens-calibrated model, dense or MoE — there is no routing here to
# make MoE a special case. It also runs on a --lens-verify-only server, and gate
# 6 is the one that proves it: that is the whole point of the route.
#
# SEVEN GATES:
#   1. SHAPE     — hits for every requested key, in request order, each hit a
#                  document byte range with a mass.
#   2. CLAIM     — no `extraction_origin`, no `fields`, no `raw`. A locate report
#                  must not be readable as an extraction of either origin.
#   3. DISCLOSE  — `uncalibrated` is present, FALSE for a key vocabulary (LOCHEAD
#                  measured this regime) and TRUE for a question one (it did
#                  not), and `locate_provenance` names the run either way.
#   3b. FLOOR     — the labelled values must actually be found. A FLOOR that
#                  catches breakage, NOT a calibration: the real rate is
#                  LOCHEAD's per-model top3 rate (9B L11h6 96.0%, 35B L11h5
#                  89.3%, 27B L27h10 94.7%) over 75 keys, and four keys on one
#                  document cannot
#                  re-measure it. Note top3 is not top1 —
#                  the 9B is 88.0% top1, which is the rate a caller acting on a
#                  SINGLE span is actually running on.
#   4. VERBATIM  — every returned range slices real document bytes, and the
#                  ranges for one key are disjoint and ordered by PEAK (not by
#                  `mass`, which is the span SUM and can rank differently — this
#                  gate caught that disagreement on its first real run).
#   5. TOP_K     — top_k is honored as a CEILING on spans per key, and fewer is
#                  a legitimate answer (nothing is padded to reach it).
#   6. REFUSALS  — `messages` and `document_id` are 400s, not silent no-ops.
#   7. STABLE    — two identical locate calls return identical bytes.
#   8. NO-POISON  — verify, then locate over a LONGER document, then the SAME
#                  verify: the two reports must agree decision-for-decision and
#                  every body_mass must land in [0, 1].
#
#                  THIS IS THE ONE THAT MATTERS, and it is a regression gate for
#                  a shipped defect (2026-09-18). /v1/locate taps ONE layer while
#                  /v1/verify taps TWO; on a SHARED scheduler galloc cannot see
#                  that difference (ggml_gallocr_needs_realloc keys on node count
#                  and node SIZES, never on GGML_TENSOR_FLAG_OUTPUT), so after a
#                  locate over a longer document verify's smaller graph reused
#                  locate's cached plan, in which the citation tap was not
#                  protected. DeltaNet layers 4 and 6 wrote over it and every
#                  later /v1/verify returned a confident FALSE receipt — negative
#                  body_mass, four of seven badges flipped, no error raised, for
#                  the life of the process.
#
#                  It reproduced ONLY under --lens-verify-only (the full server's
#                  reserve_max_batch plants a large stable plan first), and the
#                  ordering it needs — locate then re-audit on the cheap server —
#                  is exactly the designed client path. Hence VERIFY_ONLY=1 below
#                  is not an optional extra leg.
#
# Heavy: loads a model on Metal. Run ONE model at a time.
#
# Usage:
#   tests/smoke/server_locate_smoke.sh
#   MODEL=models/Qwen3.8-9B-Q8_0.gguf tests/smoke/server_locate_smoke.sh
#   VERIFY_ONLY=1 tests/smoke/server_locate_smoke.sh     # the cheap-server case
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SERVER="${SERVER:-build-metal/bin/qwenium-server}"
MODEL="${MODEL:-models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf}"
PORT="${PORT:-18108}"
CTX="${CTX:-4096}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"

for f in "$SERVER" "$MODEL"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

WORK="$(mktemp -d /tmp/qinf_locate_smoke.XXXXXX)"
SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT

EXTRA=()
[[ "$VERIFY_ONLY" == "1" ]] && EXTRA+=(--lens-verify-only)
echo "Work dir: $WORK   model: $MODEL   verify-only: $VERIFY_ONLY"

# ${EXTRA[@]+...} and not "${EXTRA[@]}": bash 3.2 (the macOS default) treats an
# EMPTY array as unbound under `set -u`, so the plain form kills the common case.
"$SERVER" -m "$MODEL" -c "$CTX" -s 1 -p "$PORT" --attention-lens \
    ${EXTRA[@]+"${EXTRA[@]}"} >"$WORK/server.log" 2>&1 &
SERVER_PID=$!
echo -n "Waiting for server"
for _ in $(seq 1 300); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo " up."; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo " FAIL: server died"; sed -n '1,40p' "$WORK/server.log"; exit 1; fi
  echo -n "."; sleep 1
done

PORT="$PORT" python3 - <<'PY'
import json, os, sys, urllib.request, urllib.error

PORT = os.environ["PORT"]
DOC = ("Von: einkauf@bergblick.example\nBetreff: Bestellung KW42\n\n"
       "Bitte bestellen Sie fuer die Bergblick Handels GmbH 80 Stueck des "
       "Wanderstock Alu Pro zu je 24,90 EUR. Bestelldatum 2025-10-13.\n")
KEYS = ["customer", "quantity", "unit_price", "order_date"]
DOCB = DOC.encode("utf-8")
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

# The first request after start routes differently from every later one on an
# MoE (a one-time first-build effect). Discard one so gate 7 compares
# steady-state passes rather than rediscovering that.
post("/v1/locate", {"document": DOC, "key_vocabulary": KEYS})

r, err = post("/v1/locate", {"document": DOC, "key_vocabulary": KEYS, "top_k": 3})
if err:
    print(f"[setup] FAIL — locate returned HTTP {err[0]}: {err[1][:300]}"); sys.exit(1)

# ── Gate 1: SHAPE ────────────────────────────────────────────────────────────
hits = r.get("hits", {})
shape_ok = (list(hits.keys()) == KEYS and
            all(isinstance(v, list) for v in hits.values()) and
            all(set(h) >= {"byte_lo", "byte_hi", "mass"} for v in hits.values() for h in v))
check(shape_ok, "shape",
      f"{len(hits)} keys in request order, "
      f"{sum(len(v) for v in hits.values())} spans total, "
      f"locate head L{r.get('locate',{}).get('layer')}H{r.get('locate',{}).get('head')}")

# ── Gate 2: CLAIM ────────────────────────────────────────────────────────────
leaked = [k for k in ("extraction_origin", "fields", "raw", "gen", "hover") if k in r]
check(not leaked, "claim",
      "no extraction-shaped members" if not leaked else f"leaked {leaked}")

# ── Gate 3: DISCLOSE ─────────────────────────────────────────────────────────
rq, eq = post("/v1/locate", {"document": DOC, "top_k": 3, "key_vocabulary": [
    {"id": "unit_price", "question": "Was kostet ein Stueck?"}]})
if eq:
    print(f"[disclose] FAIL — question-form locate returned HTTP {eq[0]}: {eq[1][:200]}")
    sys.exit(1)
check(r.get("uncalibrated") is False and rq.get("uncalibrated") is True
      and bool(r.get("locate_provenance")), "disclose",
      f"key mode uncalibrated={r.get('uncalibrated')!r}, question mode "
      f"{rq.get('uncalibrated')!r}, provenance {str(r.get('locate_provenance'))[:48]!r}...")

# ── Gate 3b: FLOOR ───────────────────────────────────────────────────────────
# A floor, not a bar. LOCHEAD measures the rate; this only catches a route that
# has stopped finding anything at all — which is exactly what the citation head
# did here before the sweep moved the constant.
TRUTH = {"customer": "Bergblick Handels GmbH", "quantity": "80",
         "unit_price": "24,90 EUR", "order_date": "2025-10-13"}
found = []
for key, want in TRUTH.items():
    wb = DOCB.find(want.encode())
    we = wb + len(want.encode())
    ok = any(h["byte_lo"] < we and h["byte_hi"] > wb for h in hits.get(key, []))
    if ok: found.append(key)
check(len(found) >= 2, "floor",
      f"{len(found)} of 4 labelled values inside a returned span ({', '.join(found)}) "
      f"— floor is 2; the per-model rate is in this report's locate_provenance, "
      f"not a constant (9B L11h6 = 96.0% top3 / 88.0% top1, 35B L11h5 = 89.3%, "
      f"27B L27h10 = 94.7% top3 / 81.3% top1)")

# ── Gate 4: VERBATIM ─────────────────────────────────────────────────────────
bad = []
for key, spans in hits.items():
    prev_mass = None
    seen = []
    for h in spans:
        if not (0 <= h["byte_lo"] <= h["byte_hi"] <= len(DOCB)):
            bad.append(f"{key}: range [{h['byte_lo']},{h['byte_hi']}) outside the document")
        # PEAK is the ordering key, not mass — a wide flat span outsums a sharp
        # one, and the two disagree on real documents. Both must be present.
        if "peak" not in h:
            bad.append(f"{key}: hit has no 'peak' (the ordering key)")
        elif prev_mass is not None and h["peak"] > prev_mass + 1e-9:
            bad.append(f"{key}: peaks not descending ({prev_mass} then {h['peak']})")
        prev_mass = h.get("peak")
        for lo, hi in seen:
            if h["byte_lo"] < hi and h["byte_hi"] > lo:
                bad.append(f"{key}: spans overlap")
        seen.append((h["byte_lo"], h["byte_hi"]))
sample = ""
if hits.get("unit_price"):
    s = hits["unit_price"][0]
    sample = repr(DOCB[s["byte_lo"]:s["byte_hi"]].decode("utf-8", "replace"))
check(not bad, "verbatim",
      f"every range slices the document; top unit_price span = {sample}" if not bad
      else "; ".join(bad[:3]))

# ── Gate 5: TOP_K ────────────────────────────────────────────────────────────
r1, e1 = post("/v1/locate", {"document": DOC, "key_vocabulary": KEYS, "top_k": 1})
if e1:
    print(f"[top_k] FAIL — HTTP {e1[0]}"); sys.exit(1)
over3 = {k: len(v) for k, v in hits.items() if len(v) > 3}
over1 = {k: len(v) for k, v in r1["hits"].items() if len(v) > 1}
check(not over3 and not over1, "top_k",
      "honored as a ceiling at both 3 and 1" if not (over3 or over1)
      else f"exceeded: top_k=3 {over3}, top_k=1 {over1}")

# ── Gate 6: REFUSALS ─────────────────────────────────────────────────────────
_, e_msgs = post("/v1/locate", {"messages": ["a"], "key_vocabulary": KEYS})
_, e_did  = post("/v1/locate", {"document": DOC, "key_vocabulary": KEYS,
                                "document_id": "d1"})
check(e_msgs is not None and e_msgs[0] == 400 and
      e_did is not None and e_did[0] == 400, "refusals",
      f"messages -> {e_msgs[0] if e_msgs else 'accepted'}, "
      f"document_id -> {e_did[0] if e_did else 'accepted'} "
      f"(both must be 400, never a silent no-op)")

# ── Gate 7: STABLE ───────────────────────────────────────────────────────────
r2, e2 = post("/v1/locate", {"document": DOC, "key_vocabulary": KEYS, "top_k": 3})
if e2:
    print(f"[stable] FAIL — HTTP {e2[0]}"); sys.exit(1)
check(json.dumps(r2, sort_keys=True) == json.dumps(r, sort_keys=True), "stable",
      "two identical calls returned identical bytes")

# ── Gate 8: NO-POISON ──────────────────────────────────────────────────
# The locate document must be LONGER than the verify one: galloc reuses a
# cached plan when the new graph FITS, so a shorter locate cannot poison and a
# gate built on one would pass while the defect is live.
EXTRACTION = json.dumps({"customer": "Bergblick Handels GmbH", "quantity": "80",
                         "unit_price": "24,90 EUR", "order_date": "2025-10-13"},
                        ensure_ascii=False)
LONGER = DOC + ("\nNachtrag: Frachtkosten uebernimmt der Lieferant, Zahlungsziel 30 "
                "Tage netto. Bei Teillieferung bitte vorab informieren. Unsere "
                "Lieferantennummer lautet L-88213. Rahmenvertrag RV-2025-114 gilt "
                "unveraendert fort.\n") * 4

def decisions(rep):
    return [(f["key"], f.get("value"), f.get("grounded"), f.get("badge")) for f in rep["fields"]]
def masses(rep):
    return [f.get("body_mass", 0.0) for f in rep["fields"]]

v_before, e = post("/v1/verify", {"document": DOC, "key_vocabulary": KEYS,
                                  "extraction": EXTRACTION})
if e:
    print(f"[no-poison] FAIL — first verify returned HTTP {e[0]}: {e[1][:200]}"); sys.exit(1)
_, e = post("/v1/locate", {"document": LONGER, "key_vocabulary": KEYS, "top_k": 3})
if e:
    print(f"[no-poison] FAIL — locate over the longer document returned HTTP {e[0]}: {e[1][:200]}")
    sys.exit(1)
v_after, e = post("/v1/verify", {"document": DOC, "key_vocabulary": KEYS,
                                 "extraction": EXTRACTION})
if e:
    # The [0,1] guard in get_attention_taps turning a false receipt into a 400 is
    # the backstop working, not the gate passing: the scheduler split is what is
    # under test here, and a 400 means it is not in place.
    print(f"[no-poison] FAIL — verify AFTER locate returned HTTP {e[0]}: {e[1][:300]}")
    fails.append("no-poison")
else:
    same = decisions(v_before) == decisions(v_after)
    m = masses(v_after)
    in_range = all(0.0 <= x <= 1.0 for x in m)
    check(same and in_range, "no-poison",
          ("decisions identical across the locate, body_mass all within [0, 1]"
           if same and in_range else
           f"decisions_match={same} masses={['%+.3f' % x for x in m]}"))

print()
if fails:
    print("================ LOCATE SMOKE FAIL: " + ", ".join(fails) + " ================")
    sys.exit(1)
print("================ LOCATE SMOKE PASS ================")
PY
