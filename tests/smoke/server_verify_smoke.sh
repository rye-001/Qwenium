#!/usr/bin/env bash
# server_verify_smoke.sh — the end-to-end GATE for POST /v1/verify
# (docs/plan-lens-server-shape.md §3.4.5, Movement 1).
#
# verify() teacher-forces a KNOWN extraction (the exact `raw` text a prior
# /v1/extract emitted) instead of generating one: one prefill over the prompt,
# one head-less TAPPED prefill over the extraction text, no decode loop, no
# sampling, the forward pass truncated after the deeper of
# {citation_layer, coverage_layer}. The correctness claim is NOT bit-for-bit —
# the batch-vs-single numerical fork (architecture.md §11) means a
# teacher-forced multi-row prefill and a token-by-token decode take different
# Metal kernels, measured at 5.6e-4 drift against a 0.019 decision margin
# (plan §2.1) — it is DECISION-FOR-DECISION: same citations (by real-source
# membership, not exact mass), same badges, same skipped[] membership.
#
# This gate is SELF-CHECKING and needs no corpus (plan §3.4.5): it extracts a
# document once, feeds that exact extraction back through /v1/verify, and
# diffs the two reports' decisions. Any well-formed extraction works.
#
# Usage:
#   tests/smoke/server_verify_smoke.sh
#   MODEL=models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf tests/smoke/server_verify_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SERVER="${SERVER:-build-metal/bin/http_server}"
MODEL="${MODEL:-models/Qwen3.8-9B-Q8_0.gguf}"
PORT="${PORT:-18098}"
CTX="${CTX:-4096}"

for f in "$SERVER" "$MODEL"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

WORK="$(mktemp -d /tmp/qinf_verify_smoke.XXXXXX)"
SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT
echo "Work dir: $WORK   model: $MODEL"

start_server() {  # $1 = extra args, $2 = log file
  "$SERVER" -m "$MODEL" -c "$CTX" -s 1 -p "$PORT" $1 >"$2" 2>&1 &
  SERVER_PID=$!
  echo -n "  waiting for server"
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo " up."; return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo " FAIL: server died"; sed -n '1,40p' "$2"; exit 1; fi
    echo -n "."; sleep 1
  done
  echo " FAIL: server never came up"; exit 1
}

echo "=== --attention-lens: /v1/extract then /v1/verify on the SAME extraction ==="
start_server "--attention-lens" "$WORK/lens.log"

# A file, not a pipe: the flash leg below runs this exact check against a
# second, differently-configured server.
cat > "$WORK/check.py" <<'PY'
import json, sys, urllib.request, urllib.error
port, outdir = sys.argv[1], sys.argv[2]
# The numerical configuration this server should stamp on every report
# (LensReport::RuntimeConfig). The point of the member is that two reports
# are only comparable when it matches, so the gate checks it.
want_attention = sys.argv[3] if len(sys.argv) > 3 else 'materialized'

GLOSS = {
    "customer":     "the buyer — a company, person, or their email/domain",
    "product":      "the item or goods ordered",
    "quantity":     "how many units",
    "unit_price":   "the price per unit",
    "delivery":     "the delivery date or address",
    "order_number": "the order or reference number",
    "payment_terms":"the payment terms, e.g. net 30 days",  # genuinely absent
}
KEYS = ["customer", "product", "quantity", "unit_price", "delivery",
        "order_number", "payment_terms"]
DOC = ("From: purchasing@acme-corp.example\nSubject: Purchase Order\n"
       "Customer: ACME Corp GmbH\n"
       "We would like to order 45 units of Titanium Widget at 47.30 EUR per unit.\n"
       "Delivery to our Berlin warehouse by 2025-11-20.\nOrder reference: BST-88213.\n")
VOCAB = [{"key": k, "gloss": GLOSS[k]} for k in KEYS]

def post(path, body, expect=200):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=data, headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=600)
        code, payload = r.getcode(), r.read().decode()
    except urllib.error.HTTPError as e:
        code, payload = e.code, e.read().decode()
    if code != expect:
        print(f"FAIL: POST {path} expected HTTP {expect} got {code}: {payload[:400]}")
        sys.exit(1)
    return json.loads(payload)

# ── Step 1: extract ──────────────────────────────────────────────────────────
extract = post("/v1/extract", {"document": DOC, "key_vocabulary": VOCAB, "max_tokens": 220})
open(f"{outdir}/extract.json", "w").write(json.dumps(extract, indent=1))
assert extract.get("format_version", "").startswith("qemmi-lens/"), \
    f"extract: bad format_version {extract.get('format_version')!r}"
assert "raw" in extract and extract["raw"], "extract: expected a non-empty raw JSON string"
print(f"  extract: {len(extract['fields'])} fields, raw={extract['raw']!r:.120}")

# ── Step 2: verify the SAME extraction back ─────────────────────────────────
verify = post("/v1/verify", {"document": DOC, "key_vocabulary": VOCAB,
                             "extraction": extract["raw"]})
open(f"{outdir}/verify.json", "w").write(json.dumps(verify, indent=1))

for name, rep in (("extract", extract), ("verify", verify)):
    cfg = rep.get("config")
    if not isinstance(cfg, dict):
        print(f"FAIL: {name} report carries no `config` stamp: {rep.get('config')!r}")
        sys.exit(1)
    if cfg.get("attention") != want_attention:
        print(f"FAIL: {name} config.attention expected {want_attention!r}, got {cfg.get('attention')!r}")
        sys.exit(1)
    if not cfg.get("weights") or len(cfg["weights"]) != 16:
        print(f"FAIL: {name} config.weights expected a 16-hex-digit hash, got {cfg.get('weights')!r}")
        sys.exit(1)
    if not cfg.get("kv_type"):
        print(f"FAIL: {name} config.kv_type expected a type name, got {cfg.get('kv_type')!r}")
        sys.exit(1)
if extract["config"] != verify["config"]:
    print(f"FAIL: extract and verify disagree about one server's config: "
          f"{extract['config']} vs {verify['config']}")
    sys.exit(1)
print(f"  config stamp on both reports: attention={want_attention}, "
      f"weights={extract['config']['weights']}, kv_type={extract['config']['kv_type']} — PASS")
assert verify.get("format_version") == extract.get("format_version"), \
    "format_version differs between extract and verify"

# extraction_origin (lens-format.md): the ONE member that must differ between
# the two verbs. "generated" = this model wrote the values; "supplied" = the
# caller handed them in and the model only read them. A consumer presenting a
# supplied report as evidence of how the extraction was produced is the misread
# the format's non-claim exists to prevent, so the distinction is gated here
# rather than trusted.
assert extract.get("extraction_origin") == "generated", \
    f"extract: extraction_origin expected 'generated', actual {extract.get('extraction_origin')!r}"
assert verify.get("extraction_origin") == "supplied", \
    f"verify: extraction_origin expected 'supplied', actual {verify.get('extraction_origin')!r}"

# ── Decision-for-decision diff (§3.4.5) ──────────────────────────────────────
# "Same fields, same badges" — compare emission-ordered field lists directly:
# both drivers parse the SAME text (verify teacher-forces extract's own `raw`),
# so occurrence order and count must match exactly.
ef, vf = extract["fields"], verify["fields"]
assert len(ef) == len(vf), f"field count differs: extract={len(ef)} verify={len(vf)}"

def overlaps(a_lo, a_hi, b_lo, b_hi, tol=3):
    return a_lo < b_hi + tol and a_hi > b_lo - tol

mismatches = []
for i, (a, b) in enumerate(zip(ef, vf)):
    tag = f"fields[{i}] key={a.get('key')!r}"
    # "badge" ("grounded"|"ungrounded"|"absent") IS the wire field — there is no
    # separate "present" member; absent is value:null + badge:"absent".
    for field in ("key", "value", "tier", "badge", "found_in_document", "occurrence"):
        if a.get(field) != b.get(field):
            mismatches.append(f"{tag}: {field} extract={a.get(field)!r} verify={b.get(field)!r}")
    # Citations: same REAL-SOURCE membership, not exact mass (§2.1's 5.6e-4
    # drift). Only meaningful for present, verbatim values.
    if a.get("badge") != "absent" and a.get("found_in_document") and a.get("citations"):
        if not b.get("citations"):
            mismatches.append(f"{tag}: extract cited a source, verify cited none")
        else:
            a0, b0 = a["citations"][0], b["citations"][0]
            if not overlaps(a0["byte_lo"], a0["byte_hi"], b0["byte_lo"], b0["byte_hi"]):
                mismatches.append(
                    f"{tag}: top-1 citation span differs extract=[{a0['byte_lo']},{a0['byte_hi']}) "
                    f"verify=[{b0['byte_lo']},{b0['byte_hi']})")

# skipped[]: SAME membership (by document byte span) — the coverage decision.
def skipped_spans(r):
    return sorted((s["byte_lo"], s["byte_hi"]) for s in r.get("skipped", []))
es, vs = skipped_spans(extract), skipped_spans(verify)
if es != vs:
    mismatches.append(f"skipped[] membership differs: extract={es} verify={vs}")

if mismatches:
    print("FAIL: decision-for-decision mismatches:")
    for m in mismatches:
        print(f"  - {m}")
    sys.exit(1)

print(f"  verify reproduced {len(ef)} fields decision-for-decision "
      f"({sum(1 for f in ef if f.get('badge') != 'absent')} present, "
      f"{len(es)} skipped spans) — PASS")

# ── Fail-loud checks ─────────────────────────────────────────────────────────
# Missing "extraction" ⇒ 400.
r = post("/v1/verify", {"document": DOC, "key_vocabulary": VOCAB}, expect=400)
assert r.get("code") == "bad_request", f"missing extraction: expected bad_request, got {r}"

# Unparseable extraction text ⇒ 422, same shape contract as extract.
r = post("/v1/verify", {"document": DOC, "key_vocabulary": VOCAB,
                        "extraction": "not json at all, just prose"}, expect=422)
assert r.get("code") == "unparseable_extraction", \
    f"unparseable extraction: expected unparseable_extraction, got {r}"
assert "raw" in r, "422 response expected a 'raw' member"

print("  fail-loud: missing extraction -> 400, unparseable extraction -> 422 — PASS")
PY

python3 "$WORK/check.py" "$PORT" "$WORK" materialized

# ── --attention-lens + --flash-attn: allowed HERE, by measurement ────────────
# The pair is decided per MODEL, off the calibration entry
# (LensConstants::flash_prefill_ok). This 9B passed the drift gate — 15/15
# documents token-identical, 0/98 decisions moved — so the flag is accepted
# and means flash in PREFILL only; the tapped decode stays materialized.
# Qwen3.6-35B-A3B FAILS the same gate and is refused, which the unit test
# LensCalibration.FlashPrefillIsPerModelAndDefaultsToRefused pins without
# needing a 17GB model here.
#
# This is the leg where the licence is cashed: extract's tapped rows now come
# from a decode over a KV written by a FLASH prefill, while verify's tapped
# pass is forced materialized. Decision-for-decision agreement across that
# fork is exactly what the gate licensed.
echo "=== --attention-lens --flash-attn: flash prefill, materialized tapped pass ==="
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
start_server "--attention-lens --flash-attn" "$WORK/flash.log"
grep -q "PREFILL only" "$WORK/flash.log" || { echo "FAIL: server did not report a prefill-scoped flash"; sed -n '1,40p' "$WORK/flash.log"; exit 1; }
grep -q "drift gate" "$WORK/flash.log" || { echo "FAIL: the banner did not name the gate that licensed it"; exit 1; }
python3 "$WORK/check.py" "$PORT" "$WORK" flash-prefill

echo "=== /v1/verify WITHOUT --attention-lens -> 404 ==="
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
start_server "" "$WORK/noflag.log"
CODE=$(curl -s -o "$WORK/noflag_resp.json" -w '%{http_code}' \
  -X POST "http://127.0.0.1:$PORT/v1/verify" \
  -H 'Content-Type: application/json' \
  -d '{"document":"x","key_vocabulary":["a"],"extraction":"{}"}')
[[ "$CODE" == "404" ]] || { echo "FAIL: expected 404 without --attention-lens, got $CODE"; cat "$WORK/noflag_resp.json"; exit 1; }
echo "  404 without --attention-lens — PASS"

echo "ALL PASS"
