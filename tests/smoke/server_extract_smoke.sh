#!/usr/bin/env bash
# server_extract_smoke.sh — the end-to-end GATE for the Qemmi-Lens extract
# endpoint (--attention-lens, POST /v1/extract; docs/plan-qemmi-lens.md P2/A2).
# This is the LIVE re-gate — the discipline that caught the reverted
# null-in-grammar A5.1. Unit tests cover the pure halves; acceptance is here.
#
# Stage 2 (docs/note-nogrammar-refutation.md): the fixed KV grammar is OFF the
# product path and the two-pass presence gate is DELETED. The lens decodes FREE,
# in ONE prefill. Absence is now earned by OMISSION — the model simply does not
# state the concept — because nothing forces it to fill every key any more. The
# gate's job is unchanged: prove that absent concepts do not fabricate and do not
# take the present ones down with them.
#
# Drives the free tapped extraction + the server-side lens computation (citations
# L3H13 / coverage layer-11 / badge body_mass) on EN + DE order emails, asserting:
#
#   1. Well-formed: HTTP 200, format_version qemmi-lens/v3, one field per hinted
#      {key,gloss} concept (Leg B: complete hint ⇒ no naming zoo), present values
#      verbatim-lifted (found_in_document), a badge per field, coverage present.
#   2. Signal reproduction (N3, end to end): for present distinctive fields the
#      top-1 citation byte span lands INSIDE the value's document source span.
#   3. Collapse-immunity (the config that killed the reverted A5.1): a superset
#      hint with a genuinely-ABSENT concept — including the product-first ordering
#      that produced the total-null collapse — leaves EVERY present field intact
#      and marks only the absent concept `absent`. No `","`/all-null collapse.
#   4. Absent-specificity ~1.00: every genuinely-absent concept comes back
#      `badge:"absent"` (never fabricated present) — the safety property.
#   5. The SHAPE CONTRACT, end to end (lens-format.md): an unparseable extraction
#      is REFUSED with 422 `unparseable_extraction` carrying `raw` — never a
#      partial extraction, and never confused with a 400 bad_request. Forced
#      deterministically with a max_tokens too small to close the object (the
#      "ran off the token budget mid-string" failure the grammar never prevented).
#   6. Fail-loud: empty key_vocabulary ⇒ 400 bad_request; a bare-string concept
#      still parses (back-compat); /v1/extract WITHOUT --attention-lens ⇒ 404.
#
# NOT covered here and deliberately so: the free-vs-constrained comparison lives
# in QDOCS_S1 (build-metal/bin/attn-provenance), which drives the same shipped
# driver with the refuted grammar as a control arm.
#
# Not a byte-exact golden: Metal decode is token-stable-not-byte-identical (§11),
# so the gate asserts structure + rates, not exact masses. Present-recall has a
# tolerance bar (a weak-surface-form concept may drop — it fails SAFE); a
# collapse (0 survivors) is the failure this gate exists to catch. The produced
# lens JSON is written to the work dir as a reference artifact.
#
# Qwen3.6-PINNED: the lens constants (L3H13, layer-11, body_mass 0.538) are
# Qwen3.6-only. Heavy: loads a 35B-class model on Metal. One prefill per request
# now — the presence gate's N+1 prefills died with it.
#
# Usage:
#   tests/smoke/server_extract_smoke.sh
#   MODEL=models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf tests/smoke/server_extract_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SERVER="${SERVER:-build-metal/bin/http_server}"
MODEL="${MODEL:-models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf}"
PORT="${PORT:-18099}"
CTX="${CTX:-4096}"
EN_MIN_INSPAN="${EN_MIN_INSPAN:-4}"   # of the PRESENT distinctive fields, top-1 citation in-span
DE_MIN_INSPAN="${DE_MIN_INSPAN:-3}"
MIN_PRESENT="${MIN_PRESENT:-5}"       # of 6 truly-present concepts, at least this many recalled
                                      # (glossed recall ~0.92; a weak-surface drop fails SAFE)

for f in "$SERVER" "$MODEL"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

WORK="$(mktemp -d /tmp/qinf_extract_smoke.XXXXXX)"
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
stop_server() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }

# ── Phase A: --attention-lens free decode; EN + DE + collapse-immunity ────────
echo "=== Phase A: --attention-lens (free decode, absent-by-omission: EN + DE) ==="
start_server "--attention-lens" "$WORK/lens.log"

python3 - "$PORT" "$WORK" "$EN_MIN_INSPAN" "$DE_MIN_INSPAN" "$MIN_PRESENT" <<'PY'
import json, sys, urllib.request, urllib.error
port, outdir = sys.argv[1], sys.argv[2]
en_min, de_min, min_present = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

# Glosses are ACCEPTED AND CURRENTLY UNUSED (Stage 2): their only consumer was
# the deleted presence gate's Pass-A question. Kept in the request shape (no second
# breaking change) and still sent here so the accepted-shape stays gated. They are
# NOT fed to the extractor — that would change the prompt regime Stage 1 measured.
GLOSS = {
    "customer":      "the buyer — a company, person, or their email/domain",
    "product":       "the item or goods ordered",
    "quantity":      "how many units",
    "unit_price":    "the price per unit",
    "delivery":      "the delivery date or address",
    "order_number":  "the order or reference number",
    "payment_terms": "the payment terms, e.g. net 30 days",
    "shipping_method":"how the goods ship, e.g. courier",
    "tax_id":        "the VAT or tax identification number",
}
def kv(*keys): return [{"key": k, "gloss": GLOSS[k]} for k in keys]

# Both docs name the customer EXPLICITLY (a clear surface form) so present-recall
# is clean; payment_terms / shipping_method / tax_id are genuinely ABSENT.
PRESENT = ["customer","product","quantity","unit_price","delivery","order_number"]
EN = ("From: purchasing@acme-corp.example\nSubject: Purchase Order\n"
      "Customer: ACME Corp GmbH\n"
      "We would like to order 45 units of Titanium Widget at 47.30 EUR per unit.\n"
      "Delivery to our Berlin warehouse by 2025-11-20.\nOrder reference: BST-88213.\n")
DE = ("Von: einkauf@nordwind-gmbh.de\nBetreff: Bestellung\n"
      "Kunde: Nordwind GmbH\n"
      "wir moechten 2500 Stueck Einweghandschuhe Nitril zum Stueckpreis von 0.12 EUR bestellen.\n"
      "Lieferung an unser Lager in Hamburg bis 2025-11-20.\nBestellnummer: 7781.\n")

def extract(doc, keys, expect=200, max_tokens=220):
    body = json.dumps({"document": doc, "key_vocabulary": keys,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/extract",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=600)
        code, payload = r.getcode(), r.read().decode()
    except urllib.error.HTTPError as e:
        code, payload = e.code, e.read().decode()
    if code != expect:
        print(f"FAIL: expected HTTP {expect} got {code}: {payload[:200]}"); sys.exit(1)
    return json.loads(payload) if code == 200 else payload

def overlaps(a_lo, a_hi, b_lo, b_hi, tol=3):
    return a_lo < b_hi + tol and a_hi > b_lo - tol

# FIRST occurrence wins, deliberately. v3 allows several fields per key (one per
# occurrence the model emitted); the first keeps its v2 value and position, so
# reading first-wins is what keeps this gate measuring what it measured before.
# A dict comprehension over d["fields"] would be LAST-wins and would silently
# change which value every assertion below is about.
def by_key(d):
    out = {}
    for f in d["fields"]:
        out.setdefault(f["key"], f)
    return out

# Accumulator for the absent-specificity safety property.
spec_absent_total = spec_absent_ok = 0     # genuinely-absent concepts marked absent

def check_superset(name, doc, present_order, absent_keys, min_inspan):
    """One free-decode request over present_order + absent_keys. Asserts collapse-
    immunity + absent detection; returns the present-survivor count."""
    global spec_absent_total, spec_absent_ok
    concepts = kv(*present_order, *absent_keys)
    d = extract(doc, concepts)
    open(f"{outdir}/lens_{name}.json", "w").write(json.dumps(d, indent=1))
    assert d.get("format_version") == "qemmi-lens/v3", f"{name}: bad format_version"
    # v3: a key may appear several times (one field per emitted occurrence), so
    # collapse consecutive repeats before comparing. The property this gate is
    # really asserting is unchanged — concept ORDER and COMPLETENESS, i.e. the
    # naming zoo stays shut — and it must not be weakened into "some superset".
    got = [f["key"] for f in d["fields"]]
    got_distinct = [k for i, k in enumerate(got) if i == 0 or k != got[i - 1]]
    want = present_order + list(absent_keys)
    assert got_distinct == want, f"{name}: fields {got_distinct} != hint {want} (order/completeness)"
    fields = by_key(d)

    # (4) absent-specificity: every genuinely-absent concept must be `absent`.
    for k in absent_keys:
        spec_absent_total += 1
        badge = fields[k]["badge"]
        if badge == "absent":
            spec_absent_ok += 1
            assert fields[k]["value"] is None, f"{name}:{k} absent but value not null"
            assert fields[k].get("tier") is None, f"{name}:{k} absent but tier not null"
            assert not fields[k]["citations"], f"{name}:{k} absent but has citations"
        else:
            print(f"  !! {name}: absent concept {k} came back {badge} (FALSE-PRESENT)")

    # (3) collapse-immunity + (1)(2): present concepts intact + cited.
    survivors = 0; inspan = 0
    for k in present_order:
        f = fields[k]
        if f["badge"] == "absent":
            print(f"  .. {name}: present concept {k} dropped (absent) — fails SAFE")
            continue
        survivors += 1
        assert f["badge"] in ("grounded","ungrounded"), f"{name}:{k} bad badge {f['badge']}"
        assert f.get("tier") in ("distinctive","short_numeric"), f"{name}:{k} bad tier"
        assert f["found_in_document"], f"{name}:{k} present but value not verbatim"
        assert f["citations"], f"{name}:{k} present but no citations"
        vs = f["value_span"]; c = f["citations"][0]
        if f["tier"] == "distinctive" and vs and overlaps(c["byte_lo"],c["byte_hi"],vs["lo"],vs["hi"]):
            inspan += 1
    for s in d["skipped"]:
        assert s["peak"] < d["used_threshold"], f"{name}: skipped line above used threshold"
    # Survivor bar scales with the present-set size: tolerate ≤1 weak-surface drop
    # (fails safe), but a collapse (0 survivors) is the failure this gate catches.
    # The global MIN_PRESENT caps it so a full 6-concept set still requires ≥5.
    present_bar = min(max(1, len(present_order) - 1), min_present)
    print(f"  {name}: present survivors {survivors}/{len(present_order)} (bar {present_bar}), "
          f"distinctive in-span {inspan} (bar {min_inspan}), "
          f"absent {[fields[k]['badge'] for k in absent_keys]}; skipped={len(d['skipped'])}")
    assert survivors >= present_bar, f"{name}: COLLAPSE — only {survivors}/{len(present_order)} present survived"
    assert inspan >= min_inspan, f"{name}: only {inspan} distinctive citations in-span (< {min_inspan})"
    return survivors

# EN + DE: superset with a genuinely-absent concept (collapse-immunity + absent).
check_superset("en", EN, PRESENT, ["payment_terms"], en_min)
check_superset("de", DE, PRESENT, ["payment_terms"], de_min)

# Config-7 shape: product FIRST + an absent concept (the total-null collapse case).
check_superset("en_product_first", EN,
               ["product","customer","quantity","unit_price","delivery","order_number"],
               ["payment_terms"], en_min)

# Absent-specificity stress: several genuinely-absent concepts + two anchors.
check_superset("en_multi_absent", EN, ["product","order_number"],
               ["payment_terms","shipping_method","tax_id"], 1)

# (4) The safety property: absent-specificity across every absent judgment.
spec = spec_absent_ok / max(1, spec_absent_total)
print(f"  absent-specificity: {spec_absent_ok}/{spec_absent_total} = {spec:.2f} (bar 1.00 — no fabrication)")
assert spec_absent_ok == spec_absent_total, "SAFETY FAIL: a genuinely-absent concept was called present"

# (1b) No field may still carry the deleted v1 audit flag.
d_v2 = extract(EN, kv("order_number"))
assert "presence_grounded" not in json.dumps(d_v2), \
    "v2 still emits presence_grounded — the subtractive change did not land"
print("  v2 shape: presence_grounded is gone OK")

# (5) THE SHAPE CONTRACT, end to end. A max_tokens too small to close the object
# forces exactly the failure the grammar's 'guaranteed parse' never prevented:
# output that runs off the budget mid-object. It must be REFUSED loudly — 422,
# code unparseable_extraction, `raw` carried for inspection — and NEVER reported
# as a partial or empty extraction.
raw_422 = extract(EN, kv(*PRESENT), expect=422, max_tokens=3)
body_422 = json.loads(raw_422)
assert body_422.get("code") == "unparseable_extraction", \
    f"422 body has no machine-readable code: {raw_422[:200]}"
assert "raw" in body_422, "422 must carry `raw` so the failure is inspectable"
for token in ("/v1/extract", "expected", "actual"):
    assert token in body_422["error"], f"422 error not fail-loud (missing {token!r})"
assert "fields" not in body_422, "422 must NEVER carry a partial extraction"
print(f"  shape contract: truncated output -> 422 unparseable_extraction OK "
      f"(raw={body_422['raw'][:24]!r})")

# (6) Back-compat: a bare-string concept still parses (⇒ gloss:"").
d = extract(EN, ["order_number"])
assert d["fields"] and d["fields"][0]["key"] == "order_number", "bare-string concept did not parse"
print("  back-compat: bare-string key_vocabulary element parses OK")

# Fail-loud: empty key_vocabulary ⇒ 400 bad_request — and the 400/422 SPLIT is
# the point: "fix your call" and "the model produced nothing usable" are different
# events, and an importer must tell them apart without string-matching a message.
msg = extract(EN, [], expect=400)
assert "key_vocabulary" in msg or "vocab" in msg.lower() or "concept" in msg.lower(), \
    f"400 body not fail-loud: {msg[:160]}"
assert json.loads(msg).get("code") == "bad_request", f"400 has no code: {msg[:160]}"
print("  fail-loud: empty key_vocabulary -> 400 bad_request OK (split from 422)")
print("PHASE A PASS")
PY

stop_server

# ── Phase A2: `messages` — attribution, and the fail-loud either/or ────────────
# The thread unit. `messages` replaces `document`; every citation then reports
# WHICH message it landed in, so a value reads "from message N of M". That is the
# whole claim: attribution, no staleness verdict — a later-message rule was
# measured to cry wolf on 7 of 9 correctly-handled corrections and stay silent on
# the real failure (docs/note-ss3-matched-pairs.md §3).
echo "=== Phase A2: messages[] attribution + either/or contract ==="
start_server "--attention-lens" "$WORK/lens_msgs.log"

python3 - "$PORT" <<'PY2'
import json, sys, urllib.request

port = sys.argv[1]
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/extract",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

# Three messages; the order facts live in message 0 and message 2 states nothing
# extractable. Attribution must point at a real message, never at "unknown".
MSGS = [
    "From: orders@brightwork.example\nSubject: Restock\n\n"
    "Please book 45 units of the Matte Black Easel Stand at 82.00 GBP each "
    "for Brightwork Studios Ltd.",
    "Thanks — noted. The loading dock is being repainted next week, so please "
    "use the side entrance. Nothing about the order changes.",
    "Acknowledged, nothing further from our side.",
]
KEYS = ["customer", "product", "quantity", "unit_price"]

code, rep = post({"messages": MSGS, "key_vocabulary": KEYS, "max_tokens": 220})
assert code == 200, (code, rep)
assert rep["n_messages"] == 3, rep["n_messages"]
print(f"  n_messages={rep['n_messages']} OK")

present = [f for f in rep["fields"] if f["value"] is not None and f["citations"]]
assert present, "no present cited field came back"
for f in present:
    for ct in f["citations"]:
        assert ct["message"] is not None, f
        assert 0 <= ct["message"] < 3, (f["key"], ct["message"])
    assert f["citation_messages"], f
    assert f["citation_messages"][0] == f["citations"][0]["message"], f
print(f"  every citation on {len(present)} present field(s) names a real message OK")

# The order facts are all in message 0, so that is where the strongest citations
# must land. This is the end-to-end reproduction of the attribution claim.
top = [f["citation_messages"][0] for f in present]
assert top.count(0) >= (len(top) + 1) // 2, top
print(f"  strongest citation lands in message 0 for {top.count(0)}/{len(top)} fields OK")

# A plain `document` request must still report "unknown", never message 0.
code, rep = post({"document": "\n\n".join(MSGS), "key_vocabulary": KEYS, "max_tokens": 220})
assert code == 200, (code, rep)
assert rep["n_messages"] == 0, rep["n_messages"]
for f in rep["fields"]:
    for ct in f.get("citations", []):
        assert ct["message"] is None, f
    assert f["citation_messages"] == [], f
print("  plain document -> n_messages 0, message null (not 0) OK")

# Fail-loud either/or: never both, never neither, never a silent precedence.
for body, why in [({"document": "x", "messages": ["y"], "key_vocabulary": ["a"]}, "both"),
                  ({"key_vocabulary": ["a"]}, "neither"),
                  ({"messages": [], "key_vocabulary": ["a"]}, "empty array"),
                  ({"messages": [""], "key_vocabulary": ["a"]}, "empty text")]:
    code, rep = post(body)
    assert code == 400, (why, code, rep)
    assert rep.get("code") == "bad_request", (why, rep)
print("  fail-loud: both/neither/empty-array/empty-text -> 400 bad_request OK")

# v3: a repeating document against a flat schema. The model emits the key several
# times; every occurrence must come back with its OWN value and its OWN citation,
# not the first one repeated (which is what v2 did, silently, with no coverage
# flag). Gated live because the unit tests drive synthetic tap rows.
INVOICE = (
    "INVOICE\n\nSupplier: Rowan Hardware Supply Co.\nCustomer: Bluepeak Construction LLC\n"
    "Invoice Number: INV-58234\n\nLine Items:\n"
    "Galvanized Steel Bracket - Quantity: 7 - Unit Price: 12.50\n"
    "Industrial Ceiling Fan - Quantity: 19 - Unit Price: 88.00\n"
    "Copper Pipe Fitting - Quantity: 43 - Unit Price: 5.25\n\n"
    "Subtotal: 2131.25\nTax: 170.50\nTotal: 2301.75\n")
code, rep = post({"document": INVOICE,
                  "key_vocabulary": ["supplier","customer","invoice_number","line_item",
                                     "quantity","unit_price","subtotal","tax","total"],
                  "max_tokens": 300})
assert code == 200, (code, rep)
assert rep["format_version"] == "qemmi-lens/v3", rep["format_version"]

def occurrences(key):
    return [f for f in rep["fields"] if f["key"] == key and f["value"] is not None]

qty = occurrences("quantity")
if len(qty) < 2:
    # Not a failure of v3: whether the model enumerates at all depends on the
    # document and the hint. Report it rather than asserting a model behaviour.
    print(f"  !! model emitted only {len(qty)} quantity occurrence(s) — repeated-key "
          f"path not exercised this run (not a v3 defect)")
else:
    vals = [f["value"] for f in qty]
    assert len(set(vals)) == len(vals), f"occurrences share a value (v2 collapse): {vals}"
    for i, f in enumerate(qty):
        assert f["occurrence"] == i, (i, f["occurrence"])
    cites = [f["citations"][0]["byte_lo"] for f in qty if f["citations"]]
    assert len(set(cites)) == len(cites), f"occurrences share a citation: {cites}"
    print(f"  v3 repeated keys: {len(qty)} quantity occurrences {vals}, "
          f"distinct values and distinct citations OK")
print("PHASE A2 PASS")
PY2

stop_server

# ── Phase B: server WITHOUT the flag ⇒ /v1/extract is 404 ──────────────────────
echo "=== Phase B: extract disabled without --attention-lens (expect 404) ==="
start_server "" "$WORK/plain.log"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:$PORT/v1/extract" \
       -H 'Content-Type: application/json' -d '{"document":"x","key_vocabulary":["a"]}')
stop_server
[[ "$CODE" == "404" ]] || { echo "FAIL: expected 404 without --attention-lens, got $CODE"; exit 1; }
echo "  /v1/extract -> 404 when disabled OK"

echo "ALL EXTRACT SMOKE CHECKS PASSED"
