#!/usr/bin/env bash
# lens_drift_gate.sh — the standing DRIFT GATE for the lens path
# (docs/plan-lens-server-shape.md §4.3, Movement 2).
#
# The question every lens-path optimisation has to answer is not "is it still
# bit-identical" — nothing is, across a driver or a batch width, and the claim
# we actually make to a customer is about DECISIONS (architecture.md §11). It
# is: does this transform move one?
#
# Each arm runs the corpus twice — arm A shipped, arm B the candidate — and
# passes only if all three criteria hold: every document emits identical
# tokens, no body line crosses the 0.705 coverage threshold, and the largest
# peak movement stays under the measured line-level decision margin. The
# margin is measured in the same run, on the same unit, rather than quoted.
#
# Arms (DRIFT_ARM inside the probe):
#   chunk  chunked prefill over one shared KV — the Metal mm-vs-mv fork
#   flash  flash attention in prefill, decode left materialized (§4.2.1)
#
# LANGUAGE: this wrapper runs EN+DE by default, because German is what sets the
# margin. DRIFT_LANG=en gives the optimistic arm — label it "English" if you
# quote it.
#
# This is a GPU run of a few minutes per arm, not a unit test — it is not in
# ctest for the same reason the server smokes are not.
#
# Usage:
#   tests/smoke/lens_drift_gate.sh
#   ARMS=flash tests/smoke/lens_drift_gate.sh
#   MODEL=models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf tests/smoke/lens_drift_gate.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BIN="${BIN:-build-metal/bin/attn-provenance}"
MODEL="${MODEL:-models/Qwen3.8-9B-Q8_0.gguf}"
ARMS="${ARMS:-chunk flash}"
# Bilingual by DEFAULT, unlike the probe (whose default stays `en` so existing
# published numbers reproduce). German sets the margin — 0.00126 against
# English's 0.0153 — so a gate that ran English-only would report 25x of room
# where the shipped path has 1.5x. The gate should answer for the shipped path.
export DRIFT_LANG="${DRIFT_LANG:-all}"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/qinf_drift.XXXXXX")"

[ -x "$BIN" ] || { echo "drift gate: '$BIN' expected an executable probe, actual: missing — build the attn-provenance target (Release+Metal)"; exit 1; }
[ -f "$MODEL" ] || { echo "drift gate: MODEL expected a readable gguf, actual: '$MODEL' not found"; exit 1; }

echo "Drift gate | model $MODEL | arms: $ARMS | logs $OUT"

fail=0
for arm in $ARMS; do
    log="$OUT/$arm.log"
    echo
    echo "=== arm '$arm' ==="
    if BANDDRIFT=1 DRIFT_ARM="$arm" QWEN36_MODEL_PATH="$MODEL" "$BIN" > "$log" 2>&1; then
        verdict=PASS
    else
        verdict=FAIL
        fail=1
    fi
    # The probe prints its own criteria table; show that rather than paraphrase.
    sed -n '/---- drift over/,$p' "$log" | sed 's/^/  /'
    echo "  [$arm] $verdict"
done

echo
if [ "$fail" -eq 0 ]; then
    echo "DRIFT GATE: ALL ARMS PASS"
else
    echo "DRIFT GATE: FAIL — a candidate transform moved a lens decision (or a token). Logs in $OUT"
fi
exit "$fail"
