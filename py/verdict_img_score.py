#!/usr/bin/env python3
"""Score VERDICT-IMG results (docs/note-verdict-img-probe.md).

Reads the probe's TSV (tests/perf/probe_verdict_img.cpp) and prints, per
family: AUROC, raw accuracy at p >= 0.5, the blank page's p, accuracy after
subtracting the blank page's logit, mean p on yes/no items, compliance, and
the misses. Stdlib only.

  python3 py/verdict_img_score.py .session-results/verdict_img/r27.tsv
"""
import csv
import math
import sys


def logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")   # one class only: undefined
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def hard(rows):
    """The harder arm (file names {c|s}_b{base}_{FAM}_{variant}_s?t?d?): H1-H3
    per family per condition, and accuracy by the asked family's variant."""
    for r in rows:
        cond, _, fam, var, _ = r["image"].split("_")
        r["cond"], r["target"], r["variant"] = cond, fam, var
    print("cond family  AUROC  acc@0.5  lures no  yes-hard  compliant   H1 H2 H3")
    for cond in ("c", "s"):
        for f in sorted({r["family"] for r in rows}):
            rs = [r for r in rows if r["cond"] == cond and r["family"] == f]
            pos = [r["p_yes"] for r in rs if r["expected"] == "yes"]
            neg = [r["p_yes"] for r in rs if r["expected"] == "no"]
            au = auroc(pos, neg)
            acc = sum((r["p_yes"] >= 0.5) == (r["expected"] == "yes") for r in rs) / len(rs)
            own = [r for r in rs if r["target"] == f]
            lures = [r for r in own if r["variant"] in ("lure1", "lure2")]
            lure_no = sum(r["p_yes"] < 0.5 for r in lures)
            yh = [r for r in own if r["variant"] == "yeshard"]
            yh_ok = sum(r["p_yes"] >= 0.5 for r in yh)
            comp = sum(r["compliant"] == "1" for r in rs) / len(rs)
            ok = lambda b: "pass" if b else "FAIL"
            print(f"{cond}    {f:6}  {au:.3f}  {acc:6.1%}   {lure_no:2}/{len(lures)}     {yh_ok}/{len(yh)}      "
                  f"{comp:6.1%}   {ok(au >= 0.90)} {ok(acc >= 0.90)} {ok(lure_no >= 9)}")
    print("\nby the asked family's variant: mean p_yes [min..max], wrong at 0.5")
    for cond in ("c", "s"):
        for f in sorted({r["family"] for r in rows}):
            for v in ("yes", "yeshard", "no", "lure1", "lure2"):
                ps = [r for r in rows if r["cond"] == cond and r["family"] == f and r["target"] == f and r["variant"] == v]
                vals = [r["p_yes"] for r in ps]
                wrong = [r["image"] for r in ps if (r["p_yes"] >= 0.5) != (r["expected"] == "yes")]
                print(f"  {cond} {f:5} {v:8} {sum(vals)/len(vals):.3f} [{min(vals):.3f}..{max(vals):.3f}]"
                      f"  wrong {len(wrong)}{'  ' + ' '.join(wrong) if wrong else ''}")
    print("\ncross-family: questions NOT about the image's lure/hard mark, wrong at 0.5")
    for cond in ("c", "s"):
        rs = [r for r in rows if r["cond"] == cond and r["family"] != r["target"]]
        wrong = [f"{r['image']}:{r['family']}({r['p_yes']:.3f})" for r in rs
                 if (r["p_yes"] >= 0.5) != (r["expected"] == "yes")]
        print(f"  {cond}: {len(wrong)}/{len(rs)} wrong  {' '.join(wrong)}")


def stamp2(rows):
    """§8 (file names {c|s}_f{base}_STAMP_{variant}_...): S1 = strict question
    (STAMPX) at 0.5, S2 = original question (STAMP) at the pre-set cut 0.9."""
    for r in rows:
        cond, _, _, var, _ = r["image"].split("_")
        r["cond"], r["variant"] = cond, var
    print("cond question  cut   AUROC  accuracy  lures no  yes found  bar")
    for cond in ("c", "s"):
        for fam, cut, gate in (("STAMP", 0.5, "-"), ("STAMP", 0.9, "S2"), ("STAMPX", 0.5, "S1")):
            rs = [r for r in rows if r["cond"] == cond and r["family"] == fam]
            pos = [r["p_yes"] for r in rs if r["expected"] == "yes"]
            neg = [r["p_yes"] for r in rs if r["expected"] == "no"]
            acc = sum((r["p_yes"] >= cut) == (r["expected"] == "yes") for r in rs) / len(rs)
            lures = [r for r in rs if r["variant"] in ("lure1", "lure2")]
            lure_no = sum(r["p_yes"] < cut for r in lures)
            yes_ok = sum(r["p_yes"] >= cut for r in rs if r["expected"] == "yes")
            verdict = "-" if gate == "-" else (gate + (" pass" if acc >= 0.90 and lure_no >= 18 else " FAIL"))
            print(f"{cond}    {fam:7}  {cut:.1f}  {auroc(pos, neg):.3f}  {acc:6.1%}    {lure_no:2}/{len(lures)}     "
                  f"{yes_ok:2}/{len(pos)}     {verdict}")
    print("\nmean p_yes [min..max] by variant")
    for cond in ("c", "s"):
        for fam in ("STAMP", "STAMPX"):
            for v in ("yes", "faint", "partial", "no", "lure1", "lure2"):
                vals = [r["p_yes"] for r in rows if r["cond"] == cond and r["family"] == fam and r["variant"] == v]
                print(f"  {cond} {fam:6} {v:7} {sum(vals)/len(vals):.3f} [{min(vals):.3f}..{max(vals):.3f}]")


def paper(rows):
    """§9 real paper (file names NN_photo.jpg / NN_scan.jpg): signature and
    date at 0.5, stamp (printed, not ink) at the pre-set cut 0.9."""
    cuts = {"SIG": 0.5, "DATE": 0.5, "STAMP": 0.9}
    print("capture family  cut  AUROC  correct  bar (>= 11/12)")
    for cap in ("photo", "scan"):
        for f in ("SIG", "STAMP", "DATE"):
            rs = [r for r in rows if r["image"].endswith(f"_{cap}.jpg") and r["family"] == f]
            if not rs:
                continue
            pos = [r["p_yes"] for r in rs if r["expected"] == "yes"]
            neg = [r["p_yes"] for r in rs if r["expected"] == "no"]
            ok = sum((r["p_yes"] >= cuts[f]) == (r["expected"] == "yes") for r in rs)
            print(f"{cap:7} {f:6}  {cuts[f]:.1f}  {auroc(pos, neg):.3f}  {ok:2}/{len(rs)}    "
                  f"{('pass' if ok >= 11 else 'FAIL') if len(rs) == 12 else f'n={len(rs)}, bar needs 12'}")
    print("\nper sheet: p_yes (expected) — * = wrong at its cut")
    for r in sorted(rows, key=lambda r: (r["image"], r["family"])):
        wrong = (r["p_yes"] >= cuts[r["family"]]) != (r["expected"] == "yes")
        print(f"  {r['image']:14} {r['family']:5} {r['p_yes']:.3f} ({r['expected']}){' *' if wrong else ''}")


def main(path):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    for r in rows:
        r["p_yes"] = float(r["p_yes"])
    print(f"{path}: {len(rows)} questions")
    if rows and rows[0]["image"][2:] in ("_photo.jpg", "_scan.jpg"):
        return paper(rows)
    if rows and rows[0]["image"][:3] in ("c_f", "s_f"):
        return stamp2(rows)
    if rows and rows[0]["image"][:2] in ("c_", "s_"):
        return hard(rows)
    fams = sorted({r["family"] for r in rows})
    print("family  AUROC  raw@0.5  p_blank  calibrated  mean p|yes  mean p|no  compliant  ms/q")
    for f in fams:
        rs = [r for r in rows if r["family"] == f and r["base"] != "-1"]
        blank = [r for r in rows if r["family"] == f and r["base"] == "-1"]
        pos = [r["p_yes"] for r in rs if r["expected"] == "yes"]
        neg = [r["p_yes"] for r in rs if r["expected"] == "no"]
        raw = sum((r["p_yes"] >= 0.5) == (r["expected"] == "yes") for r in rs) / len(rs)
        pb = blank[0]["p_yes"] if blank else float("nan")
        cal = (sum((logit(r["p_yes"]) > logit(pb)) == (r["expected"] == "yes") for r in rs) / len(rs)
               if blank else float("nan"))
        comp = sum(r["compliant"] == "1" for r in rs) / len(rs)
        ms = sum(float(r["ms"]) for r in rs) / len(rs)
        print(f"{f:6}  {auroc(pos, neg):.3f}  {raw:6.1%}   {pb:.4f}   {cal:6.1%}     "
              f"{sum(pos)/len(pos):.3f}      {sum(neg)/len(neg):.3f}     {comp:6.1%}   {ms:.0f}")
        misses = [r for r in rs if (r["p_yes"] >= 0.5) != (r["expected"] == "yes")]
        for r in misses:
            print(f"    miss {r['image']} expected {r['expected']} p_yes {r['p_yes']:.4f} top '{r['top_token']}'")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
