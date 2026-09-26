#!/usr/bin/env python3
"""LAYACHECK — run OUR corpora through Laya, to audit OUR corpora.

THIS IS NOT A BAKE-OFF. Convai's Laya is an open-weights encoder decision
model (ModernBERT-large / mmBERT-base + a trained decision head) built to
answer TypeSafe Jev's three typed primitives: `choice`, `score`, `noul`.
Those are three of our four lens jobs, so it is the only independent
instrument that can answer a question we cannot answer about ourselves:

    Are the corpora our head numbers were measured on too easy?

Every document in decide_choice_corpus(), score_corpus_extended() and
qdocs_messy_corpus() was written by the same author as the probe that scores
them. We already caught that conflict of interest once (SCOREHEAD: 24 of the
48 score documents were self-authored, which forced a corpus-origin held-out
axis). An independently trained model reading the same documents is a second
opinion on the DOCUMENTS, not a competitor.

THE TEST IS ASYMMETRIC, AND THAT IS THE POINT. Laya's own model card reports
its base checkpoints at 0.362 on typed-decisions against a 0.461 majority-class
baseline — near chance, and they say so plainly ("a fast base to specialise,
not a zero-shot decision engine"). So:

  * A Laya LOSS proves nothing. It was predicted by their own card. It must
    NOT be reported as "we beat Laya".
  * A Laya WIN, or a near-tie, is informative, and it is informative AGAINST
    US: it would mean our corpus is easy and our rates are cheap.

Only the direction that hurts us carries signal. That is why it is worth the
compute.

FAIRNESS CHOICES, all deliberately in Laya's favour:
  * `typed-decisions` is the strong arm (0.766 on their benchmark), not the
    near-chance root checkpoint.
  * German is sent to `multilingual` EXPLICITLY rather than through
    Router(), because the router dispatches on SCRIPT and German is Latin,
    so the router would hand our DE half to the English checkpoint.
  * head_max_len / max_len are raised so no document is silently truncated;
    truncation would manufacture a loss.

WHAT IS NOT MEASURED HERE: latency (different hardware and 20x the
parameters — settled, and not interesting), `locate`, and coverage/omission.
Laya has no equivalent of the last two, so there is nothing to compare.

    python3 py/laya_cross_check.py --checkpoint typed-decisions
"""
import argparse, itertools, json, os, pathlib, statistics, sys, time

os.environ.setdefault("USE_TF", "0")  # their card: TF's abseil can deadlock load

ROOT = pathlib.Path(__file__).resolve().parent
CORPORA = ROOT / "corpora" / "lens_corpora.json"

# Our landed numbers, for context in the printout ONLY. They are NOT a bar
# Laya is being asked to clear — different mechanism, and the comparison that
# matters is the one about corpus difficulty.
OURS = {
    "choice_cpp": "9B L11 h=3 = 92.5%   |  27B L39 h=7 = 97.5%",
    "choice_py":  "9B incumbent locate pair = 75.0%  (95.8% clean / 43.8% lure)",
    "score":      "9B L19 h=11 = concordance 1.0000, exact 66.7%, within-1 89.6%",
    "absent":     "9B L19 h=10 = AUC 0.9948, 89.6% of absences at zero false accusations",
}


# ── metrics, defined to match the probe's definitions exactly ───────────────
def concordance(pairs):
    """Over every pair of documents on DIFFERENT levels, is the higher one
    scored higher. Threshold-free, and the metric SCOREHEAD selected on."""
    ok = tot = 0
    for (la, sa), (lb, sb) in itertools.combinations(pairs, 2):
        if la == lb:
            continue
        tot += 1
        if (sa > sb) == (la > lb):
            ok += 1
        elif sa == sb:
            ok += 0.5
    return ok / tot if tot else float("nan")


def auc(pos, neg):
    """P(a random positive outranks a random negative), ties at 0.5."""
    if not pos or not neg:
        return float("nan")
    tot = ok = 0
    for p in pos:
        for n in neg:
            tot += 1
            ok += 1.0 if p > n else 0.5 if p == n else 0.0
    return ok / tot


def caught_at_zero_false(present, absent):
    """Our shipped operating point: put the threshold just under the WEAKEST
    present key, so no present key is ever called absent, and report how many
    absences still fall below it."""
    if not present or not absent:
        return float("nan")
    thr = min(present)
    return sum(1 for a in absent if a < thr) / len(absent)


def pct(x):
    return "  n/a " if x != x else f"{100*x:5.1f}%"


# ── laya plumbing ──────────────────────────────────────────────────────────
def load_agent(subfolder, max_len, head_max_len):
    import laya
    t0 = time.time()
    agent = (laya.load("convaiinnovations/laya") if subfolder in (None, "", "root")
             else laya.load("convaiinnovations/laya", subfolder=subfolder))
    try:
        agent.cfg["max_len"] = max(agent.cfg.get("max_len", 0), max_len)
        agent.cfg["head_max_len"] = max(agent.cfg.get("head_max_len", 0), head_max_len)
    except Exception as e:
        print(f"  ! could not raise token budget ({e}); using defaults", flush=True)
    print(f"  loaded {subfolder or 'root'} in {time.time()-t0:.1f}s "
          f"(max_len={agent.cfg.get('max_len')}, head_max_len={agent.cfg.get('head_max_len')})",
          flush=True)
    return agent


def ask(agent, document, questions):
    res = agent.predict({"body": document}, questions)
    return res["answers"]


# ── the four tests ─────────────────────────────────────────────────────────
def test_choice(agent, docs, options, key):
    q = {"department": {"type": "choice",
                        "instructions": "Which category does this document belong to?",
                        "criteria": {o["name"]: o["desc"] for o in options}}}
    rows = []
    for d in docs:
        got = ask(agent, d["document"], q)["department"]["choice"]
        rows.append((d, got, got == d["label"]))
    return rows


def test_score(agent, docs, options):
    q = {"severity": {"type": "score",
                      "instructions": "How severe is this ticket?",
                      "criteria": [o["desc"] for o in options]}}
    rows = []
    for d in docs:
        a = ask(agent, d["document"], q)["severity"]
        raw = a["score"] if isinstance(a, dict) else a
        rows.append((d, float(raw)))
    return rows


def test_absent(agent, docs, absent_concepts):
    rows = []
    for d in docs:
        keys = list(d["present"]) + list(absent_concepts)
        q = {k: {"type": "noul",
                 "instructions": f"Does this document state the {k.replace('_',' ')}?"}
             for k in keys}
        a = ask(agent, d["document"], q)
        for k in keys:
            v = a[k]
            p = float(v["noul"] if isinstance(v, dict) else v)
            rows.append((d, k, k in d["present"], p))
    return rows


# ── reporting ──────────────────────────────────────────────────────────────
def split(rows, pick):
    return {"ALL": rows, "EN": [r for r in rows if not pick(r)["de"]],
            "DE": [r for r in rows if pick(r)["de"]]}


def report_choice(name, rows, has_lure):
    print(f"\n── {name} ──  ours: {OURS[name]}")
    for lang, rs in split(rows, lambda r: r[0]).items():
        if not rs:
            continue
        acc = sum(1 for _, _, ok in rs) and sum(ok for _, _, ok in rs) / len(rs)
        line = f"  {lang:3s} n={len(rs):3d}  accuracy {pct(acc)}"
        if has_lure and lang == "ALL":
            lu = [r for r in rs if r[0].get("lure") or r[0].get("lure_by_tag")]
            cl = [r for r in rs if not (r[0].get("lure") or r[0].get("lure_by_tag"))]
            line += (f"   |  lure {pct(sum(o for _,_,o in lu)/len(lu))} (n={len(lu)})"
                     f"  clean {pct(sum(o for _,_,o in cl)/len(cl))} (n={len(cl)})")
        print(line)
    wrong = [(r[0]["tag"], r[0]["label"], r[1]) for r in rows if not r[2]]
    if wrong:
        print(f"  misses ({len(wrong)}): " +
              ", ".join(f"{t}:{w}->{g}" for t, w, g in wrong[:14]) +
              (" ..." if len(wrong) > 14 else ""))


def report_score(rows):
    print(f"\n── score ──  ours: {OURS['score']}")
    for lang, rs in split(rows, lambda r: r[0]).items():
        if not rs:
            continue
        pairs = [(d["level"], s) for d, s in rs]
        lo = min(s for _, s in pairs); hi = max(s for _, s in pairs)
        # Their score is an expected value over criteria 0..k-1 (the same
        # fractional construction ours uses); our levels are 1..4. Two
        # readings, because the choice of map is exactly the thing our own
        # score head is uncalibrated about:
        #   direct — round(s)+1, no fit at all. The honest reading.
        #   fitted — min-max onto 1..4, which uses the TEST SET to place the
        #            scale. Generous to Laya; it is the affine fix we have
        #            NOT granted ourselves (that would be our first fitted
        #            constant, and it is a user decision).
        # Concordance is scale-free and unaffected by either.
        direct = lambda s: min(4, max(1, int(round(s)) + 1))
        fitted = (lambda s: 1) if hi == lo else \
                 (lambda s: min(4, max(1, int(round(1 + 3 * (s - lo) / (hi - lo))))))
        def acc(f):
            ex = sum(1 for d, s in rs if f(s) == d["level"]) / len(rs)
            w1 = sum(1 for d, s in rs if abs(f(s) - d["level"]) <= 1) / len(rs)
            mae = statistics.fmean(abs(f(s) - d["level"]) for d, s in rs)
            return ex, w1, mae
        ed, wd, md = acc(direct)
        ef, wf, mf = acc(fitted)
        print(f"  {lang:3s} n={len(rs):3d}  concordance {concordance(pairs):.4f}"
              f"   raw {lo:.2f}..{hi:.2f}")
        print(f"      direct  exact {pct(ed)}  within-1 {pct(wd)}  MAE {md:.3f}")
        print(f"      fitted  exact {pct(ef)}  within-1 {pct(wf)}  MAE {mf:.3f}"
              f"   (min-max fitted ON the test set — generous)")
    for tag, sub in (("originals", False), ("novel", True)):
        rs = [r for r in rows if r[0]["novel"] is sub]
        if rs:
            print(f"      {tag:9s} concordance "
                  f"{concordance([(d['level'], s) for d, s in rs]):.4f} (n={len(rs)})")


def report_absent(rows):
    print(f"\n── absent / noul ──  ours: {OURS['absent']}")
    for lang, rs in split(rows, lambda r: r[0]).items():
        if not rs:
            continue
        pos = [p for _, _, is_p, p in rs if is_p]
        neg = [p for _, _, is_p, p in rs if not is_p]
        print(f"  {lang:3s} present={len(pos):3d} absent={len(neg):3d}  "
              f"AUC {auc(pos,neg):.4f}   caught at zero false accusations "
              f"{pct(caught_at_zero_false(pos,neg))}")
    per = {}
    for _, k, is_p, p in rows:
        if not is_p:
            per.setdefault(k, []).append(p)
    allpos = [p for _, _, is_p, p in rows if is_p]
    print("  per absent concept (mean prob, lower is better):  " +
          "  ".join(f"{k}={statistics.fmean(v):.3f}" for k, v in per.items()))
    print(f"  weakest present key scores {min(allpos):.3f}" if allpos else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="typed-decisions",
                    choices=["typed-decisions", "multilingual", "root"])
    ap.add_argument("--de-checkpoint", default="multilingual",
                    help="checkpoint for the German half (their router would "
                         "send Latin-script German to the English model)")
    ap.add_argument("--tests", default="choice,score,absent")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not CORPORA.exists():
        sys.exit(f"missing {CORPORA} — run the extractor first")
    C = json.load(open(CORPORA, encoding="utf-8"))
    want = set(a.tests.split(","))

    print("=" * 78)
    print("LAYACHECK — our corpora through Laya. An audit of OUR corpora.")
    print("A Laya loss proves nothing (their card: base is near chance zero-shot).")
    print("Only a Laya win or near-tie is informative, and it is informative against us.")
    print("=" * 78)

    agents = {}
    def agent_for(de):
        name = a.de_checkpoint if de else a.checkpoint
        if name not in agents:
            agents[name] = load_agent(name, a.max_len, a.head_max_len)
        return agents[name]

    def run(docs, fn, *args):
        out = []
        for d in docs:
            out += fn(agent_for(d["de"]), [d], *args)
        return out

    results = {}
    if "choice" in want:
        for key in ("choice_cpp", "choice_py"):
            rows = run(C[key], test_choice, C["choice_options"], key)
            report_choice(key, rows, has_lure=True)
            results[key] = [{"tag": d["tag"], "want": d["label"], "got": g, "ok": ok}
                            for d, g, ok in rows]
    if "score" in want:
        rows = run(C["score"], test_score, C["score_options"])
        report_score(rows)
        results["score"] = [{"tag": d["tag"], "level": d["level"], "raw": s} for d, s in rows]
    if "absent" in want:
        rows = run(C["absent"], test_absent, C["absent_concepts"])
        report_absent(rows)
        results["absent"] = [{"tag": d["tag"], "key": k, "present": p, "prob": v}
                             for d, k, p, v in rows]

    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
        print(f"\nraw results -> {a.out}")


if __name__ == "__main__":
    main()
