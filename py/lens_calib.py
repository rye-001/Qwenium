"""CALIB (docs/note-lens-laya-cross-check.md §9) — does content-free calibration improve the three decision heads?

Arms (all on the LANDED heads, question vocab, mean aggregation, mass summed
over the whole document body):
  raw    the shipped readout (baseline; must reproduce the recorded numbers)
  docCF  ICR-style, DOCUMENT side: a content-free key "N/A" rides in the same
         prompt; every key's mass minus the N/A key's mass on the same doc.
         Removes what the document draws regardless of the question.
  qNA    CBU-style, QUERY side: the same keys asked of the document "N/A";
         each key's mass divided by its own content-free mass.
  qNEU   as qNA, but the content-free document is a length-matched neutral text.
  loo    reference-set prior: each key's mass divided by its mean over the
         OTHER documents of the corpus (needs a reference set; an upper bound
         for prior-style calibration, not a runtime recipe).
"""
import json, urllib.request, itertools, random, sys, zlib, statistics as st
URL = "http://127.0.0.1:18190/v1/locate"
C = json.load(open(__import__("os").path.join(__import__("os").path.dirname(__file__), "corpora", "lens_corpora.json"), encoding="utf-8"))
NEUTRAL = ("The committee met on a quiet morning and reviewed the agenda in order. "
           "Several people spoke briefly, a few notes were taken, and the room was "
           "tidied afterwards. Nothing unusual happened and everyone left on time. ")
EPS = 1e-9
COV = []

def neutral_of(n_chars):
    s = NEUTRAL * (1 + n_chars // len(NEUTRAL))
    return s[:max(n_chars, 40)].rsplit(" ", 1)[0] + "."

def masses(doc, kv, head):
    b = json.dumps({"document": doc, "key_vocabulary": kv, "top_k": 64,
                    "key_aggregation": "mean", "head": head}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, b, {"Content-Type": "application/json"}), timeout=900))
    n = r["doc_hi"] - r["doc_lo"]
    for k in kv:
        cov = sum(h["tok_hi"] - h["tok_lo"] for h in r["hits"].get(k["id"], []))
        COV.append(cov / n if n else 1.0)
    return {k["id"]: sum(h["mass"] for h in r["hits"].get(k["id"], [])) for k in kv}

def arms(doc, kv, head):
    ids = [k["id"] for k in kv]
    raw = masses(doc, kv, head)
    wcf = masses(doc, kv + [{"id": "cf__", "question": "N/A"}], head)
    na  = masses("N/A", kv, head)
    neu = masses(neutral_of(len(doc)), kv, head)
    out = {"raw":   {k: raw[k] for k in ids},
           "docCF": {k: wcf[k] - wcf["cf__"] for k in ids},
           "qNA":   {k: raw[k] / (na[k] + EPS) for k in ids},
           "qNEU":  {k: raw[k] / (neu[k] + EPS) for k in ids}}
    return out, raw

def add_loo(rows, ids_of):
    # rows: list of dicts with "arms" and "raw"; prior per key id from OTHER docs
    for i, r in enumerate(rows):
        others = [o["raw"] for j, o in enumerate(rows) if j != i]
        r["arms"]["loo"] = {k: r["raw"][k] / (st.mean(o[k] for o in others if k in o) + EPS)
                            for k in ids_of(r)}

ARMS = ["raw", "docCF", "qNA", "qNEU", "loo"]

# ── metrics ──
def concordance(pairs):
    ok = tot = 0
    for (la, sa), (lb, sb) in itertools.combinations(pairs, 2):
        if la == lb: continue
        tot += 1
        ok += 1 if (sa > sb) == (la > lb) and sa != sb else 0.5 if sa == sb else 0
    return ok / tot

def auc(pos, neg):
    return sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg) / (len(pos) * len(neg))

def caught0(pos, neg):
    t = min(pos); return sum(1 for a in neg if a < t) / len(neg)

def langs(rows):
    return [("EN", [r for r in rows if not r["de"]]), ("DE", [r for r in rows if r["de"]]), ("ALL", rows)]

# ── choice ──
def run_choice(name):
    opts = C["choice_options"]
    kv = [{"id": o["name"], "question": o["desc"]} for o in opts]
    rows = []
    for d in C[name]:
        a, raw = arms(d["document"], kv, "choice")
        rows.append({"de": d["de"], "label": d["label"],
                     "lure": bool(d.get("lure", d.get("lure_by_tag"))), "arms": a, "raw": raw})
    add_loo(rows, lambda r: [o["name"] for o in opts])
    print(f"\nCHOICE {name}  n={len(rows)}  lures={sum(r['lure'] for r in rows)}")
    for arm in ARMS:
        line = f"  {arm:6s}"
        for lg, rs in langs(rows):
            acc = sum(max(r["arms"][arm], key=r["arms"][arm].get) == r["label"] for r in rs) / len(rs)
            lu = [r for r in rs if r["lure"]]
            la = sum(max(r["arms"][arm], key=r["arms"][arm].get) == r["label"] for r in lu) / max(1, len(lu))
            line += f" | {lg} {100*acc:5.1f}% lure {100*la:5.1f}%"
        print(line)
    return rows

# ── score ──
def expected_level(sc, ids):
    v = [max(0.0, sc[k]) for k in ids]          # simplex needs non-negative
    t = sum(v)
    return 1.5 if t <= 0 else sum(i * x for i, x in enumerate(v)) / t

def run_score():
    opts = C["score_options"]; ids = [o["name"] for o in opts]
    kv = [{"id": o["name"], "question": o["desc"]} for o in opts]
    rows = []
    for d in C["score"]:
        a, raw = arms(d["document"], kv, "score")
        rows.append({"de": d["de"], "level": d["level"], "arms": a, "raw": raw})
    add_loo(rows, lambda r: ids)
    print(f"\nSCORE  n={len(rows)}   (concordance | exact via round(E)+1 | within-1 | mean E per true level 1..4)")
    for arm in ARMS:
        for lg, rs in langs(rows):
            e = [(r["level"], expected_level(r["arms"][arm], ids)) for r in rs]
            ex = sum(min(4, max(1, round(s) + 1)) == l for l, s in e) / len(e)
            w1 = sum(abs(min(4, max(1, round(s) + 1)) - l) <= 1 for l, s in e) / len(e)
            means = [st.mean([s for l, s in e if l == L]) for L in (1, 2, 3, 4)]
            print(f"  {arm:6s} {lg:3s} conc {concordance(e):.4f}  exact {100*ex:5.1f}%  w1 {100*w1:5.1f}%"
                  f"  E: " + " ".join(f"{m:.2f}" for m in means))
    return rows

# ── absent ──
def run_absent():
    rows = []
    for d in C["absent"]:
        keys = list(d["present"]) + list(C["absent_concepts"])
        random.Random(zlib.crc32(d["tag"].encode())).shuffle(keys)
        kv = [{"id": k, "question": k} for k in keys]
        a, raw = arms(d["document"], kv, "absent")
        rows.append({"de": d["de"], "present": set(d["present"]), "keys": keys, "arms": a, "raw": raw})
    add_loo(rows, lambda r: r["keys"])
    print(f"\nABSENT  n_docs={len(rows)}   (AUC present>absent | caught at zero false accusations)")
    for arm in ARMS:
        line = f"  {arm:6s}"
        for lg, rs in langs(rows):
            pos = [r["arms"][arm][k] for r in rs for k in r["keys"] if k in r["present"]]
            neg = [r["arms"][arm][k] for r in rs for k in r["keys"] if k not in r["present"]]
            line += f" | {lg} AUC {auc(pos, neg):.4f} c0 {100*caught0(pos, neg):5.1f}%"
        print(line)
    return rows

# ── ladder: the fair test for QUERY-side calibration — rung LENGTH skewed ──
SHORT = ["cosmetic, can wait", "minor, workaround exists", "serious, many users blocked", "critical outage"]
LONG = ["a routine request with no customer impact at all where nothing is broken for anyone and the work can safely wait until someone has spare time next week or even later",
        "a minor problem where a few users are affected and they are somewhat inconvenienced but a reasonable workaround is available so they can keep working for now",
        "a serious problem where many users are affected and they are blocked from doing their work because there is no workaround available to them at the moment",
        "a critical failure where the whole service is down for every customer and orders cannot be placed so the company is losing revenue every single minute it continues"]
def run_ladder():
    ladders = {"fwd (L1 short .. L4 long)": [SHORT[0], SHORT[1], LONG[2], LONG[3]],
               "inv (L1 long .. L4 short)": [LONG[0], LONG[1], SHORT[2], SHORT[3]]}
    ids = ["level_1", "level_2", "level_3", "level_4"]
    res = {}
    for name, descs in ladders.items():
        kv = [{"id": i, "question": q} for i, q in zip(ids, descs)]
        rows = []
        for d in C["score"]:
            a, raw = arms(d["document"], kv, "score")
            rows.append({"de": d["de"], "level": d["level"], "arms": a, "raw": raw})
        add_loo(rows, lambda r: ids)
        res[name] = rows
    print("\nLADDER — rung length skewed both ways; a debiasing arm makes fwd and inv read alike")
    for arm in ARMS:
        for name, rows in res.items():
            for lg, rs in langs(rows)[:2]:
                e = [(r["level"], expected_level(r["arms"][arm], ids)) for r in rs]
                am = sum(ids.index(max(r["arms"][arm], key=r["arms"][arm].get)) + 1 == r["level"] for r in rs) / len(rs)
                means = [st.mean([s for l, s in e if l == L]) for L in (1, 2, 3, 4)]
                print(f"  {arm:6s} {name[:3]} {lg} conc {concordance(e):.4f}  argmax-exact {100*am:5.1f}%  E: "
                      + " ".join(f"{m:.2f}" for m in means))

if __name__ == "__main__":

    which = sys.argv[1:] or ["choice", "score", "absent"]
    if "choice" in which:
        run_choice("choice_py"); run_choice("choice_cpp")
    if "score" in which:  run_score()
    if "absent" in which: run_absent()
    if "ladder" in which: run_ladder()
    print(f"\nspan coverage of the doc body: min {min(COV):.3f}  mean {sum(COV)/len(COV):.3f}  (<1 = summed mass is partial)")

