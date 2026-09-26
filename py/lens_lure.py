"""LURE — the lure/clean split on the LANDED choice head.

The 43.8% on record is the INCUMBENT LOCATE pair read through
py/lens_decide1.py's own recipe (bare-string keys, peak, max aggregation,
rotations). Comparing that to the landed choice head would vary the head AND
the recipe at once. So both heads run HERE through one identical harness --
question vocabulary, mean aggregation, mass summed over the body -- and only
`head` changes. The locate arm is uncalibrated by construction (question+mean
is off-recipe for locate); that is the point of it, it is a control, not a claim.
"""
import json, urllib.request, sys, itertools, collections
URL="http://127.0.0.1:18190/v1/locate"
C=json.load(open("/Users/sfadaei/dev/jprojs/fas/github/qwen-inference/py/corpora/lens_corpora.json",encoding="utf-8"))
OPTS=C["choice_options"]

def ask(doc, order, head):
    kv=[{"id":o["name"],"question":o["desc"]} for o in order]
    b=json.dumps({"document":doc,"key_vocabulary":kv,"top_k":64,
                  "key_aggregation":"mean","head":head}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request(URL,b,
        {"Content-Type":"application/json"}),timeout=900))
    sc={o["name"]: sum(h["mass"] for h in r["hits"].get(o["name"],[])) for o in order}
    return max(sc,key=sc.get), sc, r["uncalibrated"]

def lure_of(d):  return bool(d.get("lure", d.get("lure_by_tag")))

def run(docs, head, rotations=1):
    rows=[]
    for d in docs:
        votes=collections.Counter()
        for k in range(rotations):
            order=OPTS[k:]+OPTS[:k]
            w,_,unc=ask(d["document"],order,head)
            votes[w]+=1
        got=votes.most_common(1)[0][0]
        rows.append((d, got, got==d["label"], lure_of(d)))
    return rows

def report(tag, rows):
    n=len(rows); acc=sum(r[2] for r in rows)/n
    lu=[r for r in rows if r[3]]; cl=[r for r in rows if not r[3]]
    en=[r for r in rows if not r[0]["de"]]; de=[r for r in rows if r[0]["de"]]
    print(f"  {tag:34s} all {100*acc:5.1f}%  "
          f"clean {100*sum(r[2] for r in cl)/len(cl):5.1f}% (n={len(cl)})  "
          f"LURE {100*sum(r[2] for r in lu)/len(lu):5.1f}% (n={len(lu)})  "
          f"| EN {100*sum(r[2] for r in en)/len(en):5.1f}%  DE {100*sum(r[2] for r in de)/len(de):5.1f}%")
    return acc, sum(r[2] for r in lu)/len(lu)

print("="*104)
print("LURE — landed choice head (L11 h=3) vs incumbent locate pair (L11 h=6), ONE harness")
print("="*104)
for corpus in ("choice_py","choice_cpp"):
    docs=C[corpus]
    print(f"\n{corpus}  (n={len(docs)}, lures={sum(1 for d in docs if lure_of(d))})"
          + ("   <- authentic lure text, the corpus the 43.8% came from" if corpus=="choice_py" else
             "   <- the corpus DECIDEHEAD swept -> the 92.5%"))
    for head in ("choice","locate"):
        report(f"head={head}", run(docs,head))

print("\n"+"="*104)
print("ROTATION CONTROL — choice head, choice_py, 4 rotations majority-vote")
print("(LOCABSENT measured that a key's score depends on its POSITION in the request)")
print("="*104)
report("head=choice, 4 rotations", run(C["choice_py"],"choice",rotations=4))
