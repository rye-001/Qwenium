#!/usr/bin/env python3
"""VERDICT-IMG read — can the model read what is printed on a page?
(docs/note-verdict-img-read.md)

Against a running qwenium-server (--mmproj, Qwen3.6-35B-A3B UD-Q3_K_XL), on the
pages of `probe_verdict_img_render DIR read` (synthetic delivery notes and
invoices, read_truth.json = what is printed). Per page, through
/v1/chat/completions (greedy, no thinking):

  TABLE   the items table as JSON (qty, description, amount on invoices)
  TOTAL   the invoice total (invoices only)
  NUMBER  the document number

  run DIR [--only a,b]   -> DIR/read.tsv (raw answers)
  score DIR              exact-match scoring per row, per field, clean vs scan
  lines DIR --only a,b   the whole page as lines with boxes ("bbox_2d", 0..1000)
                         -> DIR/<out>; `lines-score` scores every printed run:
                         is its text in the transcript, does a line box cover it

Synthetic pages only. Nothing here decides a cut; it measures.
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import urllib.request


def post(port, path, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def data_uri(path):
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def prompts(t):
    doc = t["doc"]
    out = {}
    if doc == "invoice":
        out["TABLE"] = (f"Read the items table of this {doc}. Output only JSON: a list with one object per "
                        'row, with the keys "qty" (number), "description" (text) and "amount" '
                        "(number, without the currency).")
        out["TOTAL"] = "What is the total amount on this invoice? Answer with the number only, without the currency."
    else:
        out["TABLE"] = (f"Read the items table of this {doc}. Output only JSON: a list with one object per "
                        'row, with the keys "qty" (number) and "description" (text).')
    out["NUMBER"] = f"What is the document number of this {doc}? Answer with the number only."
    return out


def ask(port, path, text, max_tokens):
    t = time.time()
    r = post(port, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri(path)}},
            {"type": "text", "text": text}]}],
        "max_tokens": max_tokens, "temperature": 0, "enable_thinking": False})
    return r["choices"][0]["message"]["content"], time.time() - t


def cmd_run(a):
    truth = json.load(open(os.path.join(a.dir, "read_truth.json")))
    only = set(a.only.split(",")) if a.only else None
    out = os.path.join(a.dir, a.out)
    with open(out, "w") as f:
        f.write("image\tfield\tanswer\tseconds\n")
        for t in truth:
            if only and t["image"] not in only:
                continue
            for field, text in prompts(t).items():
                ans, s = ask(a.port, os.path.join(a.dir, t["image"]), text, 400 if field == "TABLE" else 32)
                f.write(f"{t['image']}\t{field}\t{json.dumps(ans)}\t{s:.2f}\n")
                f.flush()
                print(f"{t['image']} {field} ({s:.1f} s): {ans[:90]!r}", flush=True)


def norm_text(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def num(v):
    try:
        return float(str(v).replace("EUR", "").replace(",", "").strip())
    except ValueError:
        return None


def parse_table(ans):
    m = re.search(r"\[.*\]", ans, re.S)
    if not m:
        return None
    try:
        rows = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return rows if isinstance(rows, list) else None


def cmd_score(a):
    truth = {t["image"]: t for t in json.load(open(os.path.join(a.dir, "read_truth.json")))}
    rows = list(csv.DictReader(open(os.path.join(a.dir, a.out)), delimiter="\t", quoting=csv.QUOTE_NONE))
    stats = {}
    misses = []

    def bump(key, ok):
        n, k = stats.get(key, (0, 0))
        stats[key] = (n + 1, k + (1 if ok else 0))

    for r in rows:
        t = truth[r["image"]]
        cond = "clean" if r["image"].startswith("c_") else "scan"
        doc = "inv" if t["doc"] == "invoice" else "dn"
        ans = json.loads(r["answer"])
        # The 27B does not stop on its end-of-turn token: it runs on into
        # "<|im_start|>user" before the server stops it. Score the answer only.
        ans = re.split(r"<\|im_(?:start|end)\|>", ans)[0]
        if r["field"] == "TABLE":
            got = parse_table(ans)
            want = t["items"]
            whole = got is not None and len(got) == len(want)
            for i, w in enumerate(want):
                g = got[i] if got and i < len(got) and isinstance(got[i], dict) else {}
                ok_q = num(g.get("qty")) == float(w["qty"])
                ok_d = norm_text(g.get("description", "")) == norm_text(w["description"])
                ok_a = True
                if "amount" in w:
                    ga = num(g.get("amount"))
                    ok_a = ga is not None and abs(ga - float(w["amount"])) < 0.005
                    bump((cond, doc, "amount"), ok_a)
                bump((cond, doc, "qty"), ok_q)
                bump((cond, doc, "description"), ok_d)
                bump((cond, doc, "row"), ok_q and ok_d and ok_a)
                whole = whole and ok_q and ok_d and ok_a
                if not (ok_q and ok_d and ok_a):
                    misses.append(f"{r['image']} row {i + 1}: want {w}, got {g}")
            bump((cond, doc, "table"), whole)
        elif r["field"] == "TOTAL":
            g = num(ans)
            ok = g is not None and abs(g - float(t["total"])) < 0.005
            bump((cond, doc, "total"), ok)
            if not ok:
                misses.append(f"{r['image']} total: want {t['total']}, got {ans!r}")
        elif r["field"] == "NUMBER":
            ok = norm_text(ans) == norm_text(t["number"])
            loose = re.sub(r"[^0-9a-z]", "", ans.lower()) == re.sub(r"[^0-9a-z]", "", t["number"].lower())
            bump((cond, doc, "number"), ok)
            bump((cond, doc, "number_loose"), loose)
            if not ok:
                misses.append(f"{r['image']} number: want {t['number']!r}, got {ans!r}")
    for key in sorted(stats):
        n, k = stats[key]
        print(f"{key[0]:5s} {key[1]:3s} {key[2]:13s} {k}/{n}")
    secs = sorted(float(r["seconds"]) for r in rows if r["field"] == "TABLE")
    if secs:
        print(f"TABLE seconds median {secs[len(secs) // 2]:.1f}")
    print(f"\nmisses ({len(misses)}):")
    for m in misses:
        print("  " + m)


LINES_PROMPT = ('Read all the text on this page, line by line. Output only JSON: a list with one object '
                'per line, with the keys "text" and "bbox_2d".')
PAGE_W, PAGE_H = 1024, 1440


def cmd_lines(a):
    truth = json.load(open(os.path.join(a.dir, "read_truth.json")))
    only = set(a.only.split(",")) if a.only else None
    with open(os.path.join(a.dir, a.out), "w") as f:
        f.write("image\tanswer\tseconds\n")
        for t in truth:
            if only and t["image"] not in only:
                continue
            ans, s = ask(a.port, os.path.join(a.dir, t["image"]), LINES_PROMPT, 1500)
            f.write(f"{t['image']}\t{json.dumps(ans)}\t{s:.2f}\n")
            f.flush()
            print(f"{t['image']} lines ({s:.1f} s): {len(ans)} chars", flush=True)


def parse_lines(ans):
    """Every {...} object with a "text" and a 4-number "bbox_2d", in any key order."""
    out = []
    for m in re.finditer(r"\{[^{}]*\}", ans):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        b = o.get("bbox_2d")
        if not isinstance(o.get("text"), str) or not isinstance(b, list) or len(b) != 4:
            continue
        try:
            b = [float(v) for v in b]
        except (TypeError, ValueError):
            continue
        out.append((o["text"], [b[0] * PAGE_W / 1000, b[1] * PAGE_H / 1000, b[2] * PAGE_W / 1000, b[3] * PAGE_H / 1000]))
    return out


def cover(run, box):
    ix = max(0.0, min(run[2], box[2]) - max(run[0], box[0]))
    iy = max(0.0, min(run[3], box[3]) - max(run[1], box[1]))
    area = (run[2] - run[0]) * (run[3] - run[1])
    return ix * iy / area if area > 0 else 0.0


def cmd_lines_score(a):
    truth = {t["image"]: t for t in json.load(open(os.path.join(a.dir, "read_truth.json")))}
    rows = list(csv.DictReader(open(os.path.join(a.dir, a.out)), delimiter="\t", quoting=csv.QUOTE_NONE))
    tot = {"runs": 0, "text": 0, "boxed": 0, "lines": 0, "stray": 0}
    missed = []
    for r in rows:
        t = truth[r["image"]]
        lines = parse_lines(json.loads(r["answer"]))
        tot["lines"] += len(lines)
        for txt, box in lines:
            if not any(cover(l["box"], box) > 0 for l in t["lines"]):
                tot["stray"] += 1
        for l in t["lines"]:
            tot["runs"] += 1
            want = norm_text(l["text"])
            hits = [box for txt, box in lines if want in norm_text(txt)]
            if hits:
                tot["text"] += 1
                if max(cover(l["box"], b) for b in hits) >= 0.5:
                    tot["boxed"] += 1
                else:
                    missed.append(f"{r['image']}: box misses {l['text']!r}")
            else:
                missed.append(f"{r['image']}: text missing {l['text']!r}")
    secs = sorted(float(r["seconds"]) for r in rows)
    print(f"pages {len(rows)}  printed runs {tot['runs']}  text found {tot['text']}  "
          f"text found and box covers >= half {tot['boxed']}  model lines {tot['lines']}  "
          f"lines on no printed run {tot['stray']}  seconds median {secs[len(secs) // 2]:.1f}")
    for m in missed[:40]:
        print("  " + m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score", "lines", "lines-score"])
    ap.add_argument("dir")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="read.tsv")
    a = ap.parse_args()
    {"run": cmd_run, "score": cmd_score, "lines": cmd_lines, "lines-score": cmd_lines_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
