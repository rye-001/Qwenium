#!/usr/bin/env python3
"""VERDICT-IMG ground — does the model put its own box on a mark?
(docs/note-verdict-img-ground.md)

Against a running qwenium-server (--mmproj, Qwen3.6-35B-A3B UD-Q3_K_XL):

  ground DIR   per image of DIR/boxes.tsv (probe_verdict_img_render ground):
               the calibrated verdict (3 marks, /v1/verdict) and, per mark, the
               model's own box (/v1/chat/completions, greedy, no thinking).
               -> DIR/ground.tsv
  score DIR    box accuracy (IoU against the truth) under both coordinate
               readings: 0..1000 relative, and absolute px of the resized image.
  occlude DIR --coords rel|abs
               covers the model's box (and, as the control, a box of the same
               size elsewhere, off every drawn mark) and asks the verdict again.
               -> DIR/occlude.tsv
  report DIR   the occlusion summary.
  multi DIR --wording W
               "Locate every …": all boxes the model gives for the stamp, scored
               against every drawn stamp-like thing (stamp, badge, printed
               status). -> DIR/multi_<W>.tsv; `multi-score DIR --wording W`.
  route DIR    the served "where" (POST /v1/verdict, "where": true): per image a
               cold request without it, then the same image_id warm with and
               without it. Box accuracy, the box's extra time, and that the
               answers are the same with and without it. -> DIR/route.tsv

Nothing here decides a cut; it measures.
"""

import argparse
import base64
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
import zlib

PAGE_W, PAGE_H = 1024, 1440
FAMS = ["SIG", "STAMP", "DATE"]
VERDICT_QS = [
    {"id": "SIG", "mark": "signature", "subject": "delivery note", "box": "Received by"},
    {"id": "STAMP", "mark": "stamp", "subject": "delivery note"},
    {"id": "DATE", "mark": "date", "field": "delivery date"},
]
GROUND_PROMPT = {
    "SIG": "Locate the handwritten signature in the 'Received by' box in the image, "
           "output its bbox coordinates using JSON format.",
    "STAMP": "Locate the stamp in the image, output its bbox coordinates using JSON format.",
    "DATE": "Locate the handwritten delivery date in the image, "
            "output its bbox coordinates using JSON format.",
}
PAD = 8          # px added on every side before covering a box (pred and control alike)
OCCL_VARIANTS = {"all": FAMS, "lures": ["STAMP"]}


def post(port, path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def data_uri(path):
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def verdict(port, path):
    t = time.time()
    r = post(port, "/v1/verdict", {"image": data_uri(path), "questions": VERDICT_QS})
    p = {a["id"]: a["p"]["yes"] for a in r["answers"]}
    return p, r["image"]["grid"], time.time() - t


def ground_box(port, path, fam):
    t = time.time()
    r = post(port, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri(path)}},
            {"type": "text", "text": GROUND_PROMPT[fam]}]}],
        "max_tokens": 96, "temperature": 0, "enable_thinking": False})
    text = r["choices"][0]["message"]["content"]
    num = r"(-?\d+(?:\.\d+)?)"
    m = re.search(r"\[\s*" + r"\s*,\s*".join([num] * 4) + r"\s*\]", text)
    box = [float(v) for v in m.groups()] if m else None
    return text, box, time.time() - t


def read_truth(d):
    truth, meta = {}, {}
    for row in csv.DictReader(open(os.path.join(d, "boxes.tsv")), delimiter="\t"):
        meta[row["image"]] = row["variant"]
        if row["kind"] != "-":
            truth.setdefault(row["image"], {})[row["kind"]] = [float(row[k]) for k in ("x0", "y0", "x1", "y1")]
    return truth, meta


def to_page(box, coords, grid):
    if coords == "rel":
        sx, sy = PAGE_W / 1000.0, PAGE_H / 1000.0
    else:   # absolute px of the resized image the encoder saw (merged grid x 32)
        sx, sy = PAGE_W / (grid[0] * 32.0), PAGE_H / (grid[1] * 32.0)
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cmd_ground(a):
    truth, meta = read_truth(a.dir)
    out = os.path.join(a.dir, "ground.tsv")
    done = set()
    if os.path.exists(out):
        done = {(r["image"], r["fam"]) for r in csv.DictReader(open(out), delimiter="\t")}
    f = open(out, "a")
    if not done:
        f.write("image\tvariant\tfam\tp_yes\tgrid_w\tgrid_h\tbox\traw\tverdict_s\tground_s\n")
    for img in meta:
        if all((img, fam) in done for fam in FAMS):
            continue
        p, grid, vs = verdict(a.port, os.path.join(a.dir, img))
        for fam in FAMS:
            text, box, gs = ground_box(a.port, os.path.join(a.dir, img), fam)
            f.write(f"{img}\t{meta[img]}\t{fam}\t{p[fam]:.6f}\t{grid[0]}\t{grid[1]}\t"
                    f"{json.dumps(box)}\t{json.dumps(text)}\t{vs:.2f}\t{gs:.2f}\n")
            f.flush()
            print(f"{img} {fam} p={p[fam]:.3f} box={box} ({gs:.1f} s)", flush=True)


def read_ground(d):
    return list(csv.DictReader(open(os.path.join(d, "ground.tsv")), delimiter="\t", quoting=csv.QUOTE_NONE))


def best_iou(box, kinds):
    """(iou, kind) of the drawn mark or lure the box overlaps most."""
    best = (0.0, "-")
    for k, t in kinds.items():
        best = max(best, (iou(box, t), k))
    return best


def cmd_score(a):
    truth, _ = read_truth(a.dir)
    rows = read_ground(a.dir)
    for coords in ("rel", "abs"):
        print(f"== coords {coords}")
        for variant in ("all", "lures", "none"):
            for fam in FAMS:
                sel = [r for r in rows if r["variant"] == variant and r["fam"] == fam]
                if not sel:
                    continue
                ious, hits, onto, nobox = [], 0, {}, 0
                for r in sel:
                    box = json.loads(r["box"])
                    if box is None:
                        nobox += 1
                        continue
                    pb = to_page(box, coords, (int(r["grid_w"]), int(r["grid_h"])))
                    kinds = truth.get(r["image"], {})
                    own = iou(pb, kinds[fam]) if fam in kinds else None
                    if own is not None:
                        ious.append(own)
                        hits += own >= 0.5
                    v, k = best_iou(pb, kinds)
                    onto[k if v >= 0.3 else "nothing"] = onto.get(k if v >= 0.3 else "nothing", 0) + 1
                ious.sort()
                med = ious[len(ious) // 2] if ious else float("nan")
                own_txt = f"IoU>=0.5 {hits}/{len(ious)} median {med:.2f}" if ious else "no truth box"
                print(f"  {variant:5s} {fam:5s} n={len(sel)} no-box={nobox}  {own_txt}  lands on {onto}")


def control_box(img, fam, pb, kinds):
    """A box of pb's size, inside the page, off every drawn mark/lure and off pb."""
    rng = random.Random(zlib.crc32(f"{img}|{fam}".encode()))
    w, h = pb[2] - pb[0], pb[3] - pb[1]
    avoid = list(kinds.values()) + [pb]
    for _ in range(500):
        x0 = rng.uniform(20, PAGE_W - 20 - w)
        y0 = rng.uniform(20, PAGE_H - 20 - h)
        c = [x0, y0, x0 + w, y0 + h]
        if all(iou([c[0] - 15, c[1] - 15, c[2] + 15, c[3] + 15], t) == 0.0 for t in avoid):
            return c
    return None


def cmd_occlude(a):
    truth, _ = read_truth(a.dir)
    rows = read_ground(a.dir)
    base_p = {}
    for r in rows:
        base_p.setdefault(r["image"], {})[r["fam"]] = float(r["p_yes"])
    odir = os.path.join(a.dir, f"occl_{a.coords}")
    os.makedirs(odir, exist_ok=True)
    jobs, plan = [], []
    for r in rows:
        if r["fam"] not in OCCL_VARIANTS.get(r["variant"], []):
            continue
        box = json.loads(r["box"])
        if box is None:
            continue
        img, fam = r["image"], r["fam"]
        pb = to_page(box, a.coords, (int(r["grid_w"]), int(r["grid_h"])))
        pb = [max(0, pb[0] - PAD), max(0, pb[1] - PAD), min(PAGE_W, pb[2] + PAD), min(PAGE_H, pb[3] + PAD)]
        kinds = truth.get(img, {})
        stem = os.path.splitext(img)[0]
        for kind, b in (("pred", pb), ("control", control_box(img, fam, pb, kinds))):
            if b is None:
                print(f"{img} {fam}: no room for a control box", flush=True)
                continue
            outp = os.path.join(odir, f"{stem}_{fam}_{kind}.png")
            jobs.append(f"{os.path.join(a.dir, img)}\t{outp}\t{b[0]:.0f}\t{b[1]:.0f}\t{b[2]:.0f}\t{b[3]:.0f}")
            plan.append((img, r["variant"], fam, kind, b, outp))
    jf = os.path.join(odir, "jobs.tsv")
    open(jf, "w").write("\n".join(jobs) + "\n")
    subprocess.run([a.occlude_bin, jf], check=True)
    out = os.path.join(a.dir, f"occlude_{a.coords}.tsv")
    with open(out, "w") as f:
        f.write("image\tvariant\tfam\tkind\tbox\tcovers\t" + "\t".join(f"p0_{q}\tp1_{q}" for q in FAMS) + "\tverdict_s\n")
        for img, variant, fam, kind, b, outp in plan:
            p1, _, vs = verdict(a.port, outp)
            v, k = best_iou(b, truth.get(img, {}))
            cols = "\t".join(f"{base_p[img][q]:.6f}\t{p1[q]:.6f}" for q in FAMS)
            f.write(f"{img}\t{variant}\t{fam}\t{kind}\t{json.dumps([round(x) for x in b])}\t"
                    f"{k if v > 0 else '-'}\t{cols}\t{vs:.2f}\n")
            f.flush()
            print(f"{img} {fam} {kind}: p {base_p[img][fam]:.3f} -> {p1[fam]:.3f} ({vs:.1f} s)", flush=True)


def cmd_report(a):
    rows = list(csv.DictReader(open(os.path.join(a.dir, f"occlude_{a.coords}.tsv")), delimiter="\t"))
    for variant in ("all", "lures"):
        for fam in FAMS:
            for kind in ("pred", "control"):
                sel = [r for r in rows if r["variant"] == variant and r["fam"] == fam and r["kind"] == kind]
                if not sel:
                    continue
                p0 = [float(r[f"p0_{fam}"]) for r in sel]
                p1 = [float(r[f"p1_{fam}"]) for r in sel]
                below = sum(x < 0.5 for x in p1)
                drop = sorted(x - y for x, y in zip(p0, p1))
                others = [abs(float(r[f"p0_{q}"]) - float(r[f"p1_{q}"])) for r in sel for q in FAMS if q != fam]
                print(f"{variant:5s} {fam:5s} {kind:7s} n={len(sel):2d}  p<0.5 after: {below:2d}  "
                      f"drop median {drop[len(drop) // 2]:+.3f} min {drop[0]:+.3f} max {drop[-1]:+.3f}  "
                      f"other marks |dp| max {max(others):.3f}")


def verdict_where(port, path, where, image_id):
    t = time.time()
    r = post(port, "/v1/verdict", {"image": data_uri(path), "questions": VERDICT_QS,
                                   "where": where, "image_id": image_id})
    return r, time.time() - t


def cmd_route(a):
    truth, meta = read_truth(a.dir)
    out = open(os.path.join(a.dir, "route.tsv"), "w")
    out.write("image\tvariant\tfam\tanswer\tp_cold\tp_where\tbox\tiou_own\tlands_on\tcold_s\twarm_s\twhere_s\n")
    for img in meta:
        path = os.path.join(a.dir, img)
        cold, cs = verdict_where(a.port, path, False, img)
        warm, ws = verdict_where(a.port, path, False, img)
        wh, hs = verdict_where(a.port, path, True, img)
        kinds = truth.get(img, {})
        pc = {x["id"]: x for x in cold["answers"]}
        for x in wh["answers"]:
            box = x.get("where", {}).get("box") if "where" in x else None
            own, land = "", "-"
            if box:
                pb = [box[0] * PAGE_W, box[1] * PAGE_H, box[2] * PAGE_W, box[3] * PAGE_H]
                if x["id"] in kinds:
                    own = f"{iou(pb, kinds[x['id']]):.3f}"
                v, k = best_iou(pb, kinds)
                land = k if v >= 0.3 else "nothing"
            out.write(f"{img}\t{meta[img]}\t{x['id']}\t{x['answer']}\t{pc[x['id']]['p']['yes']:.6f}\t"
                      f"{x['p']['yes']:.6f}\t{json.dumps(box) if 'where' in x else '-'}\t{own}\t{land}\t"
                      f"{cs:.2f}\t{ws:.2f}\t{hs:.2f}\n")
        out.flush()
        print(f"{img}: cold {cs:.1f} s, warm {ws:.2f} s, warm+where {hs:.2f} s", flush=True)


MULTI_WORDING = {
    "stamps": "Locate every stamp in the image, output their bbox coordinates using JSON format.",
    "stamplike": ("Locate every stamp or stamp-like mark in the image, output their bbox coordinates "
                  "using JSON format."),
}
STAMPLIKE = ("STAMP", "BADGE", "STATUS")


def all_boxes(text):
    num = r"(-?\d+(?:\.\d+)?)"
    pat = r"\[\s*" + r"\s*,\s*".join([num] * 4) + r"\s*\]"
    return [[float(v) for v in m] for m in re.findall(pat, text)]


def cmd_multi(a):
    truth, meta = read_truth(a.dir)
    out = open(os.path.join(a.dir, f"multi_{a.wording}.tsv"), "w")
    out.write("image\tvariant\tboxes\traw\tseconds\n")
    for img in meta:
        t = time.time()
        r = post(a.port, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri(os.path.join(a.dir, img))}},
                {"type": "text", "text": MULTI_WORDING[a.wording]}]}],
            "max_tokens": 200, "temperature": 0, "enable_thinking": False})
        text = r["choices"][0]["message"]["content"]
        boxes = all_boxes(text)
        out.write(f"{img}\t{meta[img]}\t{json.dumps(boxes)}\t{json.dumps(text)}\t{time.time() - t:.2f}\n")
        out.flush()
        print(f"{img} {meta[img]}: {len(boxes)} boxes ({time.time() - t:.1f} s)", flush=True)


def cmd_multi_score(a):
    truth, _ = read_truth(a.dir)
    rows = list(csv.DictReader(open(os.path.join(a.dir, f"multi_{a.wording}.tsv")), delimiter="\t",
                               quoting=csv.QUOTE_NONE))
    for variant in ("all", "lures", "none"):
        for cond in ("c", "s"):
            sel = [r for r in rows if r["variant"] == variant and r["image"][0] == cond]
            found = {k: [0, 0] for k in STAMPLIKE}
            stray, nbox = 0, 0
            for r in sel:
                kinds = truth.get(r["image"], {})
                boxes = [to_page(b, "rel", (32, 45)) for b in json.loads(r["boxes"])]
                nbox += len(boxes)
                for k in STAMPLIKE:
                    if k in kinds:
                        found[k][1] += 1
                        found[k][0] += any(iou(b, kinds[k]) >= 0.3 for b in boxes)
                for b in boxes:
                    v, k = best_iou(b, kinds)
                    if v < 0.3 or k not in STAMPLIKE:
                        stray += 1
            have = "  ".join(f"{k} {f[0]}/{f[1]}" for k, f in found.items() if f[1])
            print(f"{variant:5s} {cond}  pages {len(sel)}  boxes {nbox}  found: {have or '-'}  "
                  f"boxes on no stamp-like thing: {stray}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ground", "score", "occlude", "report", "route", "multi", "multi-score"])
    ap.add_argument("--wording", choices=sorted(MULTI_WORDING), default="stamps")
    ap.add_argument("dir")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--coords", choices=["rel", "abs"], default="rel")
    ap.add_argument("--occlude-bin", default=".session-results/verdict_img_ground/occlude")
    a = ap.parse_args()
    {"ground": cmd_ground, "score": cmd_score, "occlude": cmd_occlude, "report": cmd_report,
     "route": cmd_route, "multi": cmd_multi, "multi-score": cmd_multi_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
