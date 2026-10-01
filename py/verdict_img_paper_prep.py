#!/usr/bin/env python3
"""Prepare the real-paper captures for probe-verdict-img (docs/note-verdict-img-probe.md §9).

Reads temp/paper/NN_photo.* and NN_scan.* (any format macOS `sips` reads,
HEIC included), writes temp/paper/img/NN_{photo,scan}.jpg with the long side
at 1440 px — the same token budget as the synthetic arms — and
temp/paper/img/manifest.tsv from temp/paper/truth.tsv. Stdlib + sips only;
nothing leaves the machine.

  python3 py/verdict_img_paper_prep.py [temp/paper]
"""
import csv
import os
import re
import struct
import subprocess
import sys

# EXIF orientation -> clockwise degrees for `sips -r`. Phones store the sensor's
# landscape pixels plus this tag; the engine's loader (stb_image) ignores EXIF,
# so an unrotated photo of an upright page reaches the model sideways.
ROTATE = {1: 0, 3: 180, 6: 90, 8: 270}


def exif_orientation(path):
    """The JPEG's EXIF Orientation tag (0x0112), or 1 when there is none."""
    d = open(path, "rb").read(256 * 1024)
    i = d.find(b"Exif\x00\x00")
    if i < 0:
        return 1
    t = i + 6
    bo = "<" if d[t:t + 2] == b"II" else ">"
    ifd = struct.unpack(bo + "I", d[t + 4:t + 8])[0]
    n = struct.unpack(bo + "H", d[t + ifd:t + ifd + 2])[0]
    for k in range(n):
        e = t + ifd + 2 + 12 * k
        tag = struct.unpack(bo + "H", d[e:e + 2])[0]
        if tag == 0x0112:
            return struct.unpack(bo + "H", d[e + 8:e + 10])[0]
    return 1


def reset_orientation(path):
    """Set the EXIF Orientation tag to 1 in place: `sips -r` turns the pixels but
    keeps the tag, so a viewer would turn the page a second time."""
    with open(path, "r+b") as f:
        d = f.read(256 * 1024)
        i = d.find(b"Exif\x00\x00")
        if i < 0:
            return
        t = i + 6
        bo = "<" if d[t:t + 2] == b"II" else ">"
        ifd = struct.unpack(bo + "I", d[t + 4:t + 8])[0]
        n = struct.unpack(bo + "H", d[t + ifd:t + ifd + 2])[0]
        for k in range(n):
            e = t + ifd + 2 + 12 * k
            if struct.unpack(bo + "H", d[e:e + 2])[0] == 0x0112:
                f.seek(e + 8)
                f.write(struct.pack(bo + "H", 1))
                return


def size(path):
    out = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", path],
                         check=True, capture_output=True, text=True).stdout
    w = int(re.search(r"pixelWidth: (\d+)", out).group(1))
    h = int(re.search(r"pixelHeight: (\d+)", out).group(1))
    return w, h


def main(root):
    out = os.path.join(root, "img")
    os.makedirs(out, exist_ok=True)
    truth = list(csv.DictReader(open(os.path.join(root, "truth.tsv")), delimiter="\t"))
    captures = []
    for name in sorted(os.listdir(root)):
        m = re.match(r"^(\d\d)_(photo|scan)\.[A-Za-z0-9]+$", name)
        if not m:
            continue
        dst = f"{m.group(1)}_{m.group(2)}.jpg"
        src = os.path.join(root, name)
        o = exif_orientation(src) if name.lower().endswith((".jpg", ".jpeg")) else 1
        if o not in ROTATE:
            raise SystemExit(f"verdict_img_paper_prep: {name}: EXIF orientation expected one of "
                             f"{sorted(ROTATE)}, actual {o} (mirrored captures are not handled)")
        cmd = ["sips", "-s", "format", "jpeg"]
        if ROTATE[o]:
            cmd += ["-r", str(ROTATE[o])]
        subprocess.run(cmd + ["-Z", "1440", src, "--out", os.path.join(out, dst)],
                       check=True, capture_output=True)
        reset_orientation(os.path.join(out, dst))
        w, h = size(os.path.join(out, dst))
        if w >= h:   # every sheet is portrait A4
            raise SystemExit(f"verdict_img_paper_prep: {name}: expected a portrait page after rotation, "
                             f"actual {w}x{h} (EXIF orientation {o})")
        captures.append((m.group(1), dst))
    if not captures:
        raise SystemExit(f"verdict_img_paper_prep: captures expected as NN_photo.* / NN_scan.* in {root}, actual none")
    rows = []
    for sheet, img in captures:
        qs = [t for t in truth if t["sheet"] == sheet]
        if len(qs) != 3:
            raise SystemExit(f"verdict_img_paper_prep: sheet {sheet} expected 3 truth rows, actual {len(qs)}")
        rows += [f"{img}\t{int(sheet)}\t{t['family']}\t{t['question']}\t{t['expected']}" for t in qs]
    open(os.path.join(out, "manifest.tsv"), "w").write("\n".join(rows) + "\n")
    print(f"{len(captures)} captures, {len(rows)} questions -> {out}/manifest.tsv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "temp/paper")
