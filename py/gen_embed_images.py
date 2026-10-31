#!/usr/bin/env python3
"""Synthetic test images for the image-embedding gate (docs/plan-embed-image.md §3).

No photos, nothing downloaded: shapes drawn pixel by pixel and written as PNG
with zlib only. Sizes cover the dyn-size branches: a square, a landscape, an
aspect-extreme wide and tall, a page, and one smaller than the minimum budget.
Each image has a caption, for the retrieval sanity check (no bar).

    python3 py/gen_embed_images.py tests/fixtures/embed_image
"""
import os
import struct
import sys
import zlib

WHITE = (255, 255, 255)


def png(path, w, h, px):
    raw = b"".join(b"\x00" + bytes(c for x in range(w) for c in px(x, y)) for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(data)


def red_circle(x, y):
    return (220, 30, 30) if (x - 112) ** 2 + (y - 112) ** 2 < 70 ** 2 else WHITE


def blue_squares(x, y):
    if 80 <= y < 280 and (80 <= x < 280 or 360 <= x < 560):
        return (30, 60, 200)
    return WHITE


def green_triangle(x, y):            # 1200 x 300: apex top centre, base at the bottom
    half = (y - 20) * 2.0
    return (30, 160, 60) if 20 <= y < 280 and abs(x - 600) <= half else WHITE


def stripes(x, y):                   # 300 x 1200
    return (0, 0, 0) if (y // 60) % 2 == 0 else WHITE


def page(x, y):                      # 850 x 1100: a document page with text lines and a box
    if 80 <= y < 120 and 80 <= x < 500:
        return (20, 20, 20)                              # a title bar
    if 180 <= y < 860 and (y - 180) % 40 < 14 and 80 <= x < 770 - ((y // 40) % 3) * 90:
        return (90, 90, 90)                              # body lines
    if 920 <= y < 1020 and 480 <= x < 770 and (y in (920, 1019) or x in (480, 769)):
        return (0, 0, 0)                                 # a signature box
    return WHITE


def yellow_tiny(x, y):               # 40 x 40, below the minimum token budget
    return (240, 200, 20)


def checker(x, y):                   # 512 x 512
    return (0, 0, 0) if ((x // 64) + (y // 64)) % 2 == 0 else WHITE


IMAGES = [
    ("red_circle_224", 224, 224, red_circle, "a red circle on a white background"),
    ("blue_squares_640x480", 640, 480, blue_squares, "two blue squares side by side"),
    ("green_triangle_1200x300", 1200, 300, green_triangle, "a wide green triangle"),
    ("stripes_300x1200", 300, 1200, stripes, "black and white horizontal stripes"),
    ("page_850x1100", 850, 1100, page, "a document page with lines of text and a signature box"),
    ("yellow_tiny_40", 40, 40, yellow_tiny, "a plain yellow square"),
    ("checker_512", 512, 512, checker, "a black and white checkerboard"),
]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/embed_image"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "captions.txt"), "w") as f:
        for name, w, h, px, cap in IMAGES:
            png(os.path.join(out, name + ".png"), w, h, px)
            f.write(f"{name}\t{cap}\n")
    print(f"wrote {len(IMAGES)} images to {out}")


if __name__ == "__main__":
    main()
