#!/usr/bin/env python3
"""CONCUR — how many users a lens server serves (prefill-only routes).

Four legs, each against servers the caller already started (this script never
launches a process). Synthetic English delivery notes only — no personal data.
Documents stay at or under ~4K prompt tokens (10K is out of scope for now).

  service --port P            one process: cold/warm cost per verb at ~1K and ~4K
                              (locate, absent, verdict, compare). Needs a FULL
                              --attention-lens server (verdict is on it).
  scale   --ports P1,P2,...   N processes on one GPU, one closed-loop client per
                              process: aggregate req/s and p50/p95, cold ~4K
                              locate and a warm question loop. Run with N = 1..4.
  queue   --port P            one process, 8 clients at once: order, wait, spread.
  thrash  --port P            U users, each with its own kept document, taking
                              turns: how many requests come back warm (U = 2..8).

Usage: python3 py/lens_concur.py <leg> --port 18140 [--reps 3]
"""

import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ── synthetic documents ────────────────────────────────────────────────────────
NAMES = ["ACME Corp GmbH", "Birchwood Ltd", "Calder Logistics", "Dunmore Foods",
         "Eastgate Metals", "Fairview Textiles", "Granite Supply Co", "Harbor Tools"]
GOODS = ["steel brackets", "cotton rolls", "pallet wrap", "copper wire", "oak panels",
         "glass jars", "rubber seals", "paper sacks", "LED strips", "hinges"]
CITIES = ["Leeds", "Bristol", "Hamburg", "Rotterdam", "Lyon", "Turin", "Gdansk", "Porto"]


def make_doc(seed: int, target_tokens: int) -> str:
    """A delivery-note style document of roughly target_tokens (≈ 2.8 chars/token: numbers tokenize densely)."""
    rng = random.Random(seed)
    cust = rng.choice(NAMES)
    lines = [f"Delivery record {seed:05d}", f"Customer: {cust}",
             f"Delivery date: {rng.randint(1, 28)} {rng.choice(['March', 'April', 'May', 'June'])} 2026",
             f"Destination: {rng.choice(CITIES)} warehouse, dock {rng.randint(1, 12)}", ""]
    n = 0
    while len("\n".join(lines)) < target_tokens * 2.8:
        n += 1
        g, q, p = rng.choice(GOODS), rng.randint(10, 900), rng.randint(2, 400) + rng.randint(0, 99) / 100
        lines.append(rng.choice([
            f"Line {n}: {q} units of {g} at {p:.2f} EUR each were loaded in {rng.choice(CITIES)}.",
            f"Line {n}: the driver reported {rng.randint(0, 5)} damaged cartons of {g}; {q} units arrived intact.",
            f"Line {n}: {g} ({q} units) were held for inspection and released after {rng.randint(1, 48)} hours.",
            f"Line {n}: invoice reference INV-{rng.randint(1000, 9999)} covers {q} units of {g} at {p:.2f} EUR.",
        ]))
    lines.append(f"Signed on behalf of {cust} by the receiving clerk.")
    return "\n".join(lines)


QUESTION_SETS = [
    [{"id": "customer", "question": "Who is the customer?"},
     {"id": "date", "question": "When was the delivery?"}],
    [{"id": "damage", "question": "Were any cartons damaged?"}],
    [{"id": "dest", "question": "Where were the goods delivered?"},
     {"id": "invoice", "question": "Which invoice reference is given?"}],
    [{"id": "held", "question": "Were any goods held for inspection?"}],
]
VERDICT_QS = [{"id": "v1", "question": "Were any cartons damaged?"},
              {"id": "v2", "question": "Was the delivery made to a warehouse?"},
              {"id": "v3", "question": "Is the customer a bank?"},
              {"id": "v4", "question": "Were more than 100 units of any item delivered?"},
              {"id": "v5", "question": "Was the delivery signed?"}]


# ── HTTP ───────────────────────────────────────────────────────────────────────
def post(port: int, route: str, body: dict) -> tuple[int, float, dict]:
    req = urllib.request.Request(f"http://localhost:{port}/v1/{route}", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.loads(r.read())
            return r.status, time.perf_counter() - t, out
    except urllib.error.HTTPError as e:
        return e.code, time.perf_counter() - t, {"error": e.read().decode()[:300]}


def locate(port, doc, qs, head=None, doc_id=None):
    b = {"document": doc, "key_vocabulary": qs, "top_k": 3}
    if head:
        b["head"] = head
        b["key_aggregation"] = "mean"
        b["top_k"] = 16
    if doc_id:
        b["document_id"] = doc_id
    return post(port, "locate", b)


def ok(code, out, what):
    if code != 200:
        raise SystemExit(f"{what}: expected 200, actual {code}: {out.get('error', out)}")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def fmt(xs):
    return f"p50 {statistics.median(xs):6.2f} s  p95 {pct(xs, 95):6.2f} s  max {max(xs):6.2f} s"


# ── legs ───────────────────────────────────────────────────────────────────────
def leg_service(a):
    port, reps = a.port, a.reps
    print(f"SERVICE port {port}, {reps} reps per cell; seconds per request (client wall time)\n")
    for size in (1000, 3600):
        print(f"── ~{size} token document ──")
        seed0 = 1000 * size
        # warm-up so the first cold number is not a first-request artefact
        locate(port, make_doc(seed0 - 1, size), QUESTION_SETS[0])
        cold, cold_abs, warm, warm_abs, plen = [], [], [], [], []
        for r in range(reps):
            doc = make_doc(seed0 + r, size)
            c, t, o = locate(port, doc, QUESTION_SETS[0]); ok(c, o, "locate cold"); cold.append(t); plen.append(o["prompt_len"])
            c, t, o = locate(port, doc, QUESTION_SETS[0], head="absent"); ok(c, o, "absent cold"); cold_abs.append(t)
            did = f"svc-{size}-{r}"
            c, t, o = locate(port, doc, QUESTION_SETS[0], head="absent", doc_id=did); ok(c, o, "keep")
            for qs in QUESTION_SETS[1:]:
                c, t, o = locate(port, doc, qs, doc_id=did); ok(c, o, "locate warm")
                assert o.get("prefix") == "warm", o.get("prefix"); warm.append(t)
                c, t, o = locate(port, doc, qs, head="absent", doc_id=did); ok(c, o, "absent warm")
                assert o.get("prefix") == "warm", o.get("prefix"); warm_abs.append(t)
        print(f"  prompt tokens {min(plen)}–{max(plen)}")
        print(f"  locate  cold  {fmt(cold)}")
        print(f"  absent  cold  {fmt(cold_abs)}")
        print(f"  locate  warm  {fmt(warm)}")
        print(f"  absent  warm  {fmt(warm_abs)}")

        v1, v5, vw, vlen = [], [], [], []
        for r in range(reps):
            doc = make_doc(seed0 + 50 + r, size)
            c, t, o = post(port, "verdict", {"document": doc, "questions": VERDICT_QS[:1]}); ok(c, o, "verdict 1q"); v1.append(t)
            vlen.append(o["answers"][0]["prompt_len"])
            doc = make_doc(seed0 + 60 + r, size)
            c, t, o = post(port, "verdict", {"document": doc, "questions": VERDICT_QS}); ok(c, o, "verdict 5q"); v5.append(t)
            did = f"svc-v-{size}-{r}"
            post(port, "verdict", {"document": doc, "questions": VERDICT_QS[:1], "document_id": did})
            c, t, o = post(port, "verdict", {"document": doc, "questions": VERDICT_QS[1:2], "document_id": did})
            ok(c, o, "verdict warm"); assert o.get("prefix") == "warm", o.get("prefix"); vw.append(t)
        print(f"  verdict prompt tokens {min(vlen)}–{max(vlen)}")
        print(f"  verdict 1 question cold   {fmt(v1)}")
        print(f"  verdict 5 questions cold  {fmt(v5)}   (extra per question ≈ {(statistics.median(v5) - statistics.median(v1)) / 4:.2f} s)")
        print(f"  verdict 1 question warm   {fmt(vw)}")

        cc, cw, clen = [], [], []
        for r in range(reps):
            doc = make_doc(seed0 + 80 + r, size // 2)          # original + revision ≈ size
            units = [l for l in doc.split("\n") if l.strip()]
            rng = random.Random(r)
            rev = "\n".join(u for u in units if rng.random() > 0.2)
            c, t, o = post(port, "compare", {"original_units": units, "revised": rev}); ok(c, o, "compare cold"); cc.append(t)
            clen.append(o["prompt_len"])
            did = f"svc-c-{size}-{r}"
            post(port, "compare", {"original_units": units, "revised": rev, "document_id": did})
            rev2 = "\n".join(u for u in units if rng.random() > 0.2)
            c, t, o = post(port, "compare", {"original_units": units, "revised": rev2, "document_id": did})
            ok(c, o, "compare warm"); assert o.get("prefix") == "warm", o.get("prefix"); cw.append(t)
        print(f"  compare prompt tokens {min(clen)}–{max(clen)}")
        print(f"  compare cold  {fmt(cc)}")
        print(f"  compare warm  {fmt(cw)}   (original kept, new revision)\n")


def closed_loop(ports, n_per_client, make_request):
    """One client per port, each sending n_per_client requests back to back."""
    lat = [[] for _ in ports]
    start = threading.Barrier(len(ports))

    def client(i):
        start.wait()
        for k in range(n_per_client):
            c, t, o = make_request(ports[i], i, k)
            ok(c, o, f"client {i} request {k}")
            lat[i].append(t)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(len(ports)) as ex:
        list(ex.map(client, range(len(ports))))
    wall = time.perf_counter() - t0
    flat = [x for l in lat for x in l]
    return wall, flat


def leg_scale(a):
    ports = [int(p) for p in a.ports.split(",")]
    n = len(ports)
    print(f"SCALE {n} process(es) {ports}, one client each")
    # cold ~4K locate, a new document every request
    per = a.reps
    def cold(port, i, k):
        return locate(port, make_doc(500000 + 1000 * i + k + 100 * n, 3600), QUESTION_SETS[0])
    for p in ports:  # warm-up
        locate(p, make_doc(1, 400), QUESTION_SETS[0])
    wall, flat = closed_loop(ports, per, cold)
    print(f"  cold ~4K locate: {len(flat)} requests in {wall:.1f} s = {len(flat) / wall * 60:5.1f} req/min   {fmt(flat)}")

    # warm question loop: each client keeps its own ~4K document, then asks
    docs = [make_doc(700000 + i + 100 * n, 3600) for i in range(n)]
    for i, p in enumerate(ports):
        c, t, o = locate(p, docs[i], QUESTION_SETS[0], head="absent", doc_id=f"sc-{n}-{i}"); ok(c, o, "keep")
    def warm(port, i, k):
        heads = [None, "absent", "choice", None]
        c, t, o = locate(port, docs[i], QUESTION_SETS[k % 4], head=heads[k % 4], doc_id=f"sc-{n}-{i}")
        if c == 200 and o.get("prefix") != "warm":
            raise SystemExit(f"scale warm: expected prefix warm, actual {o.get('prefix')}")
        return c, t, o
    wall, flat = closed_loop(ports, per * 8, warm)
    print(f"  warm ~4K question: {len(flat)} requests in {wall:.1f} s = {len(flat) / wall * 60:5.1f} req/min   {fmt(flat)}")


def leg_queue(a):
    port, clients = a.port, 8
    print(f"QUEUE port {port}: {clients} clients start together, 3 cold ~1K locates each")
    locate(port, make_doc(2, 400), QUESTION_SETS[0])
    lat = [[] for _ in range(clients)]
    order = []
    lock = threading.Lock()
    start = threading.Barrier(clients)

    def client(i):
        start.wait()
        for k in range(3):
            c, t, o = locate(port, make_doc(800000 + 10 * i + k, 1000), QUESTION_SETS[0])
            ok(c, o, "queue")
            lat[i].append(t)
            with lock:
                order.append(i)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(clients) as ex:
        list(ex.map(client, range(clients)))
    wall = time.perf_counter() - t0
    flat = [x for l in lat for x in l]
    service = wall / len(flat)
    print(f"  {len(flat)} requests in {wall:.1f} s ⇒ {service:.2f} s each on average")
    print(f"  latency {fmt(flat)}  (= {pct(flat, 95) / service:.1f}× the service time at p95)")
    print(f"  per-client total: " + "  ".join(f"{sum(l):.1f}" for l in lat))
    print(f"  completion order: {order}")


def leg_thrash(a):
    port = a.port
    print(f"THRASH port {port}: U users each with a kept ~2K document, 3 rounds of turns")
    for users in (2, 4, 5, 6, 8):
        docs = [make_doc(900000 + 100 * users + u, 2000) for u in range(users)]
        warm, cold, lw, lc = 0, 0, [], []
        for u in range(users):  # everyone keeps their document once
            c, t, o = locate(port, docs[u], QUESTION_SETS[0], head="absent", doc_id=f"th-{users}-{u}"); ok(c, o, "keep")
        for rnd in range(3):
            for u in range(users):
                c, t, o = locate(port, docs[u], QUESTION_SETS[1 + rnd], head="absent", doc_id=f"th-{users}-{u}")
                ok(c, o, "turn")
                if o.get("prefix") == "warm":
                    warm += 1; lw.append(t)
                else:
                    cold += 1; lc.append(t)
        tot = warm + cold
        mean = statistics.mean(lw + lc)
        print(f"  U={users}: {warm}/{tot} warm, {cold}/{tot} cold; mean {mean:.2f} s/turn"
              + (f"  (warm {statistics.median(lw):.2f} s" if lw else "  (")
              + (f", cold {statistics.median(lc):.2f} s)" if lc else ")"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("leg", choices=["service", "scale", "queue", "thrash"])
    ap.add_argument("--port", type=int, default=18140)
    ap.add_argument("--ports", default="18130")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    {"service": leg_service, "scale": leg_scale, "queue": leg_queue, "thrash": leg_thrash}[a.leg](a)


if __name__ == "__main__":
    main()
