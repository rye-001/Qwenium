#!/usr/bin/env python3
"""Leg C through the LIVE /v1/extract endpoint — does Stage 2 reproduce Stage 1?

Stage 1 measured the grammar refutation two ways: through /v1/chat/completions
(py/lens_legc_nogrammar.py, extraction quality) and through the shipped driver
in-process (QDOCS_S1 in tests/perf/attn_provenance.cpp, the lens numerics). This
probe closes the last gap: the same 15 messy EN+DE order emails through the real
HTTP endpoint, exercising what neither of those does — the request/concepts
marshalling, absent-by-omission, the tolerant parse, and the 422 split.

Stage 1's free arm, to reproduce:
    parse 15/15 (tolerant)  ·  fidelity 75/75  ·  absent 30/30 (declined)

The corpus is extracted from tests/perf/attn_provenance.cpp at runtime, so it
cannot drift from the canonical one.

Usage:
    build-metal/bin/http_server -m models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf \
      -c 4096 -s 1 -p 18098 --attention-lens
    python3 py/lens_legc_endpoint.py [--port 18098]
"""
import argparse, json, sys, urllib.request, urllib.error

from lens_legc_nogrammar import ABSENT, load_legc_corpus, norm, matches, verify_absent


def extract(url, doc, keys, max_tokens=400):
    """One /v1/extract call. Returns (http_code, payload)."""
    body = json.dumps({"document": doc,
                       "key_vocabulary": [{"key": k, "gloss": ""} for k in keys],
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=600)
        return r.getcode(), json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18098)
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/v1/extract"

    corpus = load_legc_corpus()
    if len(corpus) != 15:
        print(f"FAIL: expected the 15-doc Leg C corpus, extracted {len(corpus)}")
        sys.exit(1)
    verify_absent(corpus)
    print(f"Leg C via the LIVE endpoint: {len(corpus)} docs  |  {url}\n")

    parsed = fid_ok = fid_tot = absent_ok = absent_tot = 0
    refused, fabricated, bad_version = [], [], []

    for d in corpus:
        present = [f["concept"] for f in d["fields"]]
        code, payload = extract(url, d["document"], present + ABSENT)

        if code == 422:
            # A loud refusal is a legitimate outcome, NOT a crash — but it is also
            # not an extraction, so it costs the parse rate. Never silently skipped.
            refused.append(d["tag"])
            fid_tot += len(d["fields"]); absent_tot += len(ABSENT)
            print(f"  {d['tag']:7} 422 refused: {json.loads(payload)['code']}")
            continue
        if code != 200:
            print(f"FAIL: {d['tag']} unexpected HTTP {code}: {str(payload)[:160]}")
            sys.exit(1)
        parsed += 1

        if payload.get("format_version") != "qemmi-lens/v2":
            bad_version.append(f"{d['tag']}:{payload.get('format_version')}")
        fields = {f["key"]: f for f in payload["fields"]}

        for lf in d["fields"]:
            fid_tot += 1
            got = fields.get(lf["concept"])
            if got and got["value"] is not None and matches(lf["value"], got["value"]):
                fid_ok += 1
        for k in ABSENT:
            absent_tot += 1
            got = fields.get(k)
            if got is None or got["badge"] == "absent" or got["value"] is None:
                absent_ok += 1
            else:
                fabricated.append(f"{d['tag']}:{k}={got['value']!r}")
        print(f"  {d['tag']:7} ok  fields={len(payload['fields'])}  "
              f"absent={[fields[k]['badge'] for k in ABSENT if k in fields]}")

    n = len(corpus)
    print(f"\n{'='*62}\n{'axis':28}{'live endpoint':>16}{'Stage 1 free':>16}")
    print(f"{'parse (200, tolerant)':28}{f'{parsed}/{n}':>16}{'15/15':>16}")
    print(f"{'value fidelity':28}{f'{fid_ok}/{fid_tot}':>16}{'75/75':>16}")
    print(f"{'absent handled':28}{f'{absent_ok}/{absent_tot}':>16}{'30/30':>16}")
    if refused:      print(f"  422 refused (loud, not partial): {refused}")
    if fabricated:   print(f"  FABRICATED: {fabricated[:6]}")
    if bad_version:  print(f"  BAD format_version: {bad_version}")

    ok = (parsed == n and fid_ok == fid_tot and absent_ok == absent_tot
          and not bad_version)
    print(f"\n{'STAGE 2 ENDPOINT GATE: PASS — Stage 1 reproduces through /v1/extract'
           if ok else 'STAGE 2 ENDPOINT GATE: FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
