#!/usr/bin/env python3
"""Leg C corpus x {grammar, no grammar} — does the fixed KV grammar earn its keep?

The ONE fixed universal KV grammar (docs/plan-qemmi-lens.md P0) buys exactly one
thing: a guaranteed parse. It costs a forced non-empty value for every hinted key
— which is the sole cause of the absent-concept collapse, and therefore of the
two-pass grounded presence gate built to work around it.

This probe measures that trade directly. It runs the SAME 15 messy EN+DE order
emails Leg C measured (extracted from the probe source at runtime, so it cannot
drift from the canonical corpus), hinting each document's own labelled — truly
present — concepts PLUS two concepts verified absent from all 15, and compares
the two arms on: parse-rate, key stability, value fidelity, absent handling, and
collapse.

Result on Qwen3.6-35B-A3B-Q2_K_XL / Metal / greedy (2026-07-16) — the grammar
loses on every axis, INCLUDING the parse it exists to guarantee. Full write-up:
docs/note-nogrammar-refutation.md.

S1.4 widens that evidence along the two axes the result was thin on — a second
decode config (--temperature) and document shapes the synthetic corpus lacks
(--corpus real). Parse-rate is the metric at risk in both.

Usage:
    build-metal/bin/http_server -m models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf \
      -c 4096 -s 1 -p 18098 --attention-lens
    python3 py/lens_legc_nogrammar.py                        # greedy, both arms
    python3 py/lens_legc_nogrammar.py --temperature 0.7      # S1.4 second config
    python3 py/lens_legc_nogrammar.py --corpus real          # S1.4 real-shaped
    python3 py/lens_legc_nogrammar.py --corpus both --temperature 0.7

Needs only /v1/chat/completions (which already takes a per-request `grammar`),
so it runs against any build — zero engine changes.

NOTE: this probe measures EXTRACTION QUALITY (parse / keys / fidelity / absent).
The lens numerics (citations, receipts) need the attention tap and are gated by
the C++ sibling: QDOCS_S1=1 build-metal/bin/attn-provenance.
"""
import argparse, json, re, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE_SRC = ROOT / "tests" / "perf" / "attn_provenance.cpp"

# Verified absent from all 15 Leg C documents (no payment/terms/zahlung/warranty/
# garantie mention anywhere in the corpus) — so a value for either is fabrication.
ABSENT = ["payment_terms", "warranty_period"]

# The ONE fixed universal KV grammar (mirrors server_lens.cpp LENS_GBNF).
LENS_GBNF = (
    'root  ::= "{" ws pair ("," ws pair)* ws "}"\n'
    'pair  ::= "\\"" key "\\"" ws ":" ws "\\"" value "\\""\n'
    'key   ::= [a-z] [a-z0-9_]*\n'
    'value ::= ([^"\\\\\\n] | "\\\\" [^])+\n'
    'ws    ::= [ \\n\\t]*\n'
)


# ── corpus ───────────────────────────────────────────────────────────────────
def load_legc_corpus():
    """Extract qdocs_messy_corpus() (15 labelled EN+DE docs) from the C++ probe."""
    src = PROBE_SRC.read_text(encoding="utf-8")
    start = src.index("static std::vector<QMessy> qdocs_messy_corpus() {")
    body = src[start:src.index("\n}", start)]

    def unesc(s):
        out, i, m = [], 0, {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "r": "\r"}
        while i < len(s):
            if s[i] == "\\" and i + 1 < len(s):
                out.append(m.get(s[i + 1], s[i + 1])); i += 2
            else:
                out.append(s[i]); i += 1
        return "".join(out)

    lit = re.compile(r'"((?:[^"\\]|\\.)*)"')
    starts = [m.start() for m in re.finditer(r'\{"m_', body)]
    docs = []
    for i, s in enumerate(starts):
        entry = body[s: starts[i + 1] if i + 1 < len(starts) else len(body)]
        head, tail = entry[: entry.index('{{"')], entry[entry.index('{{"'):]
        lits = [unesc(x) for x in lit.findall(head)]
        labels = [(unesc(a), unesc(b)) for a, b in
                  re.findall(r'\{"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\}', tail)]
        docs.append({"tag": lits[0], "document": "".join(lits[1:]),
                     "fields": [{"concept": c, "value": v} for c, v in labels]})
    return docs


# ── S1.4 — real-shaped mails (the falsification attempt) ─────────────────────
# The Leg C corpus is synthetic "realistic-messy": constructed, and every doc is
# plain text that never fights the output format. Free prompting's 15/15 parse is
# therefore EMPIRICAL, not structural — a grammar forecloses fences and preambles
# by construction, prompting does not (docs/note-nogrammar-refutation.md, Honest
# limits). These three documents carry the shapes that corpus lacks and are aimed
# squarely at the parse-rate claim — not a corpus, a falsification attempt:
#
#   r_fwd   a forwarded chain: repeated headers, multi-level > quoting, the real
#           order buried under two replies, and a SUPERSEDED quantity above it.
#   r_html  an HTML-ish body — tags, entities — carrying an embedded JSON blob.
#           This is the sharpest test we have: the document contains braces,
#           quotes and key:value pairs, so free decode may copy or continue the
#           blob (a failure the grammar forecloses by construction). If the
#           grammar has any real justification, it should surface HERE.
#   r_sig   signature-heavy DE mail: vcard-ish footer, Handelsregister block,
#           legal disclaimer, social links — the fabrication bait that made the
#           grammar lift "!! sent from my phone" into warranty_period.
REALSHAPED = [
    {"tag": "r_fwd",
     "document":
        "From: dana.reeve@northgate.example\n"
        "Sent: Thursday, 13 November 2025 09:12\n"
        "To: orders@pallas-supply.example\n"
        "Subject: FW: FW: Re: Q4 restock — Northgate Interiors\n\n"
        "Forwarding for processing. Please use the LATEST numbers below, the first "
        "reply is out of date.\n\n"
        "Dana\n\n"
        "-----Original Message-----\n"
        "From: Tom Bexley <t.bexley@northgate.example>\n"
        "Sent: Wednesday, 12 November 2025 17:48\n"
        "Subject: FW: Re: Q4 restock — Northgate Interiors\n\n"
        "> Confirming the revised order for Northgate Interiors Ltd:\n"
        "> 120 units of the Ashgrove Oak Side Table at 143.50 EUR each.\n"
        "> Order number NG-2025-4471, order date 2025-11-12.\n"
        "> Delivery to the Leeds depot by 2025-12-05.\n"
        ">\n"
        ">> On 10 Nov 2025, Pallas Supply wrote:\n"
        ">> > Original request was 80 units — please confirm before we book.\n"
        ">> > Regards, Pallas\n",
     "fields": [{"concept": "customer", "value": "Northgate Interiors Ltd"},
                {"concept": "quantity", "value": "120"},
                {"concept": "unit_price", "value": "143.50"},
                {"concept": "order_number", "value": "NG-2025-4471"},
                {"concept": "order_date", "value": "2025-11-12"},
                {"concept": "delivery", "value": "2025-12-05"}]},
    {"tag": "r_html",
     "document":
        "From: no-reply@portal.veldt-trading.example\n"
        "Subject: [Portal] Order submitted &mdash; confirmation required\n\n"
        "<div style=\"font-family:Arial\">\n"
        "  <p>Hello&nbsp;team,</p>\n"
        "  <p>A new order has been submitted through the supplier portal by "
        "<b>Veldt Trading GmbH</b>.</p>\n"
        "  <p>Please confirm within 2 business days. Details:&nbsp;</p>\n"
        "  <ul>\n"
        "    <li>Product: Kestrel Folding Chair</li>\n"
        "    <li>Quantity: 340</li>\n"
        "    <li>Unit price: 27.90 EUR</li>\n"
        "    <li>Order date: 2025-09-30</li>\n"
        "  </ul>\n"
        "  <p>Raw payload for your integration team:</p>\n"
        "  <pre>{\"order_id\": \"VT-88-2231\", \"status\": \"pending\", "
        "\"submitted_by\": \"portal-bot\", \"lines\": [{\"sku\": \"KFC-01\", "
        "\"qty\": 340}]}</pre>\n"
        "  <p>Ship to the Rotterdam hub by 2025-10-21.</p>\n"
        "  <p>&mdash; Veldt Portal (automated) &bull; do not reply</p>\n"
        "</div>\n",
     "fields": [{"concept": "customer", "value": "Veldt Trading GmbH"},
                {"concept": "product", "value": "Kestrel Folding Chair"},
                {"concept": "quantity", "value": "340"},
                {"concept": "unit_price", "value": "27.90"},
                {"concept": "order_date", "value": "2025-09-30"},
                {"concept": "delivery", "value": "2025-10-21"}]},
    {"tag": "r_sig",
     "document":
        "Von: h.brandt@keller-moebel.example\n"
        "Betreff: Bestellung KW47 - Keller Möbelwerk\n\n"
        "Sehr geehrte Damen und Herren,\n\n"
        "hiermit bestellen wir verbindlich für die Keller Möbelwerk GmbH & Co. KG "
        "220 Stück des Artikels Buche Massivholzplatte zum Stückpreis von 64,80 EUR. "
        "Bestelldatum 2025-11-17, Bestellnummer KMW-77315. "
        "Lieferung bitte bis zum 2025-12-08 an unser Werk in Fulda.\n\n"
        "Mit freundlichen Grüßen\n\n"
        "Hanne Brandt\n"
        "Leiterin Einkauf | Head of Procurement\n"
        "-----------------------------------------------------------\n"
        "Keller Möbelwerk GmbH & Co. KG\n"
        "Industriestraße 14 | 36037 Fulda | Deutschland\n"
        "Tel +49 661 4801 220 | Fax +49 661 4801 229 | Mobil +49 171 2244118\n"
        "h.brandt@keller-moebel.example | www.keller-moebel.example\n"
        "LinkedIn: /in/hannebrandt | Xing: /profile/Hanne_Brandt\n"
        "-----------------------------------------------------------\n"
        "Sitz der Gesellschaft: Fulda | Registergericht: Amtsgericht Fulda HRA 3392\n"
        "Persönlich haftende Gesellschafterin: Keller Verwaltungs-GmbH, "
        "Registergericht Amtsgericht Fulda HRB 1187\n"
        "Geschäftsführer: Dr. Ulrich Keller, Marta Sonnleitner | USt-IdNr.: DE812447901\n"
        "-----------------------------------------------------------\n"
        "Diese E-Mail enthält vertrauliche und/oder rechtlich geschützte "
        "Informationen. Wenn Sie nicht der richtige Adressat sind oder diese E-Mail "
        "irrtümlich erhalten haben, informieren Sie bitte sofort den Absender und "
        "vernichten Sie diese E-Mail. Das unerlaubte Kopieren sowie die unbefugte "
        "Weitergabe dieser Mail ist nicht gestattet.\n"
        "Bitte denken Sie an die Umwelt, bevor Sie diese E-Mail ausdrucken.\n"
        "Gesendet von meinem Mobilgerät\n",
     "fields": [{"concept": "customer", "value": "Keller Möbelwerk GmbH & Co. KG"},
                {"concept": "quantity", "value": "220"},
                {"concept": "unit_price", "value": "64,80"},
                {"concept": "order_number", "value": "KMW-77315"},
                {"concept": "order_date", "value": "2025-11-17"},
                {"concept": "delivery", "value": "2025-12-08"}]},
]


# ── the shipped v0 extraction instruction, verbatim (lens_build_instruction) ──
def instruction(keys):
    return ("\n\nExtract the following fields from the document above into a flat "
            "JSON object of \"key\": \"value\" pairs, using exactly these "
            "snake_case keys: " + ", ".join(keys) +
            ". Copy each value verbatim from the document. Output ONLY the JSON "
            "object, nothing else.")


def complete(url, doc, keys, grammar, temperature, max_tokens=400):
    body = {"model": "qwen-local",
            "messages": [{"role": "user", "content": doc + instruction(keys)}],
            "temperature": temperature, "max_tokens": max_tokens,
            "enable_thinking": False}
    if grammar:
        body["grammar"] = grammar
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return r["choices"][0]["message"]["content"]


def tolerant_parse(raw):
    """The tolerant-parse contract candidate: strip a markdown fence, take the
    outermost JSON object, else FAIL LOUDLY (return None) — never guess."""
    s = raw.strip()
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) > 1 else s
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        s = m.group(0)
    try:
        return json.loads(s), None
    except Exception as e:
        return None, str(e)


def norm(x):
    return re.sub(r"\s+", " ", str(x)).strip().lower()


def matches(truth, got):
    if got is None:
        return False
    t, g = norm(truth), norm(got)
    return t in g or g in t


def evaluate(arm, url, corpus, grammar, temperature):
    n = len(corpus)
    parsed = keys_ok = fid_ok = fid_tot = absent_ok = absent_tot = 0
    fabricated, broken = [], []
    for d in corpus:
        present = [f["concept"] for f in d["fields"]]
        hint = present + ABSENT
        raw = complete(url, d["document"], hint, grammar, temperature)
        js, err = tolerant_parse(raw)
        if js is None:
            broken.append(d["tag"])
            print(f"  {arm} {d['tag']:7} x NO PARSE ({err[:38]}) raw={raw[:64]!r}")
            continue
        parsed += 1
        if set(js.keys()) <= set(hint):
            keys_ok += 1
        else:
            print(f"  {arm} {d['tag']:7} ! key drift: {list(js.keys())}")
        lost = 0
        for f in d["fields"]:
            fid_tot += 1
            v = js.get(f["concept"])
            fid_ok += matches(f["value"], v)
            lost += v is None or norm(v) in ("", ",")
        if lost and lost == len(d["fields"]):
            broken.append(d["tag"])
        for k in ABSENT:
            absent_tot += 1
            if k not in js or js[k] is None or norm(js[k]) in ("", "null", "none", "n/a", "-"):
                absent_ok += 1
            else:
                fabricated.append(f"{d['tag']}:{k}={js[k]!r}")
    print(f"\n  -- {arm} --")
    print(f"     parse-rate      {parsed}/{n}   <- the grammar's entire reason to exist")
    print(f"     key stability   {keys_ok}/{n}")
    print(f"     value fidelity  {fid_ok}/{fid_tot} labelled fields vs ground truth")
    print(f"     absent handled  {absent_ok}/{absent_tot} (declined, not fabricated)")
    if fabricated:
        print(f"     FABRICATED      {fabricated[:6]}")
    if broken:
        print(f"     COLLAPSED/UNPARSEABLE: {sorted(set(broken))}")
    return {"parse": f"{parsed}/{n}", "keys": f"{keys_ok}/{n}",
            "fidelity": f"{fid_ok}/{fid_tot}", "absent": f"{absent_ok}/{absent_tot}",
            "broken": sorted(set(broken))}


def verify_absent(corpus):
    """The absent metric is only meaningful if the planted concepts really are
    absent. Assert it mechanically instead of by eye — any mention of these stems
    anywhere in a document would make a value for them arguably grounded, and the
    fabrication count a lie."""
    stems = ["payment", "terms", "zahlung", "warrant", "garantie", "gewährleist"]
    bad = [f"{d['tag']}:{s}" for d in corpus for s in stems
           if s in d["document"].lower()]
    if bad:
        print(f"FAIL: ABSENT concepts are not absent — {bad}\n"
              f"      a value for them would not be a fabrication; fix the corpus "
              f"or the ABSENT list.")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18098)
    ap.add_argument("--temperature", type=float, default=0,
                    help="0 = greedy (the measured regime); >0 widens the evidence")
    ap.add_argument("--corpus", choices=["legc", "real", "both"], default="legc",
                    help="legc = the 15 synthetic messy docs (the measured regime); "
                         "real = the 3 real-shaped mails (S1.4 falsification: "
                         "forwarded chain / HTML+embedded-JSON / signature-heavy)")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"

    corpus = []
    if a.corpus in ("legc", "both"):
        legc = load_legc_corpus()
        if len(legc) != 15:
            print(f"FAIL: expected the 15-doc Leg C corpus, extracted {len(legc)}")
            sys.exit(1)
        corpus += legc
    if a.corpus in ("real", "both"):
        corpus += REALSHAPED
    verify_absent(corpus)
    print(f"corpus={a.corpus} ({len(corpus)} docs)  |  temperature={a.temperature}  |  {url}")

    print("\n" + "=" * 72 + "\nARM: NO GRAMMAR (free prompting)\n" + "=" * 72)
    ng = evaluate("NG", url, corpus, None, a.temperature)
    print("\n" + "=" * 72 + "\nARM: FIXED KV GRAMMAR\n" + "=" * 72)
    gr = evaluate("GR", url, corpus, LENS_GBNF, a.temperature)

    print("\n" + "=" * 72)
    print(f"{'axis':18}{'no grammar':>14}{'grammar':>14}")
    for k in ("parse", "keys", "fidelity", "absent"):
        print(f"{k:18}{ng[k]:>14}{gr[k]:>14}")
    print(f"{'broken docs':18}{len(ng['broken']):>14}{len(gr['broken']):>14}")


if __name__ == "__main__":
    main()
