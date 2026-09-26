#!/usr/bin/env python3
"""DECIDE1 — can span-only answer Jev's `choice` type?

THE CLAIM UNDER TEST. Not "can the model pick an option" — scouting showed it
cannot (planting options in the prompt and reading the winning span scored 0/6;
the span lands on the evidence every time). The claim is the inverted one:
score each CATEGORY'S CONTENT DESCRIPTION as a locate key and take the argmax.
Retrieval, which locate is good at, instead of commitment, which it is not.

WHAT THIS DRIVES. The shipped POST /v1/locate, not a probe-local tap. That is
deliberate: the product question is "can a client do this today with no server
change", so the probe must go through the same route a client would.

    ./build-metal/bin/qwenium-server -m models/Qwen3.8-9B-Q8_0.gguf \
        -c 4096 -s 1 -p 18190 --attention-lens --lens-locate-only
    python3 py/lens_decide1.py

CONTROLS, and why each is here:
  * BILINGUAL, scored on both halves separately. An EN-only rate is a
    hypothesis until the DE half agrees (this repo has been burned twice).
  * ORDER ROTATED. LOCABSENT measured that a key's score depends on its
    POSITION in the request. Arm A fixes one canonical order and is therefore
    the pessimistic arm; arm B averages all rotations; and the per-rotation
    spread is printed so "it only worked in one order" cannot hide.
  * ADVERSARIAL LURES built into the corpus on purpose: a legal document that
    talks about fees, an HR document about salary bands, an engineering
    document about cost. A routing corpus with no cross-category vocabulary
    measures nothing.
  * DESCRIPTIONS ARE FROZEN. The four category descriptions below were written
    BEFORE the first run and are not tuned against the score. Tuning them
    against this corpus would make the number a fit, not a measurement.

KNOWN LIMITATION, stated up front: the corpus is synthetic and written by the
same author as the probe. That is the same weakness the Leg C corpus carries
("15 self-authored documents"), and it caps what this can prove. It is a
measurement of the mechanism, not a market claim.
"""
import json, statistics, sys, time, urllib.error, urllib.request

PORT = "18190"
URL = f"http://127.0.0.1:{PORT}/v1/locate"

# ── Frozen category descriptions (written once, never tuned) ────────────────
CATS_EN = {
    "finance":     "an invoice, a payment, a budget, or an amount of money owed",
    "legal":       "a contract, a clause, termination, liability, or governing law",
    "engineering": "software, a server error, a deployment, or a stack trace",
    "people":      "an employee, hiring, leave, or an HR policy",
}
CATS_DE = {
    "finance":     "eine Rechnung, eine Zahlung, ein Budget oder ein offener Betrag",
    "legal":       "ein Vertrag, eine Klausel, Kuendigung, Haftung oder geltendes Recht",
    "engineering": "Software, ein Serverfehler, ein Deployment oder ein Stacktrace",
    "people":      "ein Mitarbeiter, Einstellung, Urlaub oder eine Personalrichtlinie",
}
# Content-word decomposition, no articles or prepositions. ARM C exists because
# ARM A collapsed into `finance`: the server takes MAX over a key's tokens, so a
# description carrying more filler ("AN invoice, A payment, ... AN amount OF
# money") gets more chances for one stray token to spike. Scouting already saw
# this exact failure (`an` <-> ` agreement`, 0.507 with the same key's next span
# at 0.047). These parts are the same concepts with the filler removed, and all
# of them go in ONE call, so this costs one prefill like ARM A.
PARTS_EN = {
    "finance":     ["invoice", "payment", "budget", "amount owed"],
    "legal":       ["contract", "clause", "termination", "liability", "governing law"],
    "engineering": ["software", "server error", "deployment", "stack trace"],
    "people":      ["employee", "hiring", "leave", "staff policy"],
}
PARTS_DE = {
    "finance":     ["Rechnung", "Zahlung", "Budget", "offener Betrag"],
    "legal":       ["Vertrag", "Klausel", "Kuendigung", "Haftung", "geltendes Recht"],
    "engineering": ["Software", "Serverfehler", "Deployment", "Stacktrace"],
    "people":      ["Mitarbeiter", "Einstellung", "Urlaub", "Personalrichtlinie"],
}
NAMES = list(CATS_EN)

# ── Corpus: 4 categories x 5 docs x 2 languages = 40. `lure` marks documents
#    deliberately seeded with another category's vocabulary. ─────────────────
DOCS = [
 # ---------------- FINANCE (EN) ----------------
 ("f_en1","finance",False,False,"Reminder: invoice 4471 for 1,992.00 GBP fell due on 2025-09-30 and is still unpaid. Please arrange the transfer this week or let us know if it is disputed."),
 ("f_en2","finance",False,False,"Please approve the purchase order for 40 Alu Pro stands at 82.00 each. This comes out of the Q4 equipment budget and needs sign-off before Friday."),
 ("f_en3","finance",False,False,"Submitting my expense claim for the Berlin trip: 340.50 in total, receipts attached for the hotel and the train. Reimbursement to the usual account please."),
 ("f_en4","finance",False,True, "Our VAT filing for Q3 is due next month. The regulations changed this year, so I want to confirm the treatment of the cross-border sales before we submit."),
 ("f_en5","finance",False,False,"The direct debit failed again on the 15th. Bank says the mandate was cancelled. Can you re-send the details so we can settle the outstanding balance?"),
 # ---------------- LEGAL (EN) ----------------
 ("l_en1","legal",False,True,  "We intend to serve notice of termination under clause 11.2, effective in 90 days. Outstanding fees up to that date remain payable in the usual way."),
 ("l_en2","legal",False,False, "Please review the attached mutual NDA before Thursday. I am particularly concerned about the definition of confidential information and the survival period."),
 ("l_en3","legal",False,True,  "The data processing agreement needs updating for the new sub-processor. Our systems now store personal data in a second region, so the transfer clauses must change."),
 ("l_en4","legal",False,False, "The counterparty has rejected mediation and is referring the dispute to arbitration before a single arbitrator seated in London."),
 ("l_en5","legal",False,True,  "Question on the employment contract template: the non-compete runs for twelve months, which may be unenforceable. Can we shorten it for new starters?"),
 # ---------------- ENGINEERING (EN) ----------------
 ("e_en1","engineering",False,False,"Checkout returns HTTP 500 whenever the cart holds more than 50 items. Stack trace points at OrderValidator.java line 212. Reproducible on staging every time."),
 ("e_en2","engineering",False,False,"Rolling back release 0.9.4. The worker pods restart every four minutes under load and the null pointer in SessionCache.evict is in every log."),
 ("e_en3","engineering",False,True, "The nightly migration now takes six hours and the query plan shows a full scan. It is also driving up our database bill, so I want to add the index this sprint."),
 ("e_en4","engineering",False,False,"Security advisory for the image library we vendor: a crafted PNG can overflow the decoder. Patch is available upstream; we should bump and redeploy today."),
 ("e_en5","engineering",False,False,"Their API starts returning 429 after about 300 calls a minute and our retry logic makes it worse. We need backoff before the integration goes live."),
 # ---------------- PEOPLE (EN) ----------------
 ("p_en1","people",False,False,"I would like to request parental leave from 2026-02-01 for twelve weeks, and I would like to understand the phased return-to-work option."),
 ("p_en2","people",False,True, "Following the review cycle, please confirm the new salary band for the team member and the effective date of the increase."),
 ("p_en3","people",False,False,"New starter begins on Monday. Could you make sure the laptop, the accounts and the first-week buddy schedule are all arranged before then?"),
 ("p_en4","people",False,True, "She has resigned and her notice period is three months under her contract. We should agree the handover plan and the last working day."),
 ("p_en5","people",False,True, "Requesting approval for the team to attend the training course; the cost is 1,200 per head and it would come from the development allowance."),
 # ---------------- FINANCE (DE) ----------------
 ("f_de1","finance",True,False,"Erinnerung: Rechnung 4471 ueber 1.992,00 EUR war am 30.09.2025 faellig und ist weiterhin offen. Bitte veranlassen Sie die Ueberweisung diese Woche."),
 ("f_de2","finance",True,False,"Bitte genehmigen Sie die Bestellung von 40 Alu Pro Staendern zu je 82,00 EUR. Der Betrag geht zulasten des Sachmittelbudgets im vierten Quartal."),
 ("f_de3","finance",True,False,"Ich reiche die Reisekostenabrechnung fuer Berlin ein: insgesamt 340,50 EUR, Belege fuer Hotel und Bahn liegen bei. Erstattung bitte auf das ueblich Konto."),
 ("f_de4","finance",True,True, "Die Umsatzsteuervoranmeldung fuer das dritte Quartal steht an. Die Vorschriften haben sich geaendert, daher moechte ich die Behandlung der Auslandsumsaetze klaeren."),
 ("f_de5","finance",True,False,"Der Lastschrifteinzug ist am 15. erneut gescheitert, das Mandat wurde offenbar widerrufen. Koennen Sie die Daten neu senden, damit wir den offenen Betrag ausgleichen?"),
 # ---------------- LEGAL (DE) ----------------
 ("l_de1","legal",True,True,  "Wir werden die Kuendigung nach Ziffer 11.2 mit einer Frist von 90 Tagen aussprechen. Bis dahin faellige Entgelte bleiben selbstverstaendlich zahlbar."),
 ("l_de2","legal",True,False, "Bitte pruefen Sie die beiliegende wechselseitige Geheimhaltungsvereinbarung bis Donnerstag, insbesondere die Definition vertraulicher Informationen und die Nachwirkung."),
 ("l_de3","legal",True,True,  "Der Auftragsverarbeitungsvertrag muss wegen des neuen Unterauftragnehmers angepasst werden. Personenbezogene Daten liegen jetzt in einer zweiten Region."),
 ("l_de4","legal",True,False, "Die Gegenseite lehnt die Mediation ab und ruft das Schiedsgericht an; vorgesehen ist ein Einzelschiedsrichter mit Sitz in London."),
 ("l_de5","legal",True,True,  "Frage zum Muster des Arbeitsvertrags: das Wettbewerbsverbot laeuft zwoelf Monate und duerfte unwirksam sein. Koennen wir es fuer neue Mitarbeiter verkuerzen?"),
 # ---------------- ENGINEERING (DE) ----------------
 ("e_de1","engineering",True,False,"Der Checkout liefert HTTP 500, sobald der Warenkorb mehr als 50 Positionen enthaelt. Der Stacktrace zeigt auf OrderValidator.java Zeile 212, reproduzierbar auf Staging."),
 ("e_de2","engineering",True,False,"Wir rollen Release 0.9.4 zurueck. Die Worker starten unter Last alle vier Minuten neu, und die Nullpointer-Ausnahme in SessionCache.evict steht in jedem Log."),
 ("e_de3","engineering",True,True, "Die naechtliche Migration dauert inzwischen sechs Stunden und der Abfrageplan zeigt einen vollen Scan. Das treibt auch die Datenbankkosten, wir sollten den Index setzen."),
 ("e_de4","engineering",True,False,"Sicherheitshinweis zur eingebundenen Bildbibliothek: ein praepariertes PNG kann den Decoder ueberlaufen lassen. Der Patch ist verfuegbar, wir sollten heute neu deployen."),
 ("e_de5","engineering",True,False,"Deren Schnittstelle antwortet ab etwa 300 Aufrufen pro Minute mit 429, und unsere Wiederholungslogik verschlimmert es. Wir brauchen ein Backoff vor dem Livegang."),
 # ---------------- PEOPLE (DE) ----------------
 ("p_de1","people",True,False,"Ich moechte Elternzeit ab dem 01.02.2026 fuer zwoelf Wochen beantragen und haette gern Informationen zum stufenweisen Wiedereinstieg."),
 ("p_de2","people",True,True, "Nach dem Beurteilungszyklus bitte ich um Bestaetigung der neuen Gehaltsstufe fuer die Mitarbeiterin und des Zeitpunkts der Erhoehung."),
 ("p_de3","people",True,False,"Der neue Kollege faengt am Montag an. Koennt ihr bitte Laptop, Zugaenge und den Paten fuer die erste Woche rechtzeitig organisieren?"),
 ("p_de4","people",True,True, "Sie hat gekuendigt, die Kuendigungsfrist betraegt laut Arbeitsvertrag drei Monate. Wir sollten die Uebergabe und den letzten Arbeitstag festlegen."),
 ("p_de5","people",True,True, "Ich bitte um Freigabe fuer die Weiterbildung des Teams; die Kosten betragen 1.200 EUR pro Person und kaemen aus dem Entwicklungsbudget."),
]

AGG = "max"   # set to "mean" to exercise the server-side reduction

def locate(doc, keys, retries=2):
    body = json.dumps({"document": doc, "key_vocabulary": keys, "top_k": 1,
                       "key_aggregation": AGG}).encode()
    for a in range(retries + 1):
        try:
            t = time.perf_counter()
            r = json.load(urllib.request.urlopen(
                urllib.request.Request(URL, body, {"Content-Type": "application/json"}),
                timeout=900))
            return r["hits"], (time.perf_counter() - t) * 1000.0
        except urllib.error.URLError:
            if a == retries: raise
            time.sleep(1.0)

def score_once(doc, cats, order):
    hits, ms = locate(doc, [cats[n] for n in order])
    out = {}
    for n in order:
        h = hits[cats[n]]
        out[n] = (h[0]["peak"], doc[h[0]["byte_lo"]:h[0]["byte_hi"]]) if h else (0.0, "")
    return out, ms

def report(title, results):
    """results: list of (tag, want, got, de, lure)."""
    def rate(sel):
        s = [r for r in results if sel(r)]
        return (100.0 * sum(r[2] == r[1] for r in s) / len(s), len(s)) if s else (0.0, 0)
    a, na = rate(lambda r: True)
    e, ne = rate(lambda r: not r[3])
    d, nd = rate(lambda r: r[3])
    lu, nl = rate(lambda r: r[4])
    cl, nc = rate(lambda r: not r[4])
    print(f"\n  {title}")
    print(f"    overall {a:5.1f}%  ({na})   EN {e:5.1f}% ({ne})   DE {d:5.1f}% ({nd})"
          f"   |  lure {lu:5.1f}% ({nl})   clean {cl:5.1f}% ({nc})")
    conf = {}
    for _, want, got, _, _ in results:
        conf.setdefault(want, {}).setdefault(got, 0)
        conf[want][got] += 1
    print("    confusion (row = true, col = picked):")
    print("              " + "".join(f"{n[:5]:>8}" for n in NAMES))
    for w in NAMES:
        row = conf.get(w, {})
        print(f"      {w:9} " + "".join(f"{row.get(g,0):>8}" for g in NAMES))
    bad = [(t, w, g) for t, w, g, _, _ in results if w != g]
    if bad:
        print("    misses: " + ", ".join(f"{t}({w}->{g})" for t, w, g in bad))

def main():
    global AGG
    if len(sys.argv) > 1 and sys.argv[1] in ("max", "mean"):
        AGG = sys.argv[1]
    print(f"[key_aggregation = {AGG}]")
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10).read()
    except Exception:
        print(f"FAIL: no server on :{PORT}. Start it with\n"
              f"  ./build-metal/bin/qwenium-server -m models/Qwen3.8-9B-Q8_0.gguf "
              f"-c 4096 -s 1 -p {PORT} --attention-lens --lens-locate-only")
        return 1

    print("DECIDE1 — choice as argmax over evidence presence")
    print(f"  {len(DOCS)} documents ({sum(1 for d in DOCS if not d[2])} EN / "
          f"{sum(1 for d in DOCS if d[2])} DE), {len(NAMES)} categories, "
          f"chance = {100.0/len(NAMES):.0f}%")
    print(f"  lures: {sum(1 for d in DOCS if d[3])} documents carry another category's vocabulary")

    rotations = [NAMES[i:] + NAMES[:i] for i in range(len(NAMES))]

    armA, armB, per_rot = [], [], {i: [] for i in range(len(rotations))}
    lat_a, lat_b = [], []
    for tag, want, de, lure, doc in DOCS:
        tot = {n: 0.0 for n in NAMES}
        el = 0.0
        for i, order in enumerate(rotations):
            sc, ms = score_once(doc, CATS_EN, order)
            el += ms
            for n in NAMES:
                tot[n] += sc[n][0]
            pick_i = max(sc, key=lambda n: sc[n][0])
            per_rot[i].append((tag, want, pick_i, de, lure))
            if i == 0:
                armA.append((tag, want, pick_i, de, lure))
                lat_a.append(ms)
        lat_b.append(el)
        armB.append((tag, want, max(tot, key=tot.get), de, lure))

    report("ARM A — ONE call, canonical order (1 prefill)", armA)
    print(f"    latency: median {statistics.median(lat_a):.0f} ms")
    report("ARM B — all rotations averaged (%d prefills)" % len(rotations), armB)
    print(f"    latency: median {statistics.median(lat_b):.0f} ms total")

    print("\n  POSITION CONTROL — accuracy under each single rotation")
    accs = []
    for i in range(len(rotations)):
        r = per_rot[i]
        acc = 100.0 * sum(x[2] == x[1] for x in r) / len(r)
        accs.append(acc)
        print(f"    order starting with {rotations[i][0]:12} {acc:5.1f}%")
    print(f"    spread {min(accs):.1f}-{max(accs):.1f}%  "
          f"(if this is wide, ARM A's number is an accident of one order)")

    # ── ARM C — content-word parts, ONE call, mean per category ─────────────
    def arm_parts(parts_map, rotate):
        res = []
        flat = [(c, p) for c in NAMES for p in parts_map[c]]
        lat = []
        for tag, want, de, lure, doc in DOCS:
            agg = {n: [] for n in NAMES}
            orders = ([flat] if not rotate
                      else [flat[i:] + flat[:i] for i in
                            range(0, len(flat), max(1, len(flat) // 4))])
            for od in orders:
                hits, ms = locate(doc, [p for _, p in od])
                lat.append(ms)
                for c, p in od:
                    h = hits[p]
                    agg[c].append(h[0]["peak"] if h else 0.0)
            means = {n: (statistics.mean(v) if v else 0.0) for n, v in agg.items()}
            res.append((tag, want, max(means, key=means.get), de, lure))
        return res, lat

    cres, clat = arm_parts(PARTS_EN, rotate=False)
    report("ARM C — content-word parts, ONE call, mean per category (1 prefill)", cres)
    print(f"    latency: median {statistics.median(clat):.0f} ms")
    cres_r, clat_r = arm_parts(PARTS_EN, rotate=True)
    report("ARM D — content-word parts, rotated, mean per category", cres_r)

    # Cross-lingual: do German documents need a German schema?
    de_docs = [d for d in DOCS if d[2]]
    de_en_desc, de_de_desc = [], []
    for tag, want, de, lure, doc in de_docs:
        for cats, sink in ((CATS_EN, de_en_desc), (CATS_DE, de_de_desc)):
            sc, _ = score_once(doc, cats, NAMES)
            sink.append((tag, want, max(sc, key=lambda n: sc[n][0]), de, lure))
    ae = 100.0 * sum(x[2] == x[1] for x in de_en_desc) / len(de_en_desc)
    ad = 100.0 * sum(x[2] == x[1] for x in de_de_desc) / len(de_de_desc)
    print(f"\n  CROSS-LINGUAL SCHEMA (German documents, {len(de_docs)} docs, canonical order)")
    print(f"    English category descriptions: {ae:5.1f}%")
    print(f"    German  category descriptions: {ad:5.1f}%")
    print("    (a client sends ONE schema to every language; this says whether it must be translated)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
