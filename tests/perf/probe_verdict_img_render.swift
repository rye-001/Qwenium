// VERDICT-IMG test images — synthetic delivery notes, no personal data.
//
// 5 base notes x 2^3 variants (signature in the "Received by" box, a stamp,
// the delivery date written in) = 40 PNGs, plus one blank page of the same
// size (the yes-bias baseline). Writes manifest.tsv next to the images:
//   image \t base \t family \t question \t expected(yes|no)
// Page 1024 x 1440 px = 32 x 45 merged patches = 1440 image tokens on the
// Qwen3-VL merger (above the ~1024-token grounding floor).
//
//   swiftc -O tests/perf/probe_verdict_img_render.swift -o .session-results/verdict_img/render
//   .session-results/verdict_img/render .session-results/verdict_img
//
// HARD arm (`render DIR hard`): per base and family, five variants of the
// asked mark — yes (clean), yes (hard: faint, or a stamp half off the page),
// no (empty), no (lure 1), no (lure 2) — the other two marks drawn clean at
// random; every image rendered twice: "c" (clean PNG) and "s" (a degraded
// scan: skew, blur, noise, paper tint, JPEG q=0.45). 150 images, all three
// questions on each. File name: {c|s}_b{base}_{FAM}_{variant}_s?t?d?.{png|jpg}
//   .session-results/verdict_img/render .session-results/verdict_img_hard hard

import AppKit
import Foundation

let W = 1024, H = 1440

struct Base {
    let company: String, street: String, city: String, note: String
    let items: [(String, String)]
    let dateText: String
    let stampAt: CGPoint            // stamp centre, top-left page coordinates
    let stampColor: NSColor
    let stampRound: Bool
    let seed: UInt64
    // Lure / hard-mark positions; the defaults are the §7 harder arm's.
    var logoAt = CGPoint(x: 860, y: 1270)
    var statusAt = CGPoint(x: 600, y: 250)
    var partialY: CGFloat = 760
}

let bases: [Base] = [
    Base(company: "Northwind Supplies GmbH", street: "Hafenstrasse 12", city: "20457 Hamburg",
         note: "DN-2026-0412", items: [("4", "Pallet wrap, 500 mm"), ("12", "Carton box, size L"),
         ("2", "Tape dispenser"), ("30", "Label roll, 100 x 50")], dateText: "12.03.2026",
         stampAt: CGPoint(x: 760, y: 470), stampColor: NSColor(red: 0.80, green: 0.10, blue: 0.12, alpha: 0.85),
         stampRound: true, seed: 11),
    Base(company: "Example Logistics Ltd", street: "7 Mill Lane", city: "Leeds LS1 4AB",
         note: "DN 88213", items: [("1", "Office chair, black"), ("6", "Desk lamp"),
         ("3", "Monitor arm")], dateText: "03/07/2026",
         stampAt: CGPoint(x: 300, y: 1000), stampColor: NSColor(red: 0.10, green: 0.20, blue: 0.70, alpha: 0.85),
         stampRound: false, seed: 23),
    Base(company: "Acme Components AG", street: "Industrieweg 4", city: "8400 Winterthur",
         note: "LS-55190", items: [("100", "Hex bolt M8 x 40"), ("100", "Washer M8"), ("50", "Nut M8"),
         ("10", "Bracket, galvanised"), ("2", "Toolbox")], dateText: "21.09.2026",
         stampAt: CGPoint(x: 720, y: 1040), stampColor: NSColor(red: 0.12, green: 0.45, blue: 0.20, alpha: 0.85),
         stampRound: true, seed: 37),
    Base(company: "Blue Harbour Foods", street: "Quay Road 3", city: "Cork T12 X",
         note: "DEL-7702", items: [("20", "Olive oil, 1 l"), ("40", "Pasta, 500 g"),
         ("15", "Tomato passata")], dateText: "30 Jan 2026",
         stampAt: CGPoint(x: 260, y: 520), stampColor: NSColor(red: 0.55, green: 0.10, blue: 0.55, alpha: 0.85),
         stampRound: false, seed: 41),
    Base(company: "Kestrel Print & Paper", street: "Unit 9, Riverside Park", city: "Bristol BS2 0QT",
         note: "No. 30417", items: [("5", "Copy paper A4, box"), ("2", "Toner, black"), ("8", "Envelope C4, pack"),
         ("1", "Laminator")], dateText: "14.11.2026",
         stampAt: CGPoint(x: 780, y: 760), stampColor: NSColor(red: 0.85, green: 0.35, blue: 0.05, alpha: 0.85),
         stampRound: true, seed: 53),
]

// FRESH bases for the stricter-stamp arm (§8): new companies, layouts, stamp
// and lure positions — none of the images of §4 or §7 is reused.
func rgb(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat) -> NSColor { NSColor(red: r, green: g, blue: b, alpha: 0.85) }
let freshBases: [Base] = [
    Base(company: "Harbourline Freight BV", street: "Kade 21", city: "3011 Rotterdam", note: "VB-10442",
         items: [("8", "Steel shelf, 180 cm"), ("16", "Shelf bracket")], dateText: "05.02.2026",
         stampAt: CGPoint(x: 740, y: 600), stampColor: rgb(0.15, 0.25, 0.75), stampRound: true, seed: 101,
         logoAt: CGPoint(x: 150, y: 1000), statusAt: CGPoint(x: 600, y: 320), partialY: 900),
    Base(company: "Meridian Office Goods", street: "12 Station Road", city: "Reading RG1 1AA", note: "MOG/2231",
         items: [("3", "Filing cabinet"), ("10", "Lever arch file"), ("4", "Desk organiser")], dateText: "19/05/2026",
         stampAt: CGPoint(x: 330, y: 820), stampColor: rgb(0.75, 0.10, 0.15), stampRound: false, seed: 103,
         logoAt: CGPoint(x: 860, y: 900), statusAt: CGPoint(x: 600, y: 250), partialY: 600),
    Base(company: "Alpenrand Baustoffe", street: "Talweg 8", city: "6020 Innsbruck", note: "LS 7781",
         items: [("40", "Cement bag, 25 kg"), ("2", "Wheelbarrow"), ("12", "Trowel")], dateText: "02.06.2026",
         stampAt: CGPoint(x: 760, y: 1000), stampColor: rgb(0.10, 0.45, 0.25), stampRound: true, seed: 107,
         logoAt: CGPoint(x: 150, y: 780), statusAt: CGPoint(x: 560, y: 700), partialY: 450),
    Base(company: "Sunfield Garden Supply", street: "44 Orchard Way", city: "Exeter EX2 5AB", note: "SGS-0917",
         items: [("25", "Compost, 40 l"), ("6", "Rake"), ("50", "Plant pot, 12 cm"), ("1", "Hose reel")],
         dateText: "11 Apr 2026",
         stampAt: CGPoint(x: 280, y: 950), stampColor: rgb(0.55, 0.15, 0.55), stampRound: true, seed: 109,
         logoAt: CGPoint(x: 850, y: 700), statusAt: CGPoint(x: 600, y: 320), partialY: 1000),
    Base(company: "Nordkap Elektro AS", street: "Brugata 3", city: "0186 Oslo", note: "PK-50013",
         items: [("200", "Cable tie"), ("5", "Extension lead"), ("20", "Wall socket")], dateText: "27.08.2026",
         stampAt: CGPoint(x: 770, y: 800), stampColor: rgb(0.85, 0.35, 0.05), stampRound: false, seed: 113,
         logoAt: CGPoint(x: 160, y: 900), statusAt: CGPoint(x: 600, y: 250), partialY: 520),
    Base(company: "Crescent Lab Supplies", street: "Unit 2, Science Park", city: "Cambridge CB4 0WS", note: "CLS 4410",
         items: [("12", "Pipette tips, box"), ("4", "Beaker, 500 ml"), ("2", "Lab coat, M")], dateText: "08/10/2026",
         stampAt: CGPoint(x: 720, y: 520), stampColor: rgb(0.10, 0.20, 0.60), stampRound: true, seed: 127,
         logoAt: CGPoint(x: 860, y: 1000), statusAt: CGPoint(x: 120, y: 1040), partialY: 850),
    Base(company: "Rheinblick Getraenke", street: "Uferstrasse 30", city: "50668 Koeln", note: "LI-66021",
         items: [("30", "Mineral water, crate"), ("10", "Apple juice, crate")], dateText: "15.07.2026",
         stampAt: CGPoint(x: 300, y: 700), stampColor: rgb(0.80, 0.10, 0.12), stampRound: true, seed: 131,
         logoAt: CGPoint(x: 860, y: 780), statusAt: CGPoint(x: 600, y: 320), partialY: 1050),
    Base(company: "Kingfisher Textiles", street: "9 Weavers Row", city: "Manchester M4 6DE", note: "KT-8830",
         items: [("15", "Cotton roll"), ("40", "Thread spool"), ("6", "Fabric shears")], dateText: "23 Mar 2026",
         stampAt: CGPoint(x: 740, y: 1050), stampColor: rgb(0.12, 0.40, 0.45), stampRound: false, seed: 137,
         logoAt: CGPoint(x: 150, y: 820), statusAt: CGPoint(x: 600, y: 250), partialY: 640),
    Base(company: "Lagune Fournitures SARL", street: "5 rue du Port", city: "13002 Marseille", note: "BL 2026-118",
         items: [("100", "Envelope DL"), ("20", "Notebook A5"), ("5", "Stapler")], dateText: "30.09.2026",
         stampAt: CGPoint(x: 760, y: 700), stampColor: rgb(0.20, 0.20, 0.70), stampRound: true, seed: 139,
         logoAt: CGPoint(x: 160, y: 1000), statusAt: CGPoint(x: 560, y: 1000), partialY: 950),
    Base(company: "Pinecrest Hardware", street: "210 Mill Street", city: "Galway H91 X2", note: "PH-3307",
         items: [("60", "Wood screw, 4 x 40"), ("3", "Hammer"), ("8", "Hinge, brass"), ("2", "Spirit level")],
         dateText: "04/12/2026",
         stampAt: CGPoint(x: 320, y: 1000), stampColor: rgb(0.60, 0.25, 0.10), stampRound: true, seed: 149,
         logoAt: CGPoint(x: 860, y: 850), statusAt: CGPoint(x: 600, y: 320), partialY: 560),
]

// PAPER bases for the real-paper arm (§9): 6 delivery notes + 6 invoices, new
// companies, never used for any image of §4, §7 or §8.
let paperBases: [Base] = [
    Base(company: "Tideway Packaging Ltd", street: "3 Canal Wharf", city: "Bristol BS1 6XY", note: "DN 40117",
         items: [("6", "Bubble wrap roll"), ("20", "Mailing box, M"), ("4", "Packing tape")], dateText: "09.10.2026",
         stampAt: CGPoint(x: 760, y: 760), stampColor: rgb(0.80, 0.10, 0.12), stampRound: true, seed: 201,
         logoAt: CGPoint(x: 860, y: 900), statusAt: CGPoint(x: 600, y: 700), partialY: 800),
    Base(company: "Eisenwerk Kramer KG", street: "Hallenweg 17", city: "44135 Dortmund", note: "LS-90318",
         items: [("12", "Steel angle, 2 m"), ("50", "Bolt M10"), ("50", "Nut M10")], dateText: "17.10.2026",
         stampAt: CGPoint(x: 330, y: 900), stampColor: rgb(0.15, 0.25, 0.75), stampRound: false, seed: 203,
         logoAt: CGPoint(x: 860, y: 900), statusAt: CGPoint(x: 600, y: 700), partialY: 800),
    Base(company: "Brightleaf Stationers", street: "18 High Street", city: "York YO1 8RL", note: "BS-2275",
         items: [("10", "Ring binder"), ("5", "Whiteboard marker, 4-pack")], dateText: "22.10.2026",
         stampAt: CGPoint(x: 760, y: 900), stampColor: rgb(0.10, 0.45, 0.25), stampRound: true, seed: 207,
         logoAt: CGPoint(x: 860, y: 900), statusAt: CGPoint(x: 600, y: 700), partialY: 800),
    Base(company: "Nordwind Moebel GmbH", street: "Am Deich 5", city: "26123 Oldenburg", note: "LS 11820",
         items: [("2", "Bookcase, oak"), ("4", "Chair, grey"), ("1", "Table, 160 cm")], dateText: "01.10.2026",
         stampAt: CGPoint(x: 300, y: 800), stampColor: rgb(0.55, 0.15, 0.55), stampRound: true, seed: 211,
         logoAt: CGPoint(x: 860, y: 850), statusAt: CGPoint(x: 600, y: 700), partialY: 800),
    Base(company: "Coastal Marine Parts", street: "Pier 4, Harbour Road", city: "Plymouth PL1 3DE", note: "CMP/7702",
         items: [("8", "Rope, 20 m"), ("16", "Shackle, 10 mm"), ("2", "Fender")], dateText: "06.10.2026",
         stampAt: CGPoint(x: 760, y: 820), stampColor: rgb(0.85, 0.35, 0.05), stampRound: false, seed: 213,
         logoAt: CGPoint(x: 150, y: 900), statusAt: CGPoint(x: 600, y: 700), partialY: 800),
    Base(company: "Hofgut Sonnenberg", street: "Feldweg 2", city: "79098 Freiburg", note: "LS-3304",
         items: [("30", "Apple juice, 1 l"), ("12", "Honey, 500 g")], dateText: "12.10.2026",
         stampAt: CGPoint(x: 330, y: 850), stampColor: rgb(0.12, 0.40, 0.45), stampRound: true, seed: 217,
         logoAt: CGPoint(x: 860, y: 850), statusAt: CGPoint(x: 560, y: 820), partialY: 800),
    Base(company: "Lindqvist Consulting AB", street: "Storgatan 9", city: "411 38 Goeteborg", note: "INV 2026-0391",
         items: [("12", "Consulting, hours"), ("1", "Travel expenses")], dateText: "02.09.2026",
         stampAt: CGPoint(x: 760, y: 850), stampColor: rgb(0.80, 0.10, 0.12), stampRound: true, seed: 223,
         logoAt: CGPoint(x: 860, y: 950), statusAt: CGPoint(x: 560, y: 820), partialY: 800),
    Base(company: "Greenline Cleaning Co", street: "22 Park Avenue", city: "Leeds LS6 2AB", note: "Invoice 5518",
         items: [("4", "Office cleaning, visits"), ("1", "Window cleaning")], dateText: "15.08.2026",
         stampAt: CGPoint(x: 330, y: 850), stampColor: rgb(0.15, 0.25, 0.75), stampRound: false, seed: 227,
         logoAt: CGPoint(x: 860, y: 950), statusAt: CGPoint(x: 560, y: 820), partialY: 800),
    Base(company: "Atelier Bauer Druck", street: "Kirchplatz 3", city: "93047 Regensburg", note: "RE-20611",
         items: [("500", "Flyer A5"), ("100", "Poster A2"), ("1", "Layout")], dateText: "21.07.2026",
         stampAt: CGPoint(x: 760, y: 900), stampColor: rgb(0.10, 0.45, 0.25), stampRound: true, seed: 229,
         logoAt: CGPoint(x: 860, y: 950), statusAt: CGPoint(x: 560, y: 900), partialY: 800),
    Base(company: "Harbor Point IT Services", street: "7 Dock Street", city: "Dublin D01 K2", note: "HP-INV-884",
         items: [("3", "Laptop setup"), ("1", "Network check")], dateText: "30.06.2026",
         stampAt: CGPoint(x: 330, y: 900), stampColor: rgb(0.55, 0.15, 0.55), stampRound: true, seed: 233,
         logoAt: CGPoint(x: 860, y: 950), statusAt: CGPoint(x: 560, y: 820), partialY: 800),
    Base(company: "Fjell Outdoor AS", street: "Fjellveien 12", city: "5003 Bergen", note: "FAKTURA 7713",
         items: [("6", "Tent, 2-person"), ("12", "Sleeping mat")], dateText: "11.09.2026",
         stampAt: CGPoint(x: 760, y: 900), stampColor: rgb(0.85, 0.35, 0.05), stampRound: false, seed: 239,
         logoAt: CGPoint(x: 860, y: 950), statusAt: CGPoint(x: 560, y: 820), partialY: 800),
    Base(company: "Maison Clair Traiteur", street: "8 rue Verte", city: "69002 Lyon", note: "FA-2026-145",
         items: [("40", "Lunch menu"), ("40", "Dessert")], dateText: "05.09.2026",
         stampAt: CGPoint(x: 330, y: 900), stampColor: rgb(0.12, 0.40, 0.45), stampRound: true, seed: 241,
         logoAt: CGPoint(x: 860, y: 950), statusAt: CGPoint(x: 560, y: 820), partialY: 800),
]

// One printable sheet: what is printed, what the user adds by hand, the truth.
struct Sheet {
    let base: Base, invoice: Bool
    let printed: Marks            // printed stamps and lures (pen marks are never printed)
    let sign: String?             // "pen" | "pencil" | nil — added by hand
    let date: String?             // the date to write, pen; nil = leave empty
    let datePencil: Bool
    let printedNote: String       // for the checklist
}

// A4 PDF, vector: each page is drawPage scaled uniformly into 595 x 842 pt.
func writePaperPDF(_ sheets: [Sheet], to path: String) {
    var box = CGRect(x: 0, y: 0, width: 595, height: 842)
    let pdf = CGContext(URL(fileURLWithPath: path) as CFURL, mediaBox: &box, nil)!
    let sc = min(595 / CGFloat(W), 842 / CGFloat(H))
    for (i, sh) in sheets.enumerated() {
        pdf.beginPDFPage(nil)
        pdf.saveGState()
        pdf.translateBy(x: (595 - CGFloat(W) * sc) / 2, y: 842 - (842 - CGFloat(H) * sc) / 2)
        pdf.scaleBy(x: sc, y: -sc)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(cgContext: pdf, flipped: true)
        drawPage(sh.base, sh.printed, pdf, invoice: sh.invoice, footer: String(format: "Sheet %02d", i + 1))
        NSGraphicsContext.restoreGraphicsState()
        pdf.restoreGState()
        pdf.endPDFPage()
    }
    pdf.closePDF()
}

struct Rng { var s: UInt64
    mutating func next() -> Double { s = s &* 6364136223846793005 &+ 1442695040888963407
        return Double(s >> 11) / Double(1 << 53) } }

func font(_ name: String, _ size: CGFloat, bold: Bool = false) -> NSFont {
    if let f = NSFont(name: name, size: size) { return f }
    return bold ? NSFont.boldSystemFont(ofSize: size) : NSFont.systemFont(ofSize: size)
}

func text(_ s: String, _ x: CGFloat, _ y: CGFloat, _ f: NSFont, _ c: NSColor = .black) {
    NSAttributedString(string: s, attributes: [.font: f, .foregroundColor: c]).draw(at: NSPoint(x: x, y: y))
}

func line(_ x0: CGFloat, _ y0: CGFloat, _ x1: CGFloat, _ y1: CGFloat, _ w: CGFloat = 1.5) {
    let p = NSBezierPath(); p.move(to: NSPoint(x: x0, y: y0)); p.line(to: NSPoint(x: x1, y: y1))
    p.lineWidth = w; NSColor.black.setStroke(); p.stroke()
}

enum Sig { case none, normal, faint }
enum SigLure { case none, printedName, issuerSignature }
enum Stamp { case none, normal, faint, partial }
enum StampLure { case none, logo, printedStatus }
enum DateMark { case none, normal, faint }
enum DateLure { case none, orderDate, placeholder }

struct Marks {
    var sig = Sig.none, sigLure = SigLure.none
    var stamp = Stamp.none, stampLure = StampLure.none
    var date = DateMark.none, dateLure = DateLure.none
}

func render(_ b: Base?, signed: Bool, stamped: Bool, dated: Bool, to path: String) {
    var m = Marks()
    m.sig = signed ? .normal : .none
    m.stamp = stamped ? .normal : .none
    m.date = dated ? .normal : .none
    let rep = draw(b, m)
    let png = rep.representation(using: .png, properties: [:])!
    try! png.write(to: URL(fileURLWithPath: path))
}

func scribble(_ rng: inout Rng, from x0: CGFloat, midY: CGFloat, segments: Int) -> NSBezierPath {
    let s = NSBezierPath()
    var x = x0
    var y = midY + 10
    s.move(to: NSPoint(x: x, y: y))
    for _ in 0..<segments {
        let nx = x + 30 + CGFloat(rng.next()) * 25
        let ny = midY - 30 + CGFloat(rng.next()) * 60
        s.curve(to: NSPoint(x: nx, y: ny),
                controlPoint1: NSPoint(x: x + 10, y: y - 45 + CGFloat(rng.next()) * 20),
                controlPoint2: NSPoint(x: nx - 10, y: ny + 45 - CGFloat(rng.next()) * 20))
        x = nx; y = ny
    }
    return s
}

func draw(_ b: Base?, _ m: Marks) -> NSBitmapImageRep {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: W, pixelsHigh: H, bitsPerSample: 8,
                               samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                               colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    let ctx = NSGraphicsContext(bitmapImageRep: rep)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = ctx
    // Flip to top-left origin so page coordinates read like a document.
    let cg = ctx.cgContext
    cg.translateBy(x: 0, y: CGFloat(H)); cg.scaleBy(x: 1, y: -1)
    NSGraphicsContext.current = NSGraphicsContext(cgContext: cg, flipped: true)
    drawPage(b, m, cg)
    NSGraphicsContext.restoreGraphicsState()
    return rep
}

// One page in top-left page coordinates (W x H units) into a flipped context —
// shared by the bitmap path (draw) and the printable PDF (writePaperPDF).
// `invoice` switches the delivery-note layout to a bill; `footer` is the
// printed sheet number. Both are off for every image of §4, §7 and §8.
func drawPage(_ b: Base?, _ m: Marks, _ cg: CGContext, invoice: Bool = false, footer: String? = nil) {
    NSColor.white.setFill(); NSRect(x: 0, y: 0, width: W, height: H).fill()

    if let b = b {
        var rng = Rng(s: b.seed)
        let body = font("Helvetica", 22), bold = font("Helvetica-Bold", 22, bold: true)
        text(b.company, 70, 70, font("Helvetica-Bold", 38, bold: true))
        text(b.street, 70, 125, body); text(b.city, 70, 155, body)
        text(invoice ? "INVOICE" : "DELIVERY NOTE", 640, 70, font("Helvetica-Bold", 34, bold: true))
        text(b.note, 640, 125, body)
        line(70, 210, 954, 210, 2)

        text(invoice ? "Date paid:" : "Delivery date:", 70, 250, bold)
        if invoice { text("Invoice date:", 600, 250, bold); text(b.dateText, 760, 250, body) }
        line(240, 278, 520, 278)
        if m.date == .normal {
            text(b.dateText, 255, 238, font("Bradley Hand", 34), NSColor(red: 0.05, green: 0.1, blue: 0.55, alpha: 1))
        } else if m.date == .faint {
            text(b.dateText, 262, 248, font("Bradley Hand", 22), NSColor(white: 0.62, alpha: 1))
        }
        if m.dateLure == .placeholder {
            text("DD.MM.YYYY", 262, 250, font("Helvetica", 20), NSColor(white: 0.72, alpha: 1))
        }
        text("Order ref:", 70, 310, bold); text("PO-\(b.seed * 97 + 1000)", 200, 310, body)
        if m.dateLure == .orderDate {
            text("Order date:", 600, 310, bold); text(b.dateText, 740, 310, body)
        }
        if m.stampLure == .printedStatus {
            text(invoice ? "Status: PAID" : "Status: RECEIVED", b.statusAt.x, b.statusAt.y, font("Helvetica-Bold", 26, bold: true), b.stampColor)
        }
        if m.sigLure == .issuerSignature {
            text("Issued by:", 600, 250, bold)
            var r2 = Rng(s: b.seed &+ 999)
            let s2 = scribble(&r2, from: 730, midY: 262, segments: 6)
            s2.lineWidth = 3.0; s2.lineCapStyle = .round
            NSColor(red: 0.05, green: 0.1, blue: 0.5, alpha: 1).setStroke(); s2.stroke()
        }

        // Items table.
        let top: CGFloat = 380
        text("Qty", 70, top, bold); text("Description", 200, top, bold)
        line(70, top + 34, 954, top + 34)
        for (i, it) in b.items.enumerated() {
            let y = top + 50 + CGFloat(i) * 42
            text(it.0, 70, y, body); text(it.1, 200, y, body)
        }
        if invoice {
            // Amount column and total — a bill, but nothing any question reads.
            text("Amount", 800, top, bold)
            var total = 0.0
            for (i, it) in b.items.enumerated() {
                let amount = (Double(it.0) ?? 1) * (3.5 + Double(i) * 1.25)
                total += amount
                text(String(format: "EUR %.2f", amount), 800, top + 50 + CGFloat(i) * 42, body)
            }
            let ty = top + 50 + CGFloat(b.items.count) * 42 + 10
            line(700, ty, 954, ty)
            text("Total", 700, ty + 12, bold); text(String(format: "EUR %.2f", total), 800, ty + 12, bold)
        }

        // Received-by box.
        text(invoice ? "Payment due within 14 days." : "Goods received in good condition.", 70, 1130, body)
        text(invoice ? "Approved by (signature):" : "Received by (signature):", 70, 1180, bold)
        let box = NSRect(x: 70, y: 1215, width: 480, height: 130)
        let bp = NSBezierPath(rect: box); bp.lineWidth = 1.5; NSColor.black.setStroke(); bp.stroke()
        text("Name / date", 70, 1352, font("Helvetica", 16), .darkGray)
        if m.sig == .normal {
            let x0 = box.minX + 40 + CGFloat(rng.next()) * 30
            let sp = scribble(&rng, from: x0, midY: box.midY, segments: 9)
            sp.lineWidth = 3.2; sp.lineCapStyle = .round
            NSColor(red: 0.05, green: 0.1, blue: 0.5, alpha: 1).setStroke(); sp.stroke()
        } else if m.sig == .faint {
            let sp = scribble(&rng, from: box.minX + 60, midY: box.midY, segments: 5)
            sp.lineWidth = 1.4; sp.lineCapStyle = .round
            NSColor(white: 0.6, alpha: 1).setStroke(); sp.stroke()
        }
        if m.sigLure == .printedName {
            text("J. Miller", box.minX + 30, box.midY - 16, font("Helvetica", 28))
        }
        if m.stamp != .none {
            let rot = CGFloat(-0.25 + rng.next() * 0.5)
            let at = m.stamp == .partial ? CGPoint(x: CGFloat(W) - 30, y: b.partialY) : b.stampAt
            let color = m.stamp == .faint ? b.stampColor.withAlphaComponent(0.28) : b.stampColor
            drawStamp(cg, at: at, rot: rot, color: color, round: b.stampRound, label: invoice ? "PAID" : "RECEIVED",
                      sub: b.company.components(separatedBy: " ").first!.uppercased())
        }
        if m.stampLure == .logo {
            // A company logo: a coloured ring with the initials — stamp-shaped, not a stamp.
            let initials = String(b.company.split(separator: " ").prefix(2).map { $0.first! })
            cg.saveGState()
            cg.translateBy(x: b.logoAt.x, y: b.logoAt.y)
            b.stampColor.setFill()
            NSBezierPath(ovalIn: NSRect(x: -75, y: -75, width: 150, height: 150)).fill()
            let lf = NSAttributedString(string: initials, attributes: [.font: font("Helvetica-Bold", 54, bold: true),
                                                                      .foregroundColor: NSColor.white])
            lf.draw(at: NSPoint(x: -lf.size().width / 2, y: -lf.size().height / 2))
            cg.restoreGState()
        }
    }
    if let f = footer { text(f, 880, 1400, font("Helvetica", 14), .gray) }
}

func drawStamp(_ cg: CGContext, at: CGPoint, rot: CGFloat, color: NSColor, round: Bool, label: String, sub: String) {
    cg.saveGState()
    cg.translateBy(x: at.x, y: at.y)
    cg.rotate(by: rot)
    color.setStroke()
    let outer: NSBezierPath = round
        ? NSBezierPath(ovalIn: NSRect(x: -110, y: -110, width: 220, height: 220))
        : NSBezierPath(roundedRect: NSRect(x: -150, y: -70, width: 300, height: 140), xRadius: 12, yRadius: 12)
    outer.lineWidth = 6; outer.stroke()
    let inner: NSBezierPath = round
        ? NSBezierPath(ovalIn: NSRect(x: -92, y: -92, width: 184, height: 184))
        : NSBezierPath(roundedRect: NSRect(x: -138, y: -58, width: 276, height: 116), xRadius: 8, yRadius: 8)
    inner.lineWidth = 2.5; inner.stroke()
    let sf = font("Helvetica-Bold", 30, bold: true)
    let l = NSAttributedString(string: label, attributes: [.font: sf, .foregroundColor: color])
    l.draw(at: NSPoint(x: -l.size().width / 2, y: -28))
    let sb = NSAttributedString(string: sub, attributes: [.font: font("Helvetica-Bold", 18, bold: true), .foregroundColor: color])
    sb.draw(at: NSPoint(x: -sb.size().width / 2, y: 10))
    cg.restoreGState()
}

// A degraded scan: skew, blur, contrast loss, noise, paper tint, heavy JPEG.
func scan(_ rep: NSBitmapImageRep, seed: UInt64) -> Data {
    var rng = Rng(s: seed)
    let src = CIImage(bitmapImageRep: rep)!
    let ext = src.extent
    let angle = CGFloat(rng.next() - 0.5) * 0.05          // about ±1.4°
    let t = CGAffineTransform(translationX: ext.midX, y: ext.midY).rotated(by: angle)
        .translatedBy(x: -ext.midX, y: -ext.midY)
    let white = CIImage(color: CIColor(red: 1, green: 1, blue: 1)).cropped(to: ext)
    var img = src.transformed(by: t).composited(over: white).cropped(to: ext)
    img = img.applyingFilter("CIColorControls", parameters: [kCIInputContrastKey: 0.8,
                                                             kCIInputBrightnessKey: -0.03,
                                                             kCIInputSaturationKey: 0.6])
    img = img.applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 1.1]).cropped(to: ext)
    let off = CGAffineTransform(translationX: CGFloat(rng.next() * 500), y: CGFloat(rng.next() * 500))
    let noise = CIFilter(name: "CIRandomGenerator")!.outputImage!.transformed(by: off).cropped(to: ext)
        .applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0.10, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0.10, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 0.10, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
            "inputBiasVector": CIVector(x: -0.05, y: -0.05, z: -0.05, w: 0)])
    img = noise.applyingFilter("CIAdditionCompositing", parameters: [kCIInputBackgroundImageKey: img]).cropped(to: ext)
    let tint = CIImage(color: CIColor(red: 0.96, green: 0.94, blue: 0.87)).cropped(to: ext)
    img = img.applyingFilter("CIMultiplyCompositing", parameters: [kCIInputBackgroundImageKey: tint]).cropped(to: ext)
    let cgOut = CIContext().createCGImage(img, from: ext)!
    return NSBitmapImageRep(cgImage: cgOut).representation(using: .jpeg, properties: [.compressionFactor: 0.45])!
}

let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "."
try! FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)

let hard = CommandLine.arguments.count > 2 && CommandLine.arguments[2] == "hard"
let questions: [(String, String)] = [
    ("SIG", hard ? "Is the delivery note signed by hand in the 'Received by' box?"
                 : "Is the delivery note signed in the 'Received by' box?"),
    ("STAMP", "Does the delivery note carry a stamp?"),
    ("DATE", "Is the delivery date filled in?"),
]
var manifest = ""
if CommandLine.arguments.count > 2 && CommandLine.arguments[2] == "paper" {
    // §9: printable sheets; the user adds pen marks, then photographs / scans.
    func pm(stamp: Stamp = .none, stampLure: StampLure = .none, sigLure: SigLure = .none,
            dateLure: DateLure = .none) -> Marks {
        var m = Marks(); m.stamp = stamp; m.stampLure = stampLure; m.sigLure = sigLure; m.dateLure = dateLure
        return m
    }
    let pb = paperBases
    let sheets: [Sheet] = [
        Sheet(base: pb[0], invoice: false, printed: pm(), sign: "pen", date: "14.10.2026", datePencil: false,
              printedNote: "nothing extra"),
        Sheet(base: pb[1], invoice: false, printed: pm(stamp: .normal), sign: nil, date: nil, datePencil: false,
              printedNote: "a stamp"),
        Sheet(base: pb[2], invoice: false, printed: pm(sigLure: .printedName), sign: nil, date: "03.11.2026",
              datePencil: false, printedNote: "a typed name in the signature box (lure)"),
        Sheet(base: pb[3], invoice: false, printed: pm(stampLure: .logo, sigLure: .issuerSignature), sign: nil,
              date: nil, datePencil: false, printedNote: "a round logo (lure) and a printed 'Issued by' signature (lure)"),
        Sheet(base: pb[4], invoice: false, printed: pm(stamp: .normal, dateLure: .orderDate), sign: "pencil",
              date: nil, datePencil: false, printedNote: "a stamp and an order date printed elsewhere (lure)"),
        Sheet(base: pb[5], invoice: false, printed: pm(stampLure: .printedStatus, dateLure: .placeholder),
              sign: "pen", date: nil, datePencil: false,
              printedNote: "'Status: RECEIVED' text (lure) and a DD.MM.YYYY placeholder (lure)"),
        Sheet(base: pb[6], invoice: true, printed: pm(), sign: "pen", date: "20.09.2026", datePencil: false,
              printedNote: "nothing extra (every invoice prints its invoice date — a lure for 'Date paid')"),
        Sheet(base: pb[7], invoice: true, printed: pm(stamp: .normal), sign: nil, date: "28.08.2026",
              datePencil: false, printedNote: "a PAID stamp"),
        Sheet(base: pb[8], invoice: true, printed: pm(stampLure: .printedStatus, sigLure: .printedName), sign: nil,
              date: nil, datePencil: false, printedNote: "'Status: PAID' text (lure) and a typed name in the box (lure)"),
        Sheet(base: pb[9], invoice: true, printed: pm(stampLure: .logo, dateLure: .placeholder), sign: "pen",
              date: nil, datePencil: false, printedNote: "a round logo (lure) and a DD.MM.YYYY placeholder (lure)"),
        Sheet(base: pb[10], invoice: true, printed: pm(stamp: .normal), sign: nil, date: "25.09.2026",
              datePencil: true, printedNote: "a PAID stamp"),
        Sheet(base: pb[11], invoice: true, printed: pm(stamp: .faint), sign: "pencil", date: nil, datePencil: false,
              printedNote: "a faint PAID stamp"),
    ]
    writePaperPDF(sheets, to: out + "/paper_sheets.pdf")

    var truth = "sheet\tkind\tfamily\tquestion\texpected\n"
    var check = """
    # Real-paper test — what to do with each sheet

    Print `paper_sheets.pdf` on A4, **in colour if you can** (say if it was black
    and white). Then add the pen/pencil marks below — nothing else. The
    signature is **a made-up scribble, never your real signature**. Then, for
    every sheet:

    1. **Photo** with your phone: the whole page in view, lying flat, normal room
       light, no flash. Name it `NN_photo.<ext>` (NN = the sheet number, 01–12).
    2. **Scan** it if you can (a scanner, or the phone's document-scan feature,
       e.g. iPhone Notes → Scan Documents): `NN_scan.<ext>`.

    Any image format is fine (HEIC included; it is converted locally). Put the
    files in `temp/paper/`. They stay on this machine. If you deviate from the
    list (a mark missing, a smudge), note it — the truth table follows the list.

    | sheet | printed on it | sign in the box | write in the date field |
    |---|---|---|---|

    """
    for (i, sh) in sheets.enumerated() {
        let n = String(format: "%02d", i + 1)
        let kind = sh.invoice ? "invoice" : "delivery"
        let signTxt = sh.sign == "pen" ? "**yes, pen**" : sh.sign == "pencil" ? "**yes, light pencil, small**" : "—"
        let dateTxt = sh.date.map { "**\($0)**" + (sh.datePencil ? ", light pencil, small" : ", pen") } ?? "—"
        check += "| \(n) (\(kind)) | \(sh.printedNote) | \(signTxt) | \(dateTxt) |\n"
        let qs: [(String, String, Bool)] = sh.invoice ? [
            ("SIG", "Is the invoice signed by hand in the 'Approved by' box?", sh.sign != nil),
            ("STAMP", "Does the invoice carry a stamp?", sh.printed.stamp != .none),
            ("DATE", "Is the 'Date paid' field filled in?", sh.date != nil),
        ] : [
            ("SIG", "Is the delivery note signed by hand in the 'Received by' box?", sh.sign != nil),
            ("STAMP", "Does the delivery note carry a stamp?", sh.printed.stamp != .none),
            ("DATE", "Is the delivery date filled in?", sh.date != nil),
        ]
        for (f, q, t) in qs { truth += "\(n)\t\(kind)\t\(f)\t\(q)\t\(t ? "yes" : "no")\n" }
    }
    try! truth.write(toFile: out + "/truth.tsv", atomically: true, encoding: .utf8)
    try! check.write(toFile: out + "/checklist.md", atomically: true, encoding: .utf8)
    print("wrote \(sheets.count) sheets: paper_sheets.pdf, checklist.md, truth.tsv in \(out)")
    exit(0)
}
if CommandLine.arguments.count > 2 && CommandLine.arguments[2] == "stamp2" {
    // §8: stamp only, fresh bases, the original and the stricter question on every image.
    let qs: [(String, String)] = [
        ("STAMP", "Does the delivery note carry a stamp?"),
        ("STAMPX", "Has the delivery note been stamped with an ink stamp (a rubber-stamp impression, "
                 + "not a printed logo or printed text)?"),
    ]
    for (bi, b) in freshBases.enumerated() {
        for (vi, v) in ["yes", "faint", "partial", "no", "lure1", "lure2"].enumerated() {
            var rng = Rng(s: b.seed &* 1000 &+ UInt64(vi))
            var m = Marks()
            m.sig = rng.next() < 0.5 ? .normal : .none
            m.date = rng.next() < 0.5 ? .normal : .none
            switch v {
            case "yes": m.stamp = .normal
            case "faint": m.stamp = .faint
            case "partial": m.stamp = .partial
            case "lure1": m.stampLure = .logo
            case "lure2": m.stampLure = .printedStatus
            default: break
            }
            let stamped = m.stamp != .none
            let tag = "f\(bi)_STAMP_\(v)_s\(m.sig != .none ? 1 : 0)t\(stamped ? 1 : 0)d\(m.date != .none ? 1 : 0)"
            let rep = draw(b, m)
            let cname = "c_\(tag).png", sname = "s_\(tag).jpg"
            try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out + "/" + cname))
            try! scan(rep, seed: b.seed &* 7919 &+ UInt64(vi)).write(to: URL(fileURLWithPath: out + "/" + sname))
            for name in [cname, sname] {
                for (qf, q) in qs { manifest += "\(name)\t\(bi)\t\(qf)\t\(q)\t\(stamped ? "yes" : "no")\n" }
            }
        }
    }
    try! manifest.write(toFile: out + "/manifest.tsv", atomically: true, encoding: .utf8)
    print("wrote \(manifest.split(separator: "\n").count / 2) images, \(manifest.split(separator: "\n").count) questions to \(out)")
    exit(0)
}
if CommandLine.arguments.count > 2 && CommandLine.arguments[2] == "stamp3" {
    // docs/note-stamp-lures.md: the two stamp-shaped lures the client found (2026-10-01) on §8's
    // layouts — a printed badge drawn like a stamp (double ring + one word,
    // upright) and a stamp on the back showing through
    // (mirrored, alpha 0.22 and 0.12) — against stamp / faint stamp / none.
    // The badge sits where §7/§8 put the logo lure.
    let qs: [(String, String)] = [
        ("STAMP", "Does the delivery note carry a stamp?"),
        ("STAMPX", "Has the delivery note been stamped with an ink stamp (a rubber-stamp impression, "
                 + "not a printed logo or printed text)?"),
    ]
    let words = ["URGENT", "ORIGINAL", "PRIORITY", "EXPRESS", "COPY"]
    for (bi, b) in freshBases.enumerated() {
        for (vi, v) in ["yes", "faint", "no", "badge", "ghost22", "ghost12"].enumerated() {
            var rng = Rng(s: b.seed &* 3000 &+ UInt64(vi))
            var m = Marks()
            m.sig = rng.next() < 0.5 ? .normal : .none
            m.date = rng.next() < 0.5 ? .normal : .none
            if v == "yes" { m.stamp = .normal }
            if v == "faint" { m.stamp = .faint }
            let stamped = m.stamp != .none
            let tag = "g\(bi)_STAMP_\(v)_s\(m.sig != .none ? 1 : 0)t\(stamped ? 1 : 0)d\(m.date != .none ? 1 : 0)"
            let rep = draw(b, m)
            let ctx = NSGraphicsContext(bitmapImageRep: rep)!
            NSGraphicsContext.saveGraphicsState()
            let cg = ctx.cgContext
            cg.translateBy(x: 0, y: CGFloat(H)); cg.scaleBy(x: 1, y: -1)
            NSGraphicsContext.current = NSGraphicsContext(cgContext: cg, flipped: true)
            if v == "badge" {
                // Printed by the form: upright, 80% of a stamp's size, in the logo's place.
                cg.saveGState(); cg.translateBy(x: b.logoAt.x, y: b.logoAt.y); cg.scaleBy(x: 0.8, y: 0.8)
                drawStamp(cg, at: .zero, rot: 0, color: b.stampColor, round: b.stampRound, label: words[bi % words.count], sub: "")
                cg.restoreGState()
            } else if v.hasPrefix("ghost") {
                // A stamp on the back of the sheet: mirrored, faint, where a stamp would sit.
                let alpha: CGFloat = v == "ghost22" ? 0.22 : 0.12
                cg.saveGState(); cg.translateBy(x: b.stampAt.x, y: b.stampAt.y); cg.scaleBy(x: -1, y: 1)
                drawStamp(cg, at: .zero, rot: CGFloat(-0.25 + rng.next() * 0.5), color: b.stampColor.withAlphaComponent(alpha),
                          round: b.stampRound, label: "RECEIVED", sub: b.company.components(separatedBy: " ").first!.uppercased())
                cg.restoreGState()
            }
            NSGraphicsContext.restoreGraphicsState()
            let cname = "c_\(tag).png", sname = "s_\(tag).jpg"
            try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out + "/" + cname))
            try! scan(rep, seed: b.seed &* 7907 &+ UInt64(vi)).write(to: URL(fileURLWithPath: out + "/" + sname))
            for name in [cname, sname] {
                for (qf, q) in qs { manifest += "\(name)\t\(bi)\t\(qf)\t\(q)\t\(stamped ? "yes" : "no")\n" }
            }
        }
    }
    try! manifest.write(toFile: out + "/manifest.tsv", atomically: true, encoding: .utf8)
    print("wrote \(manifest.split(separator: "\n").count / 2) images, \(manifest.split(separator: "\n").count) questions to \(out)")
    exit(0)
}
if hard {
    let variants = ["yes", "yeshard", "no", "lure1", "lure2"]
    for (bi, b) in bases.enumerated() {
        for (fi, (fam, _)) in questions.enumerated() {
            for (vi, v) in variants.enumerated() {
                var rng = Rng(s: b.seed &* 1000 &+ UInt64(fi * 10 + vi))
                var m = Marks()
                // The two other marks: clean, present at random.
                if fam != "SIG" { m.sig = rng.next() < 0.5 ? .normal : .none }
                if fam != "STAMP" { m.stamp = rng.next() < 0.5 ? .normal : .none }
                if fam != "DATE" { m.date = rng.next() < 0.5 ? .normal : .none }
                switch (fam, v) {
                case ("SIG", "yes"): m.sig = .normal
                case ("SIG", "yeshard"): m.sig = .faint
                case ("SIG", "lure1"): m.sigLure = .printedName
                case ("SIG", "lure2"): m.sigLure = .issuerSignature
                case ("STAMP", "yes"): m.stamp = .normal
                case ("STAMP", "yeshard"): m.stamp = bi % 2 == 0 ? .faint : .partial
                case ("STAMP", "lure1"): m.stampLure = .logo
                case ("STAMP", "lure2"): m.stampLure = .printedStatus
                case ("DATE", "yes"): m.date = .normal
                case ("DATE", "yeshard"): m.date = .faint
                case ("DATE", "lure1"): m.dateLure = .orderDate
                case ("DATE", "lure2"): m.dateLure = .placeholder
                default: break   // "no": the asked mark absent, no lure
                }
                let truth = ["SIG": m.sig != .none, "STAMP": m.stamp != .none, "DATE": m.date != .none]
                let tag = "b\(bi)_\(fam)_\(v)_s\(truth["SIG"]! ? 1 : 0)t\(truth["STAMP"]! ? 1 : 0)d\(truth["DATE"]! ? 1 : 0)"
                let rep = draw(b, m)
                let cname = "c_\(tag).png", sname = "s_\(tag).jpg"
                try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out + "/" + cname))
                try! scan(rep, seed: b.seed &* 7919 &+ UInt64(fi * 10 + vi)).write(to: URL(fileURLWithPath: out + "/" + sname))
                for name in [cname, sname] {
                    for (qf, q) in questions {
                        manifest += "\(name)\t\(bi)\t\(qf)\t\(q)\t\(truth[qf]! ? "yes" : "no")\n"
                    }
                }
            }
        }
    }
    try! manifest.write(toFile: out + "/manifest.tsv", atomically: true, encoding: .utf8)
    print("wrote \(manifest.split(separator: "\n").count / 3) images, \(manifest.split(separator: "\n").count) questions to \(out)")
    exit(0)
}
for (bi, b) in bases.enumerated() {
    for v in 0..<8 {
        let signed = v & 1 != 0, stamped = v & 2 != 0, dated = v & 4 != 0
        let name = "b\(bi)_s\(signed ? 1 : 0)t\(stamped ? 1 : 0)d\(dated ? 1 : 0).png"
        render(b, signed: signed, stamped: stamped, dated: dated, to: out + "/" + name)
        let truth = ["SIG": signed, "STAMP": stamped, "DATE": dated]
        for (fam, q) in questions {
            manifest += "\(name)\t\(bi)\t\(fam)\t\(q)\t\(truth[fam]! ? "yes" : "no")\n"
        }
    }
}
render(nil, signed: false, stamped: false, dated: false, to: out + "/blank.png")
for (fam, q) in questions { manifest += "blank.png\t-1\t\(fam)\t\(q)\tno\n" }
try! manifest.write(toFile: out + "/manifest.tsv", atomically: true, encoding: .utf8)
print("wrote \(bases.count * 8 + 1) images, \(manifest.split(separator: "\n").count) questions to \(out)")
