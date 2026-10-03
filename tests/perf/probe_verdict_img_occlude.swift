// VERDICT-IMG occlusion — covers one box of a page image with the paper's colour
// (docs/note-verdict-img-ground.md). Reads a jobs file, one job per line:
//   in_path \t out_path \t x0 \t y0 \t x1 \t y1        (image px, top-left origin)
// The fill is the per-channel median of a 4 px ring just outside the box, so a
// tinted scan gets its own paper colour. Output is PNG: every pixel outside the
// box is the decoded input, unchanged.
//
//   swiftc -O tests/perf/probe_verdict_img_occlude.swift -o .session-results/verdict_img_ground/occlude
//   .session-results/verdict_img_ground/occlude jobs.tsv

import AppKit
import Foundation

func load(_ path: String) -> (CGContext, Int, Int) {
    let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil)!
    let img = CGImageSourceCreateImageAtIndex(src, 0, nil)!
    let w = img.width, h = img.height
    let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                        space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    return (ctx, w, h)
}

let jobs = try! String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
var done = 0
for row in jobs.split(separator: "\n") {
    let f = row.split(separator: "\t").map(String.init)
    guard f.count == 6 else { fatalError("jobs line: expected 6 fields, actual \(f.count): \(row)") }
    let (ctx, w, h) = load(f[0])
    let px = ctx.data!.bindMemory(to: UInt8.self, capacity: w * h * 4)
    // Memory rows run top to bottom, so top-left coordinates index directly.
    let x0 = max(0, Int(Double(f[2])!.rounded())), y0 = max(0, Int(Double(f[3])!.rounded()))
    let x1 = min(w, Int(Double(f[4])!.rounded())), y1 = min(h, Int(Double(f[5])!.rounded()))
    guard x1 > x0, y1 > y0 else { fatalError("box: expected a non-empty box inside \(w)x\(h), actual \(f[2...5])") }
    var ring: [[UInt8]] = [[], [], []]
    for y in max(0, y0 - 4)..<min(h, y1 + 4) {
        for x in max(0, x0 - 4)..<min(w, x1 + 4) where x < x0 || x >= x1 || y < y0 || y >= y1 {
            for c in 0..<3 { ring[c].append(px[(y * w + x) * 4 + c]) }
        }
    }
    let fill = ring.map { $0.isEmpty ? UInt8(255) : $0.sorted()[$0.count / 2] }
    for y in y0..<y1 {
        for x in x0..<x1 {
            for c in 0..<3 { px[(y * w + x) * 4 + c] = fill[c] }
            px[(y * w + x) * 4 + 3] = 255
        }
    }
    let rep = NSBitmapImageRep(cgImage: ctx.makeImage()!)
    try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: f[1]))
    done += 1
}
print("occluded \(done) images")
