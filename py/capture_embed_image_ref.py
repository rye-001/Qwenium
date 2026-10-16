#!/usr/bin/env python3
"""Capture the image-embedding reference from llama.cpp (docs/plan-embed-image.md §3 phase 1).

Needs a running llama-server built at the reference commit, started with
    MTMD_DEBUG_EMBEDDINGS=<SOFT> LLAMA_MEDIA_MARKER="<__media__>" llama-server
      -m embeddinggemma-2-BF16.gguf --mmproj embeddinggemma-2-mmproj-BF16.gguf
      --embeddings --pooling mean --image-min-tokens 70 --image-max-tokens <BUDGET> -np 1
and the preprocessed tensors from the clip_init capture harness (<name>.t<BUDGET>.pre:
int32 [3, H, W] then f32 planar [C][H][W]).

Writes into tests/fixtures/embed_image/:
    <name>.t<B>.soft.bin   the encoder's soft tokens: int32 n_tokens, int32 n_embd, f32 data
    reference_t<B>.txt     per image: preprocessed dims, sha256 of the whole planar tensor,
                           probe points, the framed token count, the pooled L2 vector;
                           per caption: the text vector (query prompt)

    python3 py/capture_embed_image_ref.py PORT BUDGET SOFT PREDIR COMMIT [TAG]
"""
import base64, hashlib, json, math, os, random, shutil, struct, sys, time, urllib.request

port, budget, soft, predir, commit = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
# Optional variant tag (e.g. "gelu": the mmproj copy with clip.use_gelu=true, so
# llama.cpp runs the tanh GELU the model's config.json names). Files get the tag
# so the GELU-quick reference is never overwritten.
tag = ("." + sys.argv[6]) if len(sys.argv) > 6 else ""
FX = "tests/fixtures/embed_image"
QUERY = "task: search result | query: "


def post(path, body):
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=600)
    return json.loads(r.read())


def fnv(b):
    h = 1469598103934665603
    for x in b:
        h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def l2(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


caps = [l.rstrip("\n").split("\t") for l in open(os.path.join(FX, "captions.txt"))]
out = [f"# EmbeddingGemma-2 IMAGE reference from llama.cpp {commit} (llama-server /embedding, multimodal_data)",
       f"# image budget: min 70 / max {budget} soft tokens; framing [BOS] <|image> soft <image|> [EOS]; no prefix on images",
       "# per image: 'image <name>', 'pre <C> <H> <W> <sha256 of f32 planar>', 'prefnv <FNV-1a 64 of the same bytes, hex>', 'probes <n> c y x value ...',",
       "#            'tokens <n framed>', 'soft <n_tokens> <n_embd> <file>', 'embedding <dim> values (L2)'",
       f"# per caption: 'caption <name>', 'text {QUERY}<caption>', 'embedding <dim> values (L2)'"]
for name, cap in caps:
    pre = open(os.path.join(predir, f"{name}.t{budget}.pre"), "rb").read()
    c, h, w = struct.unpack("<3i", pre[:12])
    vals = struct.unpack(f"<{c * h * w}f", pre[12:])
    rng = random.Random(name)
    pts = sorted({(rng.randrange(c), rng.randrange(h), rng.randrange(w)) for _ in range(512)})
    if os.path.exists(soft):
        os.remove(soft)
    b64 = base64.b64encode(open(os.path.join(FX, name + ".png"), "rb").read()).decode()
    r = post("/embedding", {"content": {"prompt_string": "<__media__>", "multimodal_data": [b64]}})
    for _ in range(50):
        if os.path.exists(soft):
            break
        time.sleep(0.1)
    sb = f"{name}.t{budget}{tag}.soft.bin"
    shutil.copy(soft, os.path.join(FX, sb))
    n_tok, n_embd = struct.unpack("<2i", open(soft, "rb").read(8))
    v = l2(r[0]["embedding"][0])
    out += [f"image {name}", f"pre {c} {h} {w} {hashlib.sha256(pre[12:]).hexdigest()}",
            "prefnv " + format(fnv(pre[12:]), "016x"),
            "probes " + str(len(pts)) + " " + " ".join(f"{p[0]} {p[1]} {p[2]} {vals[(p[0] * h + p[1]) * w + p[2]]:.9g}" for p in pts),
            f"tokens {n_tok + 4}", f"soft {n_tok} {n_embd} {sb}",
            "embedding " + str(len(v)) + " " + " ".join(f"{x:.9g}" for x in v)]
    print(f"{name}: pre {w}x{h}, soft {n_tok}x{n_embd}")
for name, cap in caps:
    v = l2(post("/v1/embeddings", {"input": QUERY + cap})["data"][0]["embedding"])
    out += [f"caption {name}", f"text {QUERY}{cap}", "embedding " + str(len(v)) + " " + " ".join(f"{x:.9g}" for x in v)]
open(os.path.join(FX, f"reference_t{budget}{tag.replace('.', '_')}.txt"), "w").write("\n".join(out) + "\n")
print(f"wrote {FX}/reference_t{budget}{tag.replace('.', '_')}.txt")
