# Image-embedding reference fixtures (EmbeddingGemma-2, `gemma4v`)

Phase 1 of `docs/plan-embed-image.md`. llama.cpp is **reading material + a
one-off capture target**: the engine links nothing from it. These files are
the artifact.

- **Reference:** llama.cpp `ceefd16da` (2026-10-16), which contains
  `4fbc76dec5`, "model: support embeddinggemma2 (text+vision+audio)".
  - Built into a **new** dir `build-embed2` of the existing clone; the older
    build dirs were not touched.
  - At this commit, the text vectors still match
    `tests/fixtures/embed/embeddinggemma2_llamacpp.txt` (f498f864f): worst
    cosine **0.9999975** over its 10 prompts. So the reference did not move.
- **Models:** `embeddinggemma-2-BF16.gguf` + `embeddinggemma-2-mmproj-BF16.gguf`
  (projector `gemma4v`).
- **Images: synthetic only,** drawn by `py/gen_embed_images.py` (stdlib +
  zlib, no PIL, no photos). Seven images cover the dyn-size branches:
  - square 224, landscape 640×480, checker 512;
  - aspect-extreme 1200×300 and 300×1200;
  - a page 850×1100;
  - a 40×40 image below the minimum budget (upscaled).
- **Budget:** min 70 / max **280** soft tokens. 280 is the model card's
  default (llama.cpp's own default is 1,120, so the server was started with
  `--image-max-tokens 280`).
- **Framing:** `[BOS 2] <|image> 255999` + soft tokens + `<image|> 258882
  [EOS 1]`. No prefix on images; the card says "Prefixes apply to text
  only". Captions use the query prompt `task: search result | query: `.

## Files

| file | content |
|---|---|
| `*.png`, `captions.txt` | the inputs and one caption each |
| `reference_t280.txt` | per image: preprocessed `C H W`, sha256 **and FNV-1a 64** (`prefnv`, what `Bitmap::content_id` computes — the gate in `test_image_loader`) of the whole f32 planar tensor, 512 seeded probe points (for debugging a mismatch), framed token count, pooled L2 vector (768). Per caption: its text vector. |
| `<name>.t280.soft.bin` | the encoder's soft tokens: `int32 n_tokens, int32 n_embd (512)`, then f32 row-major |
| `<name>.t280.gelu.soft.bin`, `reference_t280_gelu.txt` | the same, captured with **tanh GELU** in the vision FFN (see below) |
| `SHA256SUMS` | checksums of the above |

**Why a hash and not the whole preprocessed tensor:** the tensors are
2–7 MB each. A sha256 of the f32 bytes gates byte identity exactly, and the
probes say where a mismatch is.

**Sanity on the reference itself (no bar):** each caption ranks its own image
first, 7/7, at cosine 0.73–0.81.

## How to regenerate

```bash
LS=~/dev/projects/llama/llama.cpp                 # at ceefd16da
cmake -B $LS/build-embed2 -S $LS -DGGML_METAL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build $LS/build-embed2 --target llama-server -j 10
python3 py/gen_embed_images.py tests/fixtures/embed_image
```

**Preprocessing:** the `clip_init` harness of `tests/fixtures/vision/README.md`
(qwen3vl section), changed in two places:
- `cp.image_min_tokens = 70; cp.image_max_tokens = 280` (taken from argv);
- the output is named `<name>.t280.pre`.

It is linked against `$LS/build-embed2/bin` (`-lmtmd -lllama -lggml
-lggml-base`) and run on the CPU with flash attention off.

**Soft tokens and vectors:** at this commit, `MTMD_DEBUG_EMBEDDINGS=<path>`
dumps the encoder output with no patch needed. Upstream marks that dump as
temporary, so a later commit may not have it.

```bash
MTMD_DEBUG_EMBEDDINGS=/tmp/soft.bin LLAMA_MEDIA_MARKER="<__media__>" \
  $LS/build-embed2/bin/llama-server -m models/embeddinggemma-2-BF16.gguf \
  --mmproj models/embeddinggemma-2-mmproj-BF16.gguf --embeddings --pooling mean \
  -c 8192 -ub 8192 -b 8192 --port 18131 --image-min-tokens 70 --image-max-tokens 280 -np 1 &
python3 py/capture_embed_image_ref.py 18131 280 /tmp/soft.bin <dir of .pre files> ceefd16da
(cd tests/fixtures/embed_image && shasum -a 256 *.png *.soft.bin reference_t280.txt captions.txt > SHA256SUMS)
```

## The tanh-GELU variant (phase 3)

**Why it exists:** the model's `config.json` names the vision FFN activation
`gelu_pytorch_tanh`. The mmproj carries no `clip.use_gelu` key, so llama.cpp
falls back to **GELU-quick** for gemma4v. Both are captured, and the encoder
implements both (`VisionFfnAct`).

**How the tanh reference is made:** run the server above on a scratch copy
of the mmproj with `clip.use_gelu = true` added. llama.cpp is not patched.

```bash
python3 py/gguf_add_bool.py models/embeddinggemma-2-mmproj-BF16.gguf /tmp/mmproj-gelu.gguf clip.use_gelu
# start llama-server as above with --mmproj /tmp/mmproj-gelu.gguf, then:
python3 py/capture_embed_image_ref.py 18131 280 /tmp/soft.bin <dir of .pre files> ceefd16da gelu
```

**What separates the two:**
- the image tokens: 3.3–3.4% rel-L2;
- the pooled vectors: cosine ≥ 0.9998;
- caption → image retrieval: 7/7 for both, the same margins.
