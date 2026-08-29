# Vision encoder reference fixtures

Reference outputs of llama.cpp's SigLIP encoder for fixed deterministic
inputs. Consumed by `tests/unit/test_vision_encoder.cpp` (P2.3+) as the
gate for our standalone SigLIP implementation in `src/vision/`.

llama.cpp is **reading material + a one-off capture target**. The engine
does not link `libmtmd` or import any llama.cpp code. These binaries are
the artifact; regeneration is a disposable side activity (see procedure
below).

## Files

| File | Content | Shape | dtype |
|---|---|---|---|
| `siglip_gray_896.bin` | Post-projection final encoder output for a gray (filled 0.5) 896×896 image, with `mmproj-BF16.gguf` against the medgemma-1.5-4b-it text model. | `[n_tokens=256, n_embd=2560]` row-major, n_embd-fastest | `float32` |
| `siglip_gray_896.bin.sha256` | sha256 of `siglip_gray_896.bin` | — | text |

`n_embd=2560` is the Gemma 3 4B text embedding length; the encoder's
projection matrix lands at this dim by design (the soft tokens substitute
into the text decoder's residual stream).

## Choice of input — why gray, not a JPEG

Synthetic **gray-0.5** is the input by deliberate choice:
- **Deterministic.** Every byte of the 3×896×896 input is `0.5f`. No JPEG
  decode, no resize, no normalization-convention dispute. The encoder
  differential isolates the encoder; preprocessing is Phase 6's concern
  and gets its own fixtures then.
- **Trivially reproducible in C++.** `std::vector<float>(3*896*896, 0.5f)`
  in our test code mirrors llama.cpp's debug input exactly.
- **Catches structural errors broadly.** A gray input still exercises
  conv2d patch embed, 27 transformer layers (attention + MLP + LN), the
  4×4 average pool, and the linear projection + soft-emb-norm. A
  structural bug in any of those changes the output.

## How to regenerate

This is a one-off side activity, not part of the engine build. Procedure:

```bash
# 1. Create a disposable worktree of llama.cpp at a known commit.
#    Our main llama.cpp checkout is NOT touched.
LLAMA_SRC=~/dev/projects/llama/llama.cpp
LLAMA_WT=/tmp/qinf-llama-capture
LLAMA_COMMIT=0253fb21f595246f54c192fe8332f34173be251b  # was HEAD at capture time
cd "$LLAMA_SRC"
git worktree add "$LLAMA_WT" "$LLAMA_COMMIT" --detach

# 2. Apply the disposable patch (writes emb_data to MTMD_DEBUG_EMBEDDINGS_FILE).
#    Inserted at tools/mtmd/clip.cpp inside the existing MTMD_DEBUG_EMBEDDINGS
#    block, immediately after the Stats log line (~line 4181):
#
#        if (const char * dump_path = std::getenv("MTMD_DEBUG_EMBEDDINGS_FILE")) {
#            FILE * fp = std::fopen(dump_path, "wb");
#            if (fp) {
#                std::fwrite(emb_data.data(), sizeof(float), emb_data.size(), fp);
#                std::fclose(fp);
#            }
#        }
#
# 3. Build the debug tool.
cd "$LLAMA_WT" && mkdir -p build && cd build
cmake .. -DGGML_METAL=ON -DLLAMA_BUILD_SERVER=OFF \
         -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build . --target llama-mtmd-debug -j

# 4. Capture (from the consumer-repo root, with mmproj and text model present).
MTMD_DEBUG_EMBEDDINGS=1 \
MTMD_DEBUG_EMBEDDINGS_FILE="$(pwd)/tests/fixtures/vision/siglip_gray_896.bin" \
  "$LLAMA_WT/build/bin/llama-mtmd-debug" \
    -m  "$(pwd)/medgemma-1.5-4b-it-BF16.gguf" \
    --mmproj "$(pwd)/mmproj-BF16.gguf" \
    -p encode --image gray -n 896

# 5. Update checksum and clean up the worktree.
shasum -a 256 tests/fixtures/vision/siglip_gray_896.bin \
  > tests/fixtures/vision/siglip_gray_896.bin.sha256
git -C "$LLAMA_SRC" worktree remove --force "$LLAMA_WT"
```

## Preprocessing reference fixtures (Task 2)

Lock `src/cli/image_loader.cpp` to llama.cpp mtmd's gemma3 preprocessing
(`mtmd_image_preprocessor_fixed_size`, PAD_CEIL) byte-for-byte. Consumed by
`tests/unit/test_image_loader.cpp::MatchesLlamaCppGemma3Reference`.

| File | Content |
|---|---|
| `preproc_input.png` | Small **157×97** non-square RGB PNG (deterministic gradients + two sharp blocks). PNG is lossless → stb decode is bit-identical across stb versions, isolating resize/pad/normalize. |
| `preproc_input.probes` | Probe points sampled from the exact `[3,896,896]` planar f32 tensor llama-mtmd-cli feeds the SigLIP encoder for that PNG. Text: `dims 3 896 896` then `c y x value` lines (~2.7k points; ~40% in the letterbox pad at `-1.0`). |

Why probes, not the full tensor: the raw `[3,896,896]` f32 is 9.6 MB — too heavy
for git. The probe grid (prime-step coarse grid + explicit pad-boundary rows)
catches the stretch-vs-aspect-pad divergence decisively while staying ~46 KB.

### How to regenerate

```bash
# 1. Add a temporary dump in llama.cpp tools/mtmd/clip.cpp, right before
#    set_input_f32("inp_raw", inp_raw):
#        if (const char* dp = std::getenv("QINF_DUMP_INPRAW")) {
#            FILE* f = fopen(dp, "wb");
#            int nx = imgs.entries[0]->nx, ny = imgs.entries[0]->ny;
#            int hdr[3] = {3, ny, nx};
#            fwrite(hdr, sizeof(int), 3, f);
#            fwrite(inp_raw.data(), sizeof(float), (size_t)3*nx*ny, f);
#            fclose(f);
#        }
# 2. cmake --build build --target llama-mtmd-cli -j
# 3. QINF_DUMP_INPRAW=/tmp/ref_inpraw.bin build/bin/llama-mtmd-cli \
#      -m medgemma-1.5-4b-it-BF16.gguf --mmproj mmproj-BF16.gguf \
#      --image tests/fixtures/vision/preproc_input.png -p . -n 1 --temp 0
# 4. Sample probes from /tmp/ref_inpraw.bin (header: 3 int32 dims, then f32 [C,H,W])
#    into preproc_input.probes. Revert the clip.cpp dump.
```

## Preprocessing reference fixtures — qwen3vl (P5)

Lock `src/cli/image_loader.cpp`'s **DynSmartResize** path, as
`vision/image_preprocess.h::qwen3vl_preprocess` parameterizes it, to llama.cpp
mtmd's `mtmd_image_preprocessor_dyn_size` byte-for-byte. Consumed by
`tests/unit/test_image_loader.cpp::MatchesLlamaCppQwen3VlReference{,Upscaled}`.

| File | Content |
|---|---|
| `preproc_input.qwen3vl.ref` | Reference tensor for `preproc_input.png` (157×97 → **160×96**, 5×3 = 15 tokens). Exercises the round-to-align branch **and** a 4-column letterbox pad. |
| `preproc_input_tiny.png` | Deterministic **21×53** RGB PNG (gradients + two sharp blocks), generated by the stdlib script below. 1113 px is under the projector's 8192 px floor, so it forces the branch the fixture above never reaches. |
| `preproc_input_tiny.qwen3vl.ref` | Reference tensor for it (21×53 → **64×160**, 2×5 = 10 tokens) — the `min_pixels` **upscale** branch. |
| `qwen3vl_preproc.sha256` | sha256 of the two `.ref` files and the generated PNG. |

**Format** (both `.ref` files): 3 `int32` dims `(C, H, W)`, then `C·H·W` `float32`
**planar** — our `Bitmap` layout. `clip_image_f32::buf` is interleaved
`[H][W][C]`; the harness de-interleaves at capture time.

Whole tensors, not probes: a dyn-size canvas for a small image is ~180 KB
(Gemma 3's fixed `[3,896,896]` is 9.6 MB, which is why *that* fixture samples).
Nothing is sampled away here, so the gate is total — and the captured match was
**exact on every one of the 76,800 values**, which is why the test asserts
bit-equality with ε = 0 rather than a tolerance.

The capture confirms, from the mmproj itself rather than by inference:
`patch=16 n_merge=2 min_px=8192 max_px=4194304 mean=[0.5 0.5 0.5]
std=[0.5 0.5 0.5] resize_algo=BILINEAR pad=PAD_CEIL` — i.e. the token budget
8…4096 and the align of 32 that `qwen3vl_preprocess` hardcodes, with
`min_px == 8·32²` and `max_px == 4096·32²`.

### How to regenerate

Same trick as §8.6 of `docs/plan-qwen35-vision-impl.md`: build against the
**vendored** mtmd source without touching it, and never go through
`mtmd_init_from_file` (that needs a text model and hits upstream #20899's width
check). `clip_init` plus the internal preprocessor class is enough — no text
model, no download.

```bash
GS=build-release/_deps/ggml-src        # the pinned ggml/llama.cpp source
SCRATCH=/tmp/qinf-mtmd-capture

cmake -S "$GS" -B "$SCRATCH/build" -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
      -DCMAKE_BUILD_TYPE=Release
cmake --build "$SCRATCH/build" --target mtmd -j 8

# capture.cpp is the harness below
c++ -std=c++17 -O2 -o "$SCRATCH/capture" "$SCRATCH/capture.cpp" \
    -I"$GS/tools/mtmd" -I"$GS/ggml/include" -I"$GS/include" -I"$GS/src" \
    -Ithird_party/stb \
    -L"$SCRATCH/build/bin" -lmtmd -lllama -lggml -lggml-base

DYLD_LIBRARY_PATH="$SCRATCH/build/bin" "$SCRATCH/capture" \
    models/Qwen3.6-mtp-mmproj-BF16.gguf \
    tests/fixtures/vision/preproc_input.png \
    tests/fixtures/vision/preproc_input.qwen3vl.ref
DYLD_LIBRARY_PATH="$SCRATCH/build/bin" "$SCRATCH/capture" \
    models/Qwen3.6-mtp-mmproj-BF16.gguf \
    tests/fixtures/vision/preproc_input_tiny.png \
    tests/fixtures/vision/preproc_input_tiny.qwen3vl.ref

(cd tests/fixtures/vision && shasum -a 256 \
   preproc_input.qwen3vl.ref preproc_input_tiny.qwen3vl.ref \
   preproc_input_tiny.png > qwen3vl_preproc.sha256)
rm -rf "$SCRATCH"
```

`capture.cpp`:

```cpp
#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "mtmd-image.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <cstdio>
#include <cstdint>
#include <vector>

int main(int argc, char** argv) {                 // <mmproj> <image> <out.ref>
    clip_context_params cp{};
    cp.use_gpu = false;
    cp.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    cp.image_min_tokens = -1;   // let the projector's own defaults apply
    cp.image_max_tokens = -1;
    cp.warmup = false;
    clip_init_result r = clip_init(argv[1], cp);

    int w = 0, h = 0, nc = 0;
    unsigned char* data = stbi_load(argv[2], &w, &h, &nc, 3);
    clip_image_u8 img;
    img.set_size({w, h}, false);
    img.cpy_buf(std::vector<uint8_t>(data, data + (size_t) w * h * 3));
    stbi_image_free(data);

    mtmd_image_preprocessor_dyn_size pre(r.ctx_v);
    mtmd_image_preproc_out po = pre.preprocess(img);
    const clip_image_f32& e = po.entries[0];
    const int nx = e.nx(), ny = e.ny();
    std::fprintf(stderr, "%dx%d n_tokens=%d (x=%d y=%d)\n", nx, ny,
                 clip_n_output_tokens(r.ctx_v, &e),
                 clip_n_output_tokens_x(r.ctx_v, &e),
                 clip_n_output_tokens_y(r.ctx_v, &e));

    const std::vector<float>& buf = e.get_ro_buf();   // interleaved [H][W][C]
    const size_t plane = (size_t) nx * ny;
    std::vector<float> planar(3 * plane);             // -> planar [C][H][W]
    for (size_t p = 0; p < plane; ++p)
        for (int c = 0; c < 3; ++c) planar[c * plane + p] = buf[p * 3 + c];

    FILE* f = std::fopen(argv[3], "wb");
    const int32_t hdr[3] = {3, ny, nx};
    std::fwrite(hdr, sizeof(int32_t), 3, f);
    std::fwrite(planar.data(), sizeof(float), planar.size(), f);
    std::fclose(f);
    return 0;
}
```

`preproc_input_tiny.png` is generated (stdlib only — no PIL on this machine):

```python
import struct, zlib, sys
W, H = 21, 53
rows = []
for y in range(H):
    row = bytearray([0])  # filter type 0 (None)
    for x in range(W):
        r, g, b = (x*13 + y*7) % 256, (x*29 + y*3) % 256, (x*5 + y*23) % 256
        if 3 <= x < 8 and 5 <= y < 15:      r = g = b = 255   # sharp white block
        elif 12 <= x < 18 and 30 <= y < 44: r = g = b = 0     # sharp black block
        row += bytes((r, g, b))
    rows.append(bytes(row))

def chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data +
            struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

open(sys.argv[1], "wb").write(
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
    + chunk(b"IEND", b""))
```

## Reproducibility caveat

The reference is bitwise-stable against **this specific llama.cpp commit
+ this specific mmproj-BF16.gguf**. Updating either may change low FP bits
of the captured output (same chunked-recurrence FP class as
`docs/plan-feed-tokens.md` for the text path). When the gate test in
`tests/unit/test_vision_encoder.cpp` starts failing on a low-bit margin
after a llama.cpp pull or mmproj swap, that is a re-capture event — not
a regression in our encoder. Choose token-stable + ε vs. bitwise then,
matching the `feed_tokens` precedent (decision recorded with the test).

## Versioning

| Field | Value |
|---|---|
| llama.cpp commit | `0253fb21f595246f54c192fe8332f34173be251b` |
| llama.cpp libmtmd build tag | `0.0.8988` (informational) |
| mmproj | `mmproj-BF16.gguf` (BF16, Gemma 3 mmproj) |
| text model (loaded but unused for encode) | `medgemma-1.5-4b-it-BF16.gguf` |
| Capture date | 2026-05-28 |
| Backend | Metal (Apple Silicon) |
