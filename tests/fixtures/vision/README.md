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
