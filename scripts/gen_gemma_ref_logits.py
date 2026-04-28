#!/usr/bin/env python3
"""Generate reference logits for the Gemma canary prompt set (PR G1.0).

This script is the single source of truth for the "what should the model
output?" half of the Phase G1 logit-agreement gate. It is deliberately
written to be:

  - Deterministic across runs:  fp32 weights, eager attention, no sampling.
  - Pinned to a transformers version:  see PINNED_TRANSFORMERS below; the
    fixtures should be regenerated only as a deliberate act when this is
    bumped, not as a side effect of an unrelated upgrade.
  - Auditable:  emits a SHA-256 of every output blob plus a manifest JSON
    so PR review can verify the bytes match what the script claims.

Outputs (under tests/fixtures/gemma_ref_logits/<gen>_<model>/):
  - <prompt_hash>.tokens     int32 array of input ids (binary, little-endian)
  - <prompt_hash>.logits     float32 array of last-token logits over vocab
  - manifest.json            { prompt_hash -> {prompt, sha256, vocab_size, ...} }

Usage:
  python scripts/gen_gemma_ref_logits.py \
      --gen g1 --model 2b \
      --prompts tests/fixtures/canary_prompts_gemma.txt \
      --out tests/fixtures/gemma_ref_logits/g1_2b

The C++ side (tests/unit/test_gemma1_forward.cpp once the gate tightens)
loads the .tokens / .logits pair, runs the same prompts through Qwenium,
and asserts top-8 rank preservation + top-1 divergence index >= 128.

This script does NOT run as part of the C++ test suite.  Reference fixtures
are checked in; regenerating them requires deliberate human action.
"""

# --- pin -----------------------------------------------------------------
PINNED_TRANSFORMERS = "4.41.2"  # bump deliberately; documented in the script header
# -------------------------------------------------------------------------

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen",    required=True, choices=["g1", "g2", "g3", "g4"],
                   help="Gemma generation; selects the model family.")
    p.add_argument("--model",  required=True,
                   help="Model size tag (e.g. '2b', '7b', '4b-text').")
    p.add_argument("--hf-id",  required=False,
                   help="Override HF repo id (default derived from gen+model).")
    p.add_argument("--prompts", required=True, type=Path,
                   help="One prompt per non-comment line.")
    p.add_argument("--out",    required=True, type=Path,
                   help="Output directory for fixtures.")
    p.add_argument("--max-prompt-tokens", type=int, default=512,
                   help="Truncate prompt to N tokens before the forward pass.")
    p.add_argument("--check-determinism", action="store_true",
                   help="Run the forward pass twice and assert identical bytes.")
    args = p.parse_args()

    # Defer heavy imports so --help works without torch installed.
    try:
        import torch                                  # noqa: F401
        import transformers                           # noqa: F401
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        print(f"error: missing dependency ({e}). Install transformers=={PINNED_TRANSFORMERS}",
              file=sys.stderr)
        sys.exit(2)

    if transformers.__version__ != PINNED_TRANSFORMERS:
        print(f"warning: transformers=={transformers.__version__} does not match "
              f"PINNED_TRANSFORMERS={PINNED_TRANSFORMERS}. Continuing, but this is a "
              f"deliberate-regen-only situation; do not check in fixtures from a "
              f"non-pinned version.", file=sys.stderr)

    hf_id = args.hf_id or {
        ("g1", "2b"): "google/gemma-2b",
        ("g1", "7b"): "google/gemma-7b",
    }.get((args.gen, args.model))
    if not hf_id:
        print(f"error: no HF id mapped for gen={args.gen} model={args.model}; pass --hf-id",
              file=sys.stderr)
        sys.exit(2)

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"loading {hf_id} (eager attn, fp32) — this may take a while...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    model.eval()

    prompts = [
        line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not prompts:
        print("error: prompt file is empty", file=sys.stderr)
        sys.exit(2)

    manifest = {
        "hf_id": hf_id,
        "transformers_version": transformers.__version__,
        "pinned_transformers": PINNED_TRANSFORMERS,
        "gen": args.gen,
        "model": args.model,
        "prompts": {},
    }

    for prompt in prompts:
        prompt_hash = sha256_bytes(prompt.encode("utf-8"))[:16]
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        ids = ids[:, : args.max_prompt_tokens]

        with torch.inference_mode():
            out1 = model(ids).logits[0, -1].to(torch.float32).cpu().numpy()
            if args.check_determinism:
                out2 = model(ids).logits[0, -1].to(torch.float32).cpu().numpy()
                if not (out1 == out2).all():
                    print(f"error: non-deterministic logits for prompt {prompt!r}",
                          file=sys.stderr)
                    sys.exit(3)

        token_bytes = ids.to(torch.int32).cpu().numpy().tobytes()
        logit_bytes = out1.astype("<f4").tobytes()

        (args.out / f"{prompt_hash}.tokens").write_bytes(token_bytes)
        (args.out / f"{prompt_hash}.logits").write_bytes(logit_bytes)

        manifest["prompts"][prompt_hash] = {
            "prompt": prompt,
            "n_tokens": int(ids.shape[1]),
            "vocab_size": int(out1.shape[0]),
            "tokens_sha256": sha256_bytes(token_bytes),
            "logits_sha256": sha256_bytes(logit_bytes),
        }
        print(f"  {prompt_hash}: n_tokens={ids.shape[1]} vocab={out1.shape[0]}  '{prompt[:40]}...'")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(prompts)} prompts to {args.out}")


if __name__ == "__main__":
    main()
