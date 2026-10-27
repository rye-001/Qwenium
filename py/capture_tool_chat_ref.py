#!/usr/bin/env python3
"""Capture the tool-request render reference (docs/plan-tool-calling.md gate G1).

For each GGUF, read its tokenizer.chat_template from the header (no model load),
render every case of tests/fixtures/chat_tools/cases.json with Hugging Face's own
renderer (transformers.utils.chat_template_utils.render_jinja_template: jinja2
sandbox, trim_blocks + lstrip_blocks, json.dumps tojson), and write into
tests/fixtures/chat_tools/<sha16>/ (sha16 = sha256 of the template, 16 hex):

    template.jinja   the template, byte for byte (the C++ capture renders this file)
    models.txt       the GGUF file names that carry it
    <case>.hf.txt    the rendering, add_generation_prompt=true; or
    <case>.hf.err    the template's own refusal (raise_exception), one line

bos_token is "<bos>" for gemma4 and "" otherwise, the eos_token the GGUF's id
names. Run with the repo venv (jinja2 3.1.6, transformers 5.3.0):

    myenv/bin/python -I py/capture_tool_chat_ref.py models/Qwen3-0.6B-Q8_0.gguf ...
"""
import hashlib, json, os, struct, sys

import jinja2
import transformers
from transformers.utils.chat_template_utils import render_jinja_template

FX = "tests/fixtures/chat_tools"
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def gguf_header(path):
    """The KV pairs we need: architecture, chat template, tokens, eos/bos ids."""
    want_arrays = {"tokenizer.ggml.tokens"}
    out = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit(f"{path}: expected GGUF magic, actual: other")
        _, _, nkv = struct.unpack("<IQQ", f.read(20))

        def rs():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8")

        def rv(t, keep):
            if t == 8:
                return rs()
            if t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                if not keep:
                    if et == 8:
                        for _ in range(n):
                            (l,) = struct.unpack("<Q", f.read(8))
                            f.seek(l, 1)
                    else:
                        f.seek(SZ[et] * n, 1)
                    return None
                return [rv(et, True) for _ in range(n)]
            return struct.unpack("<" + FMT[t], f.read(SZ[t]))[0]

        for _ in range(nkv):
            k = rs()
            (t,) = struct.unpack("<I", f.read(4))
            out[k] = rv(t, k in want_arrays)
    return out


def main():
    cases = json.load(open(os.path.join(FX, "cases.json"), encoding="utf-8"))
    for path in sys.argv[1:]:
        kv = gguf_header(path)
        tmpl = kv.get("tokenizer.chat_template")
        if tmpl is None:
            sys.exit(f"{path}: 'tokenizer.chat_template': expected present, actual: absent")
        sha16 = hashlib.sha256(tmpl.encode("utf-8")).hexdigest()[:16]
        arch = kv.get("general.architecture")
        toks = kv["tokenizer.ggml.tokens"]
        eos = toks[kv["tokenizer.ggml.eos_token_id"]]
        bos = "<bos>" if arch == "gemma4" else ""
        d = os.path.join(FX, sha16)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "template.jinja"), "w", encoding="utf-8", newline="") as g:
            g.write(tmpl)
        names = set()
        mp = os.path.join(d, "models.txt")
        if os.path.exists(mp):
            names = {l.split()[0] for l in open(mp, encoding="utf-8") if l.strip()}
        names.add(os.path.basename(path))
        with open(mp, "w", encoding="utf-8") as g:
            for n in sorted(names):
                g.write(f"{n} arch={arch} bos={bos!r} eos={eos!r}\n")
        for c in cases["cases"]:
            tools = cases["tool_sets"][c["tools"]] if "tools" in c else None
            base = os.path.join(d, c["name"])
            for stale in (base + ".hf.txt", base + ".hf.err"):
                if os.path.exists(stale):
                    os.remove(stale)
            try:
                rendered, _ = render_jinja_template(
                    conversations=[c["messages"]], tools=tools, chat_template=tmpl,
                    add_generation_prompt=True, bos_token=bos, eos_token=eos, **c.get("kwargs", {}))
                with open(base + ".hf.txt", "w", encoding="utf-8", newline="") as g:
                    g.write(rendered[0])
            except jinja2.exceptions.TemplateError as e:
                with open(base + ".hf.err", "w", encoding="utf-8") as g:
                    g.write(str(e).replace("\n", " ") + "\n")
        print(f"{sha16} {os.path.basename(path)} arch={arch}: {len(cases['cases'])} cases "
              f"(jinja2 {jinja2.__version__}, transformers {transformers.__version__})")


if __name__ == "__main__":
    main()
