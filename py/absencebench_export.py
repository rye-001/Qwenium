"""Export AbsenceBench (harveyfin/AbsenceBench, CC-BY-SA 4.0) to JSONL for the
ABSBENCH leg of tests/perf/attn_provenance.cpp.

Reads the three validation parquet files from temp/absencebench/ (downloaded
2026-09-26, never committed) and writes temp/absencebench/absencebench.jsonl:
per domain, every row in a FIXED-SEED shuffled order, skipping only rows whose
original + modified text exceeds MAX_CHARS (they cannot fit the 11K-token
window anyway). The C++ leg applies the exact token-length filter with the
model's own tokenizer and takes the first 30 fitting rows per domain as the
DEV split (threshold choice) and the next 100 as the EVAL split (the score).

Run with the throwaway venv that has pyarrow:
    temp/absencebench/.venv/bin/python py/absencebench_export.py
"""
import json
import random

import pyarrow.parquet as pq

SEED = 20260926
MAX_CHARS = 30000
DOMAINS = ["poetry", "numerical", "github_prs"]
BASE = "temp/absencebench"

with open(f"{BASE}/absencebench.jsonl", "w") as out:
    for d in DOMAINS:
        rows = pq.read_table(f"{BASE}/{d}/validation-00000-of-00001.parquet").to_pylist()
        order = list(range(len(rows)))
        random.Random(SEED).shuffle(order)
        kept = 0
        for i in order:
            r = rows[i]
            if len(r["original_context"]) + len(r["modified_context"]) > MAX_CHARS:
                continue
            out.write(json.dumps({
                "domain": d,
                "id": r["id"],
                "original": r["original_context"],
                "modified": r["modified_context"],
                "omitted_index": r["omitted_index"],
            }) + "\n")
            kept += 1
        print(f"{d}: {kept} of {len(rows)} rows written (<= {MAX_CHARS} chars)")
