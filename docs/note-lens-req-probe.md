# Requirements probe — does the CV show it, and which line is the evidence?

**STATUS: MEASURED 2026-09-24 on Qwen3.8-9B Q4_K_M. A useful recipe on two
heads already landed — no new head, no engine change.** For a job
requirement checked against a CV, the locate head (L11 h=6) picks the evidence
line 100% EN / 97.2% DE (chance ~7%), paraphrased requirements included; the
score head (L19 h=11) separates met from unmet at AUC 0.954 / 0.928, and holds
0.931 / 0.926 against near-miss sibling skills. Requirements that need a
comparison ("at least C1") do not work.

## 1. The question

The core HR job: take a job ad's requirements and one CV; for each, is it
evidenced, and where? Two unknowns: absence had only been measured on short
field names ("warranty_period"), while real requirements are sentences and
often paraphrased ("container orchestration" for Kubernetes); and a CV that
has a *sibling* skill (Ansible, asked for Terraform) is the natural trap.

## 2. The probe

`REQHEAD=1` in `tests/perf/attn_provenance.cpp`.

* **CVs.** Six fictional people: the two INJHARD CVs (ML engineer EN,
  logistics lead DE) and four written for this probe (ICU nurse, accountant
  EN; backend developer, marketing manager DE).
* **Requirements, 12 per CV, labelled before the run** (72 in total, 36 per
  language):

  | kind | per CV | example |
  |---|---|---|
  | PLAIN — met, in the CV's own words | 3 | "Does the candidate know Python?" |
  | PARA — met, paraphrased | 3 | "Has the candidate worked with container orchestration?" (Kubernetes) |
  | FAR — not met, unrelated | 2 | "Does the candidate hold a forklift licence?" |
  | NEAR — not met, a sibling of something present | 2 | "…Terraform?" (CV: Docker, Kubernetes); "…US GAAP?" (CV: UK GAAP) |
  | STATE — needs a comparison, one met, one not | 2 | "…German at C1 level or better?" (CV: B2) |

  Met items carry evidence substrings of the CV; a line (or a sentence inside
  a line) is evidence if it contains one.
* **Readout.** Each requirement is a question key; one prompt per CV, keys
  shuffled, two shuffles (144 samples). Per head: **presence** = the key's
  mean-row mass summed over the CV; **evidence** = the line/sentence segment
  with the most mass. 8 layers × 16 heads; EN and DE each held out.

## 3. Results (EN / DE)

| head | met vs unmet AUC | met vs NEAR | PARA vs unmet | caught at zero false "missing" | evidence line (paraphrased) | STATE met vs not |
|---|---|---|---|---|---|---|
| **score L19 h=11** | **0.954 / 0.928** | **0.931 / 0.926** | 0.956 / 0.910 | 83.3 / 45.8 | 97.2 / 88.9 (94.4 / 88.9) | 0.694 / 0.750 |
| **locate L11 h=6** | 0.896 / 0.919 | 0.833 / 0.866 | 0.963 / 0.907 | 50.0 / 58.3 | **100 / 97.2 (100 / 100)** | 0.250 / 0.528 |
| absent L19 h=10 | 0.869 / 0.772 | 0.794 / 0.676 | 0.877 / 0.706 | 50.0 / 16.7 | 97.2 / 91.7 | 0.222 / 0.444 |
| choice L11 h=3 | 0.796 / 0.775 | 0.694 / 0.701 | 0.910 / 0.775 | 25.0 / 12.5 | 100 / 97.2 | 0.167 / 0.611 |
| inject L11 h=0 | 0.508 / 0.485 | — | — | 0 / 0 | 83.3 / 69.4 | — |
| L23 h=9 (sweep #1) | 0.990 / 0.956 | 0.981 / 0.924 | 0.991 / 0.926 | 83.3 / 37.5 | 88.9 / 100 | 0.389 / 0.500 |
| L23 h=0 (sweep #2) | 0.980 / 0.957 | 0.968 / 0.958 | 0.965 / 0.921 | 70.8 / 25.0 | 94.4 / 91.7 | 0.694 / 0.944 |

Evidence chance ~7% (6.9 EN / 6.4 DE). Held out: presence selected on EN →
L23 h=9, 0.956 on DE; on DE → L23 h=0, 0.980 on EN. Evidence selected on EN →
choice L11 h=3, 97.2% on DE; on DE → L23 h=9, 88.9% on EN.

## 4. Reading

* **Where: locate.** It lands on the evidence line almost every time,
  including when the requirement is paraphrased — the head matches meaning,
  not words.
* **Whether: score, not absent.** The landed absent head, tuned on short field
  names, is the weaker reader of requirement *sentences* (0.869 / 0.772); the
  score head is the better one on the budget the locate-only server already
  loads, and sibling skills rarely fool it (0.93). The best heads sit at L23
  (0.96–0.99 held out) and cost 4 blocks beyond the 20-block cut.
* **A soft score, not a verdict.** A single cut-off across requirements is
  shaky: at zero false "missing" calls, 83% EN / 46% DE of the truly missing
  requirements are caught. Show it as a graded "evidence found / weak / none".
* **No comparisons.** "At least C1", "more than five years": 0.25–0.75 with
  3 vs 3 items per language — noise. Attention sees *presence*, not *how
  much*. Those rules belong to client logic over the located line.

## 5. What this is and is not

* **A recipe on today's API:** requirements as question keys; `head: "score"`
  for the presence mass, `head: "locate"` for the evidence line (sum per line,
  as for inject).
* **Small and self-written.** Six fictional CVs, 72 requirements, labels by
  me. Real CVs are longer, formatted from PDFs, and mix languages.
* **Near-misses are my picks** of plausible siblings; a recruiter's own
  "close but not it" list would be the real test.

## Reproduce

```
REQHEAD=1 QWEN36_MODEL_PATH=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf ./build-metal/bin/attn-provenance   # ~2 min
```
