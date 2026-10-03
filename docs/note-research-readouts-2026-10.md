# Research shortlist: prefill-only attention / verdict readouts (2026-10-03)

User asked for the top papers we could implement in the lens (text and image),
prefill only. Read 2026-10-03. **Agreed next: #1** (user, 2026-10-03).

## 1. Null-question calibration — AGREED NEXT
- ICR, "Attention in LLMs Yields Efficient Zero-Shot Re-Rankers" (ICLR 2025,
  arXiv 2410.02642): score = attention from the question tokens to each
  document token, summed over layers/heads, MINUS the same with a
  content-free question ("N/A"); tokens below mean − 2 sd dropped. Two
  passes, no training. Removes position and attention-sink bias. Weak when
  the distractor shares words with the query (lexical bias).
- Follow-up "How Calibration Content Shapes Attention-Based Reranking"
  (arXiv 2609.17764, Sept 2026): with long instructions the null pass removes
  real signal; fix = "interpolated null calibration" (control how much of the
  instruction enters the null pass). Demonstrations help and do not touch the
  null pass.
- Why for us: LOCABSENT died of a position confound; the image attention tap
  (CLOSED) failed its left/right swap control. This is a new mechanism aimed
  at exactly that → a legitimate reason to re-test the image attention lens,
  with the swap control (G1) kept. Cost: one short extra prefill of the null
  question on the kept document / kept image.

## 2. AttnTrace (arXiv 2508.03793, 2025)
- Per text unit: average of its top-K attended tokens, not all tokens;
  context subsampling (B≈30 random subsets) against attention dispersion when
  similar texts compete. Teacher-forced passes (like /v1/verify).
- For us: top-K = a parameter on key aggregation (cheap A/B on our gates);
  subsampling = opt-in "deep check" (B passes).

## 3. AbsenceBench (NeurIPS 2025 D&B, arXiv 2506.11440)
- Find what was removed from a document; frontier models ~70% F1; cause:
  attention cannot attend to a gap; placeholders "<missing line>" +42%.
- For us: an outside test for the omission report (/v1/compare) — our one
  unique asset. Hypothesis (not a result): scoring each original unit by
  attention sidesteps the gap problem. Dataset download needs the user's OK.

## Runners-up
- QRHead (EMNLP 2025, 2506.09944): head selection by query→context attention
  on a few examples = what our head hunts already do.
- Lookback Lens (EMNLP 2024, 2407.07071): context-vs-generated attention ratio
  → hallucination flag; needs a small trained classifier.
- "Same Attention, Different Truths" (2608.07302, Aug 2026): logit lens on the
  attended image patches; real objects decode to their names, hallucinated
  ones do not. Training-free. Pairs with #1 for images.
- "Mechanisms of Object Localization in VLMs" (2605.19792): few heads mediate
  localization (LLaVA, InternVL); no Qwen, no swap control reported.
