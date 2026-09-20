#pragma once
// server_lens.h — Qemmi-Lens extraction: document → audited key-value JSON on
// the attention trust layer (docs/plan-qemmi-lens.md, P2/A2).
//
// Two concerns, split so the numerics are testable without a model:
//
//   1. compute_lens_report(LensRun) — PURE. Given one tapped decode run
//      (prompt/gen tokens, byte maps, and per-step kq_soft rows for the two
//      frozen lens layers), it computes the lens report: per-field citations
//      (L3H13, N3), grounded/ungrounded badges (body_mass, N3b), and the
//      document coverage report (layer-11 max-heads span-peak, COV1). No engine,
//      no ggml — the unit test synthesizes rows. This is the relocated probe
//      math (attn_provenance.cpp run_lens_gen / eval_field / parse_fields).
//
//   2. run_lens_extract(...) — the DRIVER. Assembles the ChatML thinking-off
//      prompt from (document, concepts), runs a FREE decode with the P1
//      attention tap armed on the two lens layers, and hands the captured rows
//      to compute_lens_report. Single-slot (inherits the qwen36 slot-0 KV-gather
//      limit, architecture.md §12).
//
// ── No grammar (Stage 2, 2026-07-17) ─────────────────────────────────────────
// The lens path once constrained this decode with ONE fixed KV grammar. It was
// REFUTED by measurement (docs/note-nogrammar-refutation.md): on the Leg C corpus
// the grammar lost on every axis INCLUDING the guaranteed parse it existed for
// (14/15 vs free's 15/15), and its forced non-empty value was the SOLE cause of
// the absent-concept collapse — and therefore of the two-pass presence gate built
// to work around it. Stage 1 re-validated the trust layer over free output
// (top-3 in-span 61/61; like-for-like in-span mass retention mean −1.0%).
//
// The grammar's parse guarantee is replaced by an explicit contract, not by a
// weaker constraint: TOLERANT on shape (strip a fence, take the outermost JSON
// object) and LOUD on failure (LensUnparseableError ⇒ 422, never a partial
// extraction). See docs/lens-format.md §"The shape contract".
//
// `lens_grammar_gbnf()` survives for ONE reason: the QDOCS_S1 probe runs it as a
// CONTROL ARM against the free path through this same driver, so the comparison
// stays reproducible on shipped code. It is not reachable from /v1/extract.
// NOTE: this says nothing about the engine's GBNF machinery or the server's
// per-request `grammar` field on /v1/completions and /v1/chat/completions —
// that is a separate, shipped, unaffected feature.
//
// The constants are PER MODEL, and what guards them is a CALIBRATION REFUSAL at
// server startup, not a numeric self-check: the loaded model is looked up in
// kLensCalibrations by {architecture, block_count}, and --attention-lens is
// refused fail-loud before the server binds if it has no entry
// (lens_calibration_for / lens_calibration_refusal below, called from main() in
// http_server.cpp). There is NO known-answer sanity check on the citation head —
// an earlier version of this comment claimed one and was wrong. Drift of the
// head WITHIN a calibrated model is therefore unguarded at runtime; it is
// caught, if at all, by the offline probe (tests/perf/attn_provenance.cpp).
//
// Three calibrated models today, all Qwen: Qwen 3.6-35B-A3B (L3H13),
// Qwen 3.8-9B (L27H13) and Qwen 3.8-27B (L19H20). No lens claims for other
// families, and that is a
// MEASURED position, not neglect — Gemma was searched properly and 0 of 768
// candidate heads clear even a 70% bar against the 90% requirement
// (docs/note-lens-gemma4-probe.md, docs/note-lens-gemma-norm-weighted.md). The
// lens is a Qwen-family capability by measurement; the refusal is the mechanism
// that keeps that honest.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class ForwardPassBase;
class Tokenizer;
struct ModelMetadata;
typedef struct ggml_backend_sched* ggml_backend_sched_t;

namespace qinf {

class GrammarVocab;

// ── Lens constants — one model's measured coordinates ────────────────────────
// The member defaults below ARE the Qwen 3.6 calibration (plan §1.6;
// docs/note-qemmi-docs-p0.md), and they are the only place those numbers are
// written down: the qwen35moe rows of kLensCalibrations{} take them by
// default-construction rather than restating them.
struct LensConstants {
    int    citation_head        = 13;     // L3H13 retrieval head (N3)
    int    citation_layer       = 3;      // physical attention layer for citations
    int    coverage_layer       = 11;     // physical attention layer for coverage (COV1)
    double coverage_used_peak   = 0.705;  // span-peak ≥ this ⇒ "consulted" (COV1)
    double ungrounded_body_mass = 0.538;  // mean body_mass ≥ this ⇒ grounded (N3b)
    int    citation_topk        = 8;      // source positions reported per token/field
    // What the report's `model` field says. It travels WITH the coordinates
    // because it describes them: a report computed from this entry was produced
    // by this model, and a hardcoded name would mislabel every other entry's
    // receipts (it did — `run.model` was a literal "Qwen3.6" string until the
    // table landed, which a Qwen 3.8 extraction would have carried).
    const char* model_label     = "Qwen3.6 (attention lens)";

    // Which probe measured the two coordinates, as the receipt reports them.
    // These were the LITERALS "N3" and "COV1" in the report builder until
    // 2026-09-18 — correct for the 35B whose defaults these are, and a false
    // receipt for every row added since: the 9B's head comes from
    // note-lens-qwen38-probe.md, the 27B's from LEGCSEARCH, and
    // their coverage layers from COVSEARCH. Exactly the defect class the
    // `citation_source` comment in server_lens.cpp already describes for the
    // layer/head numbers themselves. A row whose constant is INHERITED rather
    // than measured must say so here, because this string is the only place a
    // caller reading one report can see it.
    const char* citation_probe  = "N3";
    const char* coverage_probe  = "COV1";

    // ── Is a FLASH PREFILL admissible on this model? ─────────────────────────
    // Per-model because the answer IS per-model, and measurably so. A flash
    // prefill is not byte-inert — the attention output feeds the residual
    // stream, so every later layer writes different K/V and the tapped decode
    // reads a differently-written cache. Whether that moves a lens DECISION is
    // what the drift gate scores (tests/perf/attn_provenance.cpp,
    // BANDDRIFT=1 DRIFT_ARM=flash DRIFT_LANG=all), and the two calibrated
    // models answer differently:
    //
    //   Qwen 3.8-9B   dense  — drift 0.00084 vs margin 0.00126, 0/98 decisions
    //                          moved, 15/15 documents token-identical. PASS.
    //   Qwen 3.6-35B  MoE    — drift 0.0242 vs margin 0.0237. FAIL, and the
    //                          sibling chunked-prefill arm changes one
    //                          extraction in fifteen outright.
    //
    // The 35B's failure is not a worse number, it is a different KIND of
    // quantity: expert routing is a top-k argmax, so 2.3% of its selections
    // pick a DIFFERENT EXPERT under the perturbation (MOEROUTE=1, and Gemma
    // 4-26B-A4B reproduces it at 6.4% while two dense models change nothing).
    // A margin argument does not apply to a discrete quantity at all, so no
    // MoE entry should carry `true` on the strength of a small drift number.
    //
    // DEFAULT false, which is what makes this safe: a new entry is refused
    // until someone runs the gate, rather than inheriting a permission
    // measured on a different model. `flash_prefill_provenance` names the run,
    // same discipline as LensCalibration::provenance — a bare boolean is a
    // claim with no receipt.
    bool        flash_prefill_ok = false;
    const char* flash_prefill_provenance = "not scored by the drift gate";

    // ── Where does a KEY token look? (/v1/locate) ────────────────────────────
    // A SECOND, independent pair, and the independence is the finding. Locate
    // read the citation head for one day on the assumption that a retrieval
    // head is a retrieval head; the LOCHEAD sweep (2026-09-18, Leg C messy
    // corpus, 15 documents EN+DE, 75 keys, scored like LEGCSEARCH) put that
    // head at RANK 107 OF 160 for this regime — 41.3% top1, 68.0% top3. The
    // generated→source head is not the key→source head.
    //
    // Chosen 2026-09-18 by the user: **L11 h=5, 82.7% top1 / 89.3% top3.**
    // The decision was depth, not score. Locate truncates after its OWN layer,
    // and layer 11 is exactly `max(citation_layer, coverage_layer)` on this
    // model — so locate costs a `--lens-verify-only` server nothing at all: not
    // one additional block. L23 h=5 measured 100% top3 in BOTH languages and was
    // declined at 24/40 blocks, which would have doubled the verifier slice and
    // ended the "the auditor is 30% of the weights" property (§5 of
    // docs/plan-lens-only-engine.md). L15 h=8 (96.0%, 16/40) is the middle row
    // if 89.3% ever proves too loose.
    //
    // What 89.3% top3 MEANS for the caller, stated plainly because the cut flow
    // depends on it: about one key in nine has its answer outside all three
    // returned spans. A client that cuts a document to those spans deletes that
    // answer, and the later extraction is confidently wrong with a clean
    // receipt. That is why the flow keeps a human looking at the cut
    // (../qemmi-lens/docs/plan-locate-and-cut.md §5 rule 5), and why `top_k` ≥ 3
    // is not a preference.
    //
    // DEFAULT -1 = NOT MEASURED, and /v1/locate refuses such a model outright.
    // The defaults of this struct ARE the Qwen 3.6 calibration, so the values
    // below are the 35B's measured pair; every OTHER row must set -1 explicitly
    // until its own sweep runs. Inheriting one model's layer index is exactly
    // the false-receipt failure this whole table exists to prevent, and layer
    // geometry visibly does not transfer here — the 35B's citation head is at
    // L3 of 40, the 9B's at L27 of 33.
    // ── A LAYER IS NOT A JOB: one layer hosts several heads doing different
    //    things, and `locate_layer`/`locate_head` name exactly ONE of them.
    //
    // This is written here because the field name invites the opposite reading.
    // `locate_layer` sounds like "the layer the lens uses", so the natural next
    // step — reusing this pair for any other question you can phrase as a key —
    // is wrong, and has been measured wrong three times now:
    //
    //   * The CITATION pair read as a locate pair: rank 372 of 384 on the 27B
    //     (LOCHEAD). Near-worst, not merely degraded.
    //   * The LOCATE pair read as a decision head: rank 20 of 128 on 4-way
    //     routing and rank 53 of 128 on an ordinal scale (DECIDEHEAD
    //     2026-09-20, note-lens-qwen38-probe.md). On the ordinal task it scores
    //     45.8% where the best head scores 87.5% — a 41.7-point gap that a
    //     reader assuming "locate_layer is the lens layer" would never suspect.
    //
    // What the same sweep found, and the reason this is a comment rather than a
    // lament: on Qwen3.8-9B the JOBS SEPARATE BY HEAD, NOT BY LAYER. At layer
    // 11 — the twelve blocks a `--lens-locate-only` server already loads —
    // h=6 is the locate head and h=3 answers 4-way routing at 92.5%. A
    // different head on the SAME layer, so a decision costs zero extra depth.
    // The best routing head overall (L19 h=10, 97.5%) buys ~5 points for 8 more
    // blocks, which is a trade, not an upgrade.
    //
    // FOUR JOBS NOW READ FOUR DIFFERENT HEADS on this model, and the spread is
    // the argument: locate L11 h=6, choice L11 h=3, absent L19 h=10, score
    // L19 h=11. Two pairs share a layer and do unrelated work; two heads are
    // ADJACENT on layer 19 and are not interchangeable — h=10 won absence while
    // h=11 wins the ordinal, and on the ordinal job h=10 separates the levels
    // at 1.97 standard deviations against h=11's 3.48. So L19 is where this
    // model finishes semantic matching; it is not a head that does everything.
    // Every one of these was landed only after a sweep, and every sweep so far
    // has found the incumbent pair to be a poor reader of the new job.
    //
    // Each pair is read with its OWN recipe, which is why they cannot be
    // swapped even when they share coordinates: locate wants `max` key
    // aggregation and a peak score, while all three decision pairs want `mean`
    // and a score summed over the body. Landing or moving any of them is a
    // calibration change and therefore a user decision.
    //
    // Quantization DOES move head rankings — the L11 h=6 / L15 h=11 tie for
    // LOCATE on Q8_0 breaks at Q4_K_M — so any pair must name the file type it
    // was measured on, exactly as the rows below record theirs. The decision
    // candidates were then checked on both and held: L11 h=3 scores 92.5% on
    // routing under Q8_0 AND Q4_K_M, and L15 h=11 tops the ordinal task on
    // both. Stability across quants was measured, not assumed.
    int         locate_layer = 11;
    int         locate_head  = 5;
    const char* locate_provenance =
        "LOCHEAD 2026-09-18, Leg C messy corpus 15 docs EN+DE, 75 keys: "
        "L11 h=5 = 82.7% top1 / 89.3% top3; best was L23 h=5 at 100% top3, "
        "declined on depth (24/40 blocks)";

    // ── The CHOICE pair: pick one of N supplied option descriptions ─────────
    //
    // A THIRD job, and a third head — see the "A LAYER IS NOT A JOB" note above
    // locate_layer. Reading choice off the locate pair scores 87.5% where this
    // pair scores 92.5%, and off the citation pair it is worse still.
    //
    // DEFAULT -1 = NOT MEASURED, and the choice head is refused on such a model
    // exactly as an unswept locate pair is. These defaults are the Qwen 3.6
    // values, i.e. unmeasured: DECIDEHEAD has only run on the 9B.
    //
    // WHAT THE RATE IS CONDITIONAL ON, and why the provenance spells it out:
    // unlike locate, this coordinate is only worth its number under a specific
    // recipe — the QUESTION instruction shape, `mean` key aggregation, and the
    // document score summed over the body rather than taken at its peak. Change
    // any of the three and 92.5% is not the rate any more. A bare layer/head
    // pair would be a receipt with its conditions stripped off.
    int         choice_layer = -1;
    int         choice_head  = -1;
    const char* choice_provenance = "not swept by DECIDEHEAD";

    // ── The ABSENT pair: is this key's evidence in the document at all? ────
    //
    // The FOURTH job. Unlike locate and choice it produces no argmax — it is a
    // SEPARATION between "present" and "absent", so it was selected by AUC and
    // its threshold is a product choice, not a constant. `LOCABSENT` is where
    // the threshold work lives; this pair is only the coordinate it reads.
    //
    // THIS ONE IS NOT FREE, and that is the difference from choice. Choice sits
    // on locate's own layer (11), so it costs a truncated server nothing.
    // Absence does not: its best head is at 19, and there is NO stable shallow
    // alternative — the best head at layer 11 is h=8 on Q4_K_M but h=6 on Q8_0,
    // so a "free" absence pair would be a coordinate that changes meaning when
    // the file changes. Paying 8 blocks for a stable one is the trade taken.
    //
    // DEFAULT -1 = NOT MEASURED, refused rather than borrowed, as above.
    int         absent_layer = -1;
    int         absent_head  = -1;
    const char* absent_provenance = "not swept by ABSENTHEAD";

    // ── The SCORE pair: place a document on an ordered scale of N levels ───
    //
    // The FIFTH job, and the one whose number means the least at face value.
    // Read the provenance before quoting anything from it.
    //
    // IT WAS NOT SELECTED BY ACCURACY, and it must not be judged by it. An
    // ordinal has no meaningful argmax: SCOREHEAD found the readout flips
    // between the two ADJACENT levels a document sits between while the
    // ordering stays perfect, so ranking heads by exact-match was ranking them
    // on a coin flip — which is precisely how DECIDEHEAD ended up with a pair
    // that could not be reproduced by selecting twice. This pair was chosen by
    // ORDINAL CONCORDANCE (over every pair of documents on different levels,
    // is the higher one scored higher), tie-broken by how far apart it pushes
    // adjacent levels. Both are scale-free, so neither is flattered by the
    // compressed range described below.
    //
    // THE SCALE IS NOT CALIBRATED, ONLY THE ORDER IS. On the sweep corpus the
    // four true levels come out at 0.62 / 1.02 / 1.49 / 2.13 — monotone, well
    // separated, and visibly NOT on a 0..3 scale. A caller that rounds this to
    // an integer level will read low. The honest output is the fractional
    // score and the per-level masses, with the mapping owned by whoever owns
    // the rubric; an affine correction would be two FITTED numbers, of a kind
    // no other constant in this table is, fitted to one rubric and with no
    // evidence it transfers to a customer's own levels. None is landed here.
    //
    // FREE, and the first pair that is free without a compromise. It shares
    // layer 19 with the absent pair, so a server already serving absence pays
    // nothing for it, and layer 19 is also where the ordinal signal PEAKS —
    // L23, L27 and L31 all score worse. Choice was free by luck and absence
    // bought its depth; this one needed neither.
    //
    // DEFAULT -1 = NOT MEASURED, refused rather than borrowed, as above.
    int         score_layer = -1;
    int         score_head  = -1;
    const char* score_provenance = "not swept by SCOREHEAD";
};

// ── The calibration table — which models the lens may run on ─────────────────
// These are coordinates and thresholds, not a mechanism: "layer 3 head 13" is a
// retrieval head *of one model*, and nothing about it transfers. Run the lens
// on an uncalibrated model and /v1/extract returns a confidently-shaped report
// computed from someone else's coordinates — citations pointing at whatever the
// named layer/head happens to be there, badges off an unvalidated threshold.
// That is a false receipt, so an unlisted model is REFUSED rather than
// best-efforted. The table IS the calibration record: an entry exists because a
// probe measured it, and `provenance` says which one.
//
// ── Why the key is {architecture, block_count} and not architecture alone ────
// Approved 2026-09-05. An architecture string is a FAMILY, not a model:
// `qwen35` hosts Qwen3.5-0.8B (24), Qwen3.5-9B (32), Qwen3.8-9B (33),
// Qwen3.6-27B (64) and Qwen3.8-27B (65) — five different models, five
// different layer counts, one string (src/models/qwen35.h). An arch-keyed
// allowlist would admit all five under coordinates measured on one of them,
// which is the exact false receipt the refusal exists to prevent.
//
// And the key is the RAW GGUF block_count, deliberately, not the decode-stack
// depth (block_count − nextn_predict_layers). Decode depth is the more natural
// quantity, and it is the WRONG key here: it collides Qwen3.5-9B with
// Qwen3.8-9B at 32, and Qwen3.6-27B with Qwen3.8-27B at 64 — in both pairs a
// calibrated model and an uncalibrated one. Raw block_count separates every
// model above, because the trailing MTP head shifts the Qwen 3.8 builds by one.
//
// 2026-09-15 — this got sharper, not milder, when Qwen3.8-27B was calibrated.
// TWO of the five are calibrated now, with DIFFERENT coordinates (L27H13 and
// L19H20), so an arch-keyed allowlist could not even pick a winner to be wrong
// with; and the depth collision at 64 now pairs a genuinely calibrated model
// (Qwen3.8-27B, depth 65-1) with an uncalibrated one (Qwen3.6-27B, raw 64)
// whose own measured answer is 7.1% under those coordinates.
// That separation is an accident of these files, not a law; it holds for every
// model this repo targets, and a future collision must be resolved by adding a
// field to the key, never by widening an entry to cover a model nobody measured.
// `file_type` sentinel: this row does not restrict on quantization, which is
// the behaviour every row had before the field existed.
inline constexpr uint32_t kLensAnyFileType = 0xFFFFFFFFu;

struct LensCalibration {
    const char*   architecture;   // GGUF general.architecture
    uint32_t      block_count;    // GGUF <arch>.block_count, raw (see the key note)
    // ── The third key field, added 2026-09-18 because the collision the note
    // above predicted actually arrived ────────────────────────────────────────
    // NO ROW PINS A FILE TYPE TODAY. The field stays anyway, and this comment
    // is the reason — read it before deleting an "unused" key component.
    //
    // It was added for Ternary-Bonsai-27B (prism-ml), which is `qwen35` with
    // block_count 64 — and so is Qwen3.6-27B, which sits in models/
    // uncalibrated and which
    // LensCalibrationGuard.RefusesUncalibratedModelsOfACalibratedArchitecture
    // pins at nullptr. {arch, block_count} could not separate them, and the
    // note above says how to resolve exactly this: ADD A FIELD to the key,
    // never widen an entry over a model nobody measured. That Bonsai row was
    // reverted on 2026-09-19 when the model was de-scoped
    // (docs/note-lens-bonsai-27b-probe.md keeps the measurements), but the
    // collision it exposed is NOT Bonsai-specific: a plain non-MTP build of
    // Qwen3.8-27B also keys as {qwen35, 64} — 65 = 64 decode + 1 NextN — and
    // would silently inherit the MTP row's calibration without this field.
    //
    // general.file_type is the right field rather than a convenient one,
    // because the calibration measured quant-SENSITIVE: Qwen3.8-27B's own
    // L19H20 scores EN 92 / DE 90.4 at Q3_K_M and EN 90.4 / DE 89.8 on ternary
    // weights, which crosses the 90% bar.
    //
    // kLensAnyFileType means "any quantization", which is what the four
    // pre-existing rows carry so that this field CANNOT refuse a model that
    // worked before it landed. That is deliberately a preserved looseness, not
    // an endorsement: models/ holds Qwen3.8-9B at BOTH Q8_0 and Q4_K_M, both
    // key {qwen35, 33}, and only the Q8_0 was ever measured. Pinning the
    // existing rows would be correct and would also refuse a model that runs
    // today, so it is a separate decision, not a side effect of this one.
    //
    // A row that names a concrete file_type wins over one that does not, so a
    // pinned row and an unrestricted row can share {arch, block_count} without
    // either shadowing the other. NEW rows should pin.
    uint32_t      file_type;      // GGUF general.file_type, or kLensAnyFileType
    const char*   model;          // the exact model the numbers were measured on
    const char*   provenance;     // the probe note that measured them
    LensConstants constants;
};

inline const std::vector<LensCalibration>& lens_calibrations() {
    static const std::vector<LensCalibration> kLensCalibrations = {
        // Qwen 3.6-35B-A3B, the model the lens was built on. Two GGUF builds of
        // one base model: the MTP build carries a trailing NextN draft block
        // (41 = 40 + 1), the plain build does not (40). The decode stack is the
        // same 40 layers in both and the lens never touches the draft head, so
        // both are the SAME calibration — listed twice rather than keyed on a
        // depth that would collide with uncalibrated models elsewhere.
        {"qwen35moe", 41, kLensAnyFileType, "Qwen3.6-35B-A3B (MTP build)",
         "docs/note-qemmi-docs-p0.md (N3, N3b, COV1)", LensConstants{}},
        {"qwen35moe", 40, kLensAnyFileType, "Qwen3.6-35B-A3B (plain build, same 40-layer stack)",
         "docs/note-qemmi-docs-p0.md (N3, N3b, COV1)", LensConstants{}},
        // Qwen 3.8-9B. Its own head is L27H13, not L3H13 — 98% top-3 vs 84% on
        // the same messy corpus, and 0% vs 7% ungrounded false alarm
        // (note-lens-qwen38-probe.md §5.3, confirmed on an independent corpus,
        // not overfit to the selection prompt). Thread-scale citation was then
        // validated on this entry at 4774–6200 tokens with no degradation
        // (note-ss2-thread-alarm.md Gate 0: 89% top-1 / 98% top-3).
        //
        // coverage_used_peak stays 0.705 — and as of 2026-09-15 that is a
        // MEASURED choice, not the inherited one this comment used to describe.
        //
        // It read "the weak arm on both models, 87%/84% used-clear, never
        // searched". Both halves of that are now superseded. The layer WAS
        // searched (COVSEARCH: KEEP L11 on both), and the "weak arm" verdict
        // came from a control group that turned out to be broken — OMISSION1
        // ablated all 133 leg C spans per model and found that 53% (9B) / 38%
        // (27B) of the spans labelled "filler" are CAUSALLY USED. Scored
        // against causal labels instead, coverage separates used from unused at
        // AUC 0.912 / 0.955, which is a strong signal, not a weak one.
        //
        // Why 0.705 and not the accuracy-optimal ~0.30: `skipped[]` is a
        // RECALL-first screen ("anything ignored is in this list",
        // docs/lens-format.md), and accuracy weights a missed omission the same
        // as a spurious entry. Dropping to 0.31 takes the 9B from 97% recall to
        // 83%. 0.705 is the right operating point for the claim we make.
        //
        // COVCAUSAL found L7 (9B) and L15 (27B) hold recall EXACTLY equal to
        // L11 and buy ~7 points of precision — a shorter list, not a better
        // claim, and two different layers, so there is no shared default to
        // move to. Not moved. See docs/plan-lens-only-engine.md §4.
        {"qwen35", 33, kLensAnyFileType, "Qwen3.8-9B",
         "docs/note-lens-qwen38-probe.md §5.3; docs/note-ss2-thread-alarm.md",
         LensConstants{/*citation_head*/ 13, /*citation_layer*/ 27, /*coverage_layer*/ 11,
                       /*coverage_used_peak*/ 0.705, /*ungrounded_body_mass*/ 0.538,
                       /*citation_topk*/ 8, /*model_label*/ "Qwen3.8-9B (attention lens)",
                       /*citation_probe*/ "note-lens-qwen38-probe.md \u00a75.3",
                       /*coverage_probe*/ "COVSEARCH",
                       /*flash_prefill_ok*/ true,
                       /*flash_prefill_provenance*/
                       "drift gate 2026-09-13 (BANDDRIFT DRIFT_ARM=flash DRIFT_LANG=all): "
                       "15/15 token-identical, 0/98 decisions crossed, max |dpeak| 0.000835 "
                       "vs line-level margin 0.001259",
                       // LOCHEAD 2026-09-19. Ranks 1 and 2 tie exactly (L11 h=6
                       // and L15 h=11, both 88.0/96.0, both EN 95.0 / DE 97.1);
                       // L11 wins on DEPTH, the same tiebreak the 35B row
                       // documents. It is free here: max(citation 27, coverage
                       // 11, locate 11) + 1 = 28, unchanged.
                       //
                       // Locate and citation run in OPPOSITE directions on this
                       // model — locate peaks at L11 (96.0%) and decays to 84.0%
                       // at L31; citation is flat until a sharp onset at L27.
                       // Reading locate off the citation layer scores 49.3%,
                       // rank 107 of 128. They are not interchangeable.
                       /*locate_layer*/ 11, /*locate_head*/ 6,
                       /*locate_provenance*/
                       // TWO RATES, because this row is kLensAnyFileType and
                       // therefore serves BOTH files. Quoting only the Q8_0
                       // number made every Q4_K_M report overstate itself by
                       // 2.7 points of top-3 — a receipt claiming a rate that
                       // was never measured on the file that produced it.
                       // Re-stated 2026-09-20 when Q4_K_M became the default
                       // deployment; `config.weights` already tells a reader
                       // WHICH file, this tells them what it scores.
                       "LOCHEAD 2026-09-19, Leg C messy corpus 15 docs EN+DE, 75 keys. "
                       "Q8_0: L11 h=6 = 88.0% top1 / 96.0% top3 (EN 95.0 / DE 97.1), "
                       "rank 1 of 128, tied with L15 h=11 and taken on depth. "
                       "Q4_K_M: the SAME pair = 84.0% top1 / 93.3% top3 "
                       "(EN 92.5 / DE 94.3), rank 2 of 128 — the tie breaks and L15 h=11 "
                       "leads there, so this coordinate is no longer optimal on that "
                       "file, though both halves still clear the 90% top-3 bar. "
                       "Read the rate for the file in config.weights. TOP-1 IS 88.0% "
                       "(Q8_0) or 84.0% (Q4_K_M) — a caller that acts on a single span "
                       "is acting on that rate, not on top-3",
                       // DECIDEHEAD 2026-09-20. THE SAME LAYER, A DIFFERENT
                       // HEAD: h=3 routes where h=6 locates, so choice costs
                       // this model ZERO extra depth — a --lens-locate-only
                       // server already loads the 12 blocks it needs.
                       //
                       // Not the best head available (L19 h=10 reaches 97.5% on
                       // Q8_0), declined on depth: +5 points for 8 more blocks,
                       // the same tiebreak the locate pair above records.
                       //
                       // Chosen for STABILITY as much as rate. L11 h=3 measures
                       // 92.5% with an identical EN 95.0 / DE 90.0 split on
                       // BOTH Q8_0 and Q4_K_M, where the pooled winner moves
                       // between quants (L19 h=10 on Q8_0, L15 h=9 on Q4_K_M).
                       // It is the most reproducible number in either sweep,
                       // which matters more than 5 points for a coordinate that
                       // has to survive a file swap.
                       /*choice_layer*/ 11, /*choice_head*/ 3,
                       /*choice_provenance*/
                       "DECIDEHEAD 2026-09-20, 40 docs EN+DE, 4-way routing, chance 25%: "
                       "L11 h=3 = 92.5% (EN 95.0 / DE 90.0), rank 7 of 128, IDENTICAL on "
                       "Q8_0 and Q4_K_M. Measured ONLY under: question instruction shape, "
                       "mean key aggregation, document score summed over the body (not "
                       "peak) — change any of the three and this rate does not apply. "
                       "Corpus is synthetic and self-authored; held-out cross-language "
                       "selection on the same sweep ran 85-95% and is NOT symmetric "
                       "(EN selects L15 h=9 and reads 85.0% on DE; DE selects L19 h=2 and "
                       "reads 95.0% on EN) \u2014 this pair was landed for cross-QUANT "
                       "stability and for being free, never for held-out symmetry. Rank "
                       "restated 2026-09-20 when DECIDEHEAD was corrected to select under "
                       "the question shape only and to tie-break a saturated accuracy on "
                       "margin; the RATE and the EN/DE split are unchanged, only the "
                       "ordering around it moved",
                       // ABSENTHEAD 2026-09-20. RANK 1 OF 128 ON BOTH QUANTS
                       // (0.9948 on Q4_K_M, 0.9953 on Q8_0), and the held-out
                       // selection picks the SAME configuration on both files.
                       // The incumbent locate pair reads 0.9612 / 0.9713 here.
                       //
                       // NOT free: L19 is 20 of 33 blocks against locate's 12,
                       // and unlike choice there is no stable shallow
                       // alternative — the best head at L11 is h=8 on Q4_K_M
                       // and h=6 on Q8_0. A cheaper pair would change meaning
                       // with the file, so depth was paid for stability.
                       /*absent_layer*/ 19, /*absent_head*/ 10,
                       /*absent_provenance*/
                       "ABSENTHEAD 2026-09-20, Leg C corpus, 75 present vs 90 absent over "
                       "6 verified-absent concepts, position-balanced: L19 h=10 = AUC "
                       "0.9948 (Q4_K_M) / 0.9953 (Q8_0), rank 1 of 128 on BOTH. Held out "
                       "(select on one language, score on the other) 0.9905 on both files. "
                       "OPERATING POINT at zero false accusations: 89.6% of absences caught "
                       "on Q4_K_M, 79.2% on Q8_0; at a 10% false-alarm target both reach "
                       "100%. Measured ONLY under: question instruction shape, mean key "
                       "aggregation, document score summed over the body. AUC IS A "
                       "SEPARATION, NOT A RATE \u2014 the threshold is a product choice, see "
                       "LOCABSENT. Corpus synthetic and self-authored",
                       // SCOREHEAD 2026-09-20. RANK 1 OF 128 ON BOTH QUANTS
                       // and UNIQUE at the top on both — concordance 1.0000 is
                       // reached by exactly one head of 128, not shared.
                       //
                       // ADJACENT TO THE ABSENT HEAD AND NOT THE SAME HEAD:
                       // h=10 (absence) reads this job at concordance 0.9861
                       // and separates adjacent levels at 1.97 SD against
                       // h=11's 3.48. The two neighbours order almost equally
                       // well and pull the levels apart very differently, which
                       // is the whole reason separation and not concordance
                       // picked this pair.
                       //
                       // FREE: layer 19 is already loaded for absence, and the
                       // ordinal signal PEAKS there (L23 0.9722, L27 0.9803,
                       // L31 0.9444), so there is no depth trade to decline.
                       //
                       // Q4_K_M is the STRONGER file here — 3.48 separation
                       // against Q8_0's 2.70 — the third signal running in
                       // which the higher quant is the wrong direction.
                       /*score_layer*/ 19, /*score_head*/ 11,
                       /*score_provenance*/
                       "SCOREHEAD 2026-09-20, 48 docs EN+DE, 4 ordered levels: L19 h=11 = "
                       "ordinal concordance 1.0000 on Q4_K_M AND Q8_0, rank 1 of 128 on "
                       "both and the ONLY head at 1.0000 on either. Adjacent-level "
                       "separation 3.48 SD (Q4_K_M) / 2.70 (Q8_0). Held out on THREE axes, "
                       "all selecting this same pair: language (EN and DE each pick it and "
                       "each score 66.7% exact / 87.5% within-1 on the other), corpus "
                       "origin (the 24 documents added for this sweep pick it too, and are "
                       "HARDER than the 24 they extend), and bilingual halves (swaps to the "
                       "neighbouring h=10, which still reads 0.9904 on the half it did not "
                       "see). THE EXACT-MATCH RATE IS NOT THE PRODUCT NUMBER: 66.7% "
                       "(Q4_K_M) / 58.3% (Q8_0) measures an argmax this pair was "
                       "deliberately NOT selected by; within-1 is 89.6% / 83.3% and the "
                       "ORDERING is perfect. THE SCALE IS UNCALIBRATED \u2014 four true "
                       "levels read 0.62 / 1.02 / 1.49 / 2.13, so rounding to an integer "
                       "reads low; emit the fraction and the masses. Measured ONLY under: "
                       "question instruction shape, mean key aggregation, document score "
                       "summed over the body. Supersedes DECIDEHEAD's 87.5% for L15 h=11, "
                       "which was selection inflation on 24 documents and reads 64.6% here. "
                       "Corpus synthetic and self-authored"}},
        // Qwen 3.8-27B. Its own head is L19H20 — and the method that found the
        // 9B's head would have picked the WRONG one here: the N3 leg selects on
        // three synthetic prompts, chose L11H22, and that head then scored 84.6%
        // on the messy corpus, rank 14 of 384. L19H20 was selected on the corpus
        // that judges it (LEGCSEARCH) and confirmed on the N3 prompts it had not
        // seen. See docs/note-lens-qwen38-27b-probe.md §3.
        //
        // The first model to clear EVERY arm of leg C: citation 91% top3
        // (EN 92 / DE 90), coverage 97% used-clear (EN 98 / DE 97), 0/75
        // ungrounded false alarms. The 9B, which ships, has never passed it —
        // its coverage arm is 87% and 83% on German.
        //
        // Two caveats that belong next to the numbers, not only in the note.
        // DE citation is 90.4% against a 90% bar: four tokens the other way and
        // this arm fails. And every measurement here is Q3_K_M, where the 9B's
        // are Q8_0 — greedy agreement slipped to 27/28 and 28/29, which is the
        // quantization talking. `config.weights` on the report is what records
        // which one a given receipt actually ran under.
        {"qwen35", 65, kLensAnyFileType, "Qwen3.8-27B",
         "docs/note-lens-qwen38-27b-probe.md (LEGCSEARCH §3, held-out §4, COVSEARCH §5)",
         LensConstants{/*citation_head*/ 20, /*citation_layer*/ 19, /*coverage_layer*/ 11,
                       /*coverage_used_peak*/ 0.705, /*ungrounded_body_mass*/ 0.538,
                       /*citation_topk*/ 8, /*model_label*/ "Qwen3.8-27B (attention lens)",
                       /*citation_probe*/ "LEGCSEARCH",
                       /*coverage_probe*/ "COVSEARCH",
                       /*flash_prefill_ok*/ true,
                       /*flash_prefill_provenance*/
                       "drift gate 2026-09-15 (BANDDRIFT DRIFT_ARM=flash DRIFT_LANG=all): "
                       "15/15 token-identical, 0/98 decisions crossed, max |dpeak| 0.000544 "
                       "vs line-level margin 0.000991 — 1.8x, and the binding language here "
                       "is ENGLISH (EN 0.000991 vs DE 0.020000), the reverse of the 9B",
                       // LOCHEAD 2026-09-19. THIS IS THE FIRST ROW WHERE LOCATE
                       // MOVES THE CUT: max(citation 19, coverage 11, locate 27)
                       // + 1 = 28 of 65, where citation+coverage alone gave 20.
                       // That cost was accepted deliberately, because the free
                       // zone is empty — the best candidate at or below L19 is
                       // L15 h=17 at 90.7% pooled, and it FAILS German at 88.6%.
                       // An EN-only reading (92.5%) would have shipped it free.
                       //
                       // L27 h=10 is the SHALLOWEST candidate clearing 90% top3
                       // on both halves, chosen over the best one: L35 h=16
                       // scores a perfect 100.0/100.0/100.0 but cuts 36 of 65,
                       // +16 blocks against L27's +8. Six other heads also hit
                       // 100/100/100 (L39 h=7, L43 h=22, L47 h=17/h=2, L39 h=12,
                       // L47 h=1), so depth buys a plateau here, not a peak.
                       /*locate_layer*/ 27, /*locate_head*/ 10,
                       /*locate_provenance*/
                       "LOCHEAD 2026-09-19, Leg C messy corpus 15 docs EN+DE, 75 keys: "
                       "L27 h=10 = 81.3% top1 / 94.7% top3 (EN 97.5 / DE 91.4), rank 40 "
                       "of 384, the shallowest head clearing 90% top3 on BOTH halves. "
                       "Best available was L35 h=16 at 100% top3, declined on depth "
                       "(36/65 blocks vs 28/65). TOP-1 IS 81.3% \u2014 a caller that acts "
                       "on a single span is acting on that rate, not on 94.7%. Measured "
                       "at Q3_K_M, where the 9B's pair is Q8_0",
                       // DECIDEHEAD 2026-09-20 RAN HERE AND IS NOT LANDED. The
                       // pair is left unmeasured-by-declaration rather than
                       // filled in, because every candidate cheap enough to be
                       // worth taking FAILED its held-out check and the one
                       // that passed costs 8 blocks more than absence.
                       //
                       // Read the provenance for the numbers before proposing
                       // any of them again — in particular L31 h=10, whose
                       // 95.0% is a POOLED rate that German does not reproduce.
                       /*choice_layer*/ -1, /*choice_head*/ -1,
                       /*choice_provenance*/
                       "SWEPT AND DECLINED, not unmeasured. DECIDEHEAD 2026-09-20 on "
                       "Q3_K_M, 40 docs EN+DE, 4-way routing, chance 25%, question shape "
                       "and mean key aggregation. The signal is strong: L47 h=13 = 100.0% "
                       "pooled, unique of 384, and BOTH held-out halves select it "
                       "(90.0% EN->DE, 100.0% DE->EN). It costs 48 of 65 blocks. The "
                       "cheapest STABLE head is L39 h=7 at 40 blocks: 97.5% pooled, both "
                       "halves select it, and its worst held-out direction (95.0%) beats "
                       "L47's (90.0%). NOTHING CHEAPER SURVIVES A HELD-OUT CHECK within "
                       "its own depth budget: at 32 blocks L31 h=10 reads 95.0% pooled but "
                       "German selects L15 h=17 and scores 75.0% on English; at 28 blocks "
                       "(locate's own layer, i.e. free) L27 h=6 reads 92.5% pooled and "
                       "fails the same way. So choice is NOT free on this model, unlike "
                       "the 9B where it shares locate's layer \u2014 landing it means "
                       "paying 40 blocks, which is a product decision nobody has made. "
                       "Q3_K_M ONLY: no second 27B file exists, so the cross-quant "
                       "agreement that qualified every 9B pair was not available here",
                       // ABSENTHEAD 2026-09-20, corrected leg. Rank 1 of 384 and
                       // UNIQUE at the top; both held-out halves select this
                       // same head AND the same variant, each with exactly one
                       // candidate at its top.
                       //
                       // Costs 4 blocks over locate (28 -> 32) and is worth
                       // them: the free head at L27 h=22 catches 79-88% of
                       // absences where this one catches 93-94%, at the same
                       // near-zero false-accusation rate.
                       /*absent_layer*/ 31, /*absent_head*/ 23,
                       /*absent_provenance*/
                       "ABSENTHEAD 2026-09-20 on Q3_K_M, Leg C corpus, present vs absent "
                       "over 6 verified-absent concepts, position-balanced: L31 h=23 = AUC "
                       "0.9956 (EN 0.9953 / DE 0.9966), rank 1 of 384 and the ONLY head at "
                       "that AUC. d-prime 4.17. Held out (select on one language, score on "
                       "the other) picks THE SAME head and THE SAME variant both ways, one "
                       "candidate at the top of each half: 0.9966 and 0.9953. OPERATING "
                       "POINT at a 2% false-alarm target: 92.9% / 93.8% of absences caught "
                       "at 0.0% / 2.5% actual false accusations; the free head at L27 h=22 "
                       "reaches only 79.2-88.1% there. Measured ONLY under: question "
                       "instruction shape, mean key aggregation, document score summed "
                       "over the body. AUC IS A SEPARATION, NOT A RATE \u2014 the "
                       "threshold is a product choice, see LOCABSENT. The incumbent locate "
                       "pair reads 0.9600 here, rank 41 of 384. Q3_K_M ONLY: no second 27B "
                       "file exists, so cross-quant agreement was not measurable",
                       // SCOREHEAD 2026-09-20: swept, NO-GO. Not a weak result
                       // that might be worth taking — an unreproducible one.
                       /*score_layer*/ -1, /*score_head*/ -1,
                       /*score_provenance*/
                       "SWEPT AND REFUSED. SCOREHEAD 2026-09-20 on Q3_K_M, 48 docs EN+DE, "
                       "4 ordered levels. The pooled winner L43 h=13 reaches ordinal "
                       "concordance 0.9965, which looks landable and is not: ZERO of the "
                       "three held-out axes agree on a head. Language picks L43 h=15 vs "
                       "L39 h=12; bilingual halves pick L43 h=6 vs L31 h=19 and one "
                       "direction COLLAPSES to 0.7548; corpus origin picks L35 h=2 vs L43 "
                       "h=13. Top-10 overlap falls to 2 of 10. The level profile is also "
                       "squashed (0.97 / 1.13 / 1.27 / 1.62 against the 9B's 0.62 / 1.02 / "
                       "1.49 / 2.13). The 9B's bar \u2014 both held-out axes agree \u2014 "
                       "rejects this, so no pair is landed. UNTESTED HYPOTHESIS: the only "
                       "27B file is Q3_K_M and the 9B showed ordinal separation is "
                       "quant-sensitive in the same direction (Q4_K_M 3.48 vs Q8_0 2.70), "
                       "so a gentler quant may carry the signal. Not a claim \u2014 there "
                       "is no second file to test it on"}},
    };
    return kLensCalibrations;
}

// The calibration for one loaded model, or nullptr if it has none.
inline const LensCalibration* lens_calibration_for(const std::string& arch,
                                                  uint32_t block_count,
                                                  uint32_t file_type) {
    // Two passes, not one, and the order is the contract: a row that pins a
    // quantization beats a row that accepts any. One pass with `||` would let
    // table ORDER decide which of the two answers a caller gets.
    for (const LensCalibration& c : lens_calibrations())
        if (arch == c.architecture && block_count == c.block_count &&
            c.file_type == file_type) return &c;
    for (const LensCalibration& c : lens_calibrations())
        if (arch == c.architecture && block_count == c.block_count &&
            c.file_type == kLensAnyFileType) return &c;
    return nullptr;
}

// The refusal text. Fail-loud contract order: parameter, expected, actual.
// The calibrated set rendered as one string, "arch/block (model)" separated by
// commas. ONE builder, because the startup banner and the refusal below are two
// views of the same table: an operator who reads one and later hits the other
// must not be given two different answers to "which models have a lens?".
inline std::string lens_calibration_list() {
    std::string list;
    for (const LensCalibration& c : lens_calibrations()) {
        if (!list.empty()) list += ", ";
        list += std::string(c.architecture) + "/" + std::to_string(c.block_count) + "/" +
                (c.file_type == kLensAnyFileType ? std::string("any-quant")
                                                 : std::string("ftype ") + std::to_string(c.file_type)) +
                " (" + c.model + ")";
    }
    return list;
}

inline std::string lens_calibration_refusal(const std::string& arch, uint32_t block_count,
                                            uint32_t file_type) {
    return "--attention-lens: expected a model with a calibrated lens entry, one of {" +
           lens_calibration_list() + "}, actual architecture '" + arch + "' with block_count " +
           std::to_string(block_count) + " and file_type " + std::to_string(file_type) +
           " — the lens constants are coordinates measured on one model and do not "
           "transfer to another model of the same architecture";
}

// ── Pure-computation input: one tapped decode run ────────────────────────────
// One decode step's tapped rows, flat [n_head * n_kv] row-major [head][kv].
// citation_row is the citation_layer's kq_soft row; coverage_row the
// coverage_layer's. steps[t] is the row computed with gen token t as the query
// (so it is the provenance of gen token t+1 — N3's off-by-one).
struct LensStep {
    int                n_kv = 0;
    std::vector<float> citation_row;
    std::vector<float> coverage_row;
};

struct LensRun {
    std::vector<std::string> prompt_text;   // decoded text per prompt token (len P)
    std::vector<size_t>      prompt_cum;    // cum bytes over prompt (len P+1)
    std::vector<std::string> gen_tok_text;  // decoded text per gen token (len G)
    std::vector<size_t>      gen_cum;       // cum bytes over gen (len G+1)
    std::string document;                   // the user's raw document (values re-found here)
    std::string gen_text;                   // concat of gen_tok_text = the emitted JSON
    // The document's token range within the ChatML-wrapped prompt: tokens
    // [doc_lo, doc_hi) are the document; [0,doc_lo) is the chat header and
    // [doc_hi,P) the instruction + assistant tag. All lens signals (citations,
    // coverage, body_mass) are restricted to the document range — that is where
    // N3b's 0.538 threshold was calibrated. doc_byte_offset is the document's
    // start byte within the prompt (to translate citations to document-relative).
    int    doc_lo = 0, doc_hi = 0;
    size_t doc_byte_offset = 0;
    int  n_head = 0;
    std::vector<LensStep> steps;            // one per gen token; steps.size() == G
    // Byte offset, within `document`, where each request message starts (message
    // i spans [message_offsets[i], message_offsets[i+1])). Empty ⇒ the caller
    // sent a plain `document` and citations carry message -1.
    std::vector<size_t> message_offsets;
    std::string model;                      // passthrough for the report header
    // false ⇒ prompt exceeded the 4 K CALIBRATION floor (a disclosure on the
    // report, not an error). Unrelated to the 10 K workload envelope.
    bool validated_envelope = true;
    // Where `gen_text` came from: false ⇒ this model emitted it (/v1/extract),
    // true ⇒ the caller supplied it and it was teacher-forced (/v1/verify).
    //
    // This is the SINGLE source of the origin fact. compute_lens_report derives
    // LensReport::extraction_origin from it, and the shape-contract refusal
    // words itself from it — a refusal that blames "the model" for text the
    // caller handed in sends the reader to inspect the wrong thing entirely.
    // Defaults false so every existing producer (including the MLX leg's
    // lens_from_run, which builds a LensRun by hand) keeps its exact meaning.
    bool extraction_supplied = false;
};

// ── Lens report (the interchange format; P3 versions/documents it) ───────────
// An attended prompt position. `message` is the index of the request message
// this byte range falls in, or -1 when the request sent a plain `document`
// (no boundaries to resolve against). It is a LOCATION, not a verdict — see
// the CF1 non-claim in lens-format.md: the top citation is not "the winner".
struct LensCitation { int pos; double mass; size_t byte_lo, byte_hi; int message = -1; };
struct LensPromptToken { int pos; std::string text; std::string region; };  // region: "body"|"instr"
struct LensCoverageSpan { int lo, hi; double peak; std::string text; size_t byte_lo, byte_hi; };

struct LensField {
    std::string key, value;
    // A5.3 machine-readable trust tier of the VALUE: "distinctive" (citations
    // claimed) | "short_numeric" (weak, coverage-backstopped). "" for absent.
    std::string tier;
    int    gen_lo = -1, gen_hi = -1;      // gen-token span of the value
    bool   found_in_document = false;     // value appears verbatim in the body
    size_t value_byte_lo = 0, value_byte_hi = 0;  // its first byte span (valid iff found)
    bool   grounded = true;               // body_mass ≥ threshold (badge)
    double body_mass = 0.0;               // mean citation-head mass on body positions
    std::vector<LensCitation> citations;  // top-k document source positions (body only)
    // Distinct request-message indices this field's citations landed in, in
    // descending citation mass. Empty when the request sent a plain `document`.
    // Usually one element ("read from message 23"); more than one means the
    // model looked at this key in several messages, which is exactly the
    // coexisting-conflict presentation the format already requires — the lens
    // does NOT say which of them is current (turn order does not identify
    // supersession, docs/note-ss3-matched-pairs.md §3).
    std::vector<int> citation_messages;
    // ── Absent by omission (Stage 2; was the two-pass presence gate, A5.1) ───
    // false ⇒ ABSENT: the model simply did not emit this hinted concept (or
    // emitted it empty). Serializes value:null, badge:"absent". This is now a
    // MECHANICAL read of the parsed output, not a verdict: with no grammar the
    // model declines natively and correctly (30/30 on the Leg C corpus, against
    // the grammar's 10/30 — docs/note-nogrammar-refutation.md), so there is
    // nothing to gate. The N+1 presence prefills are gone with it.
    bool present = true;
    // Which occurrence of this key in the model's output this field is (0-based,
    // emission order). A document whose real structure REPEATS — an invoice with
    // three line items against a flat schema — makes the model emit the key
    // several times. Every occurrence is now reported with its OWN value and its
    // OWN citations; before 2026-09-05 the first was reported three times over
    // and the rest were discarded silently (measured: LensDuplicateKeys).
    // Occurrence order is EMISSION order, which is positional in the document —
    // it is NOT a claim that occurrence 0 pairs with occurrence 0 of another key.
    // Grouping repeated keys into records is a leaf-path design and is not this.
    int occurrence = 0;
};

// One candidate span for one key (docs/plan-candidate-set.md, pass 2). `value`
// is always a byte-exact slice of `document` — non-verbatim spans are dropped
// by the producer before they reach here, never stored and flagged.
struct LensCandidate {
    std::string value;
    size_t      byte_lo = 0, byte_hi = 0;
};

struct LensReport {
    // v2 (Stage 2, 2026-07-17): `presence_grounded` is GONE from every field.
    // v1 folded its additions in place because they were purely additive — a v0
    // importer that hard-refused unknown fields stayed safe. This one REMOVES a
    // column from a shipped shape, so it cannot ride in place: an importer reading
    // presence_grounded would silently get nothing. Subtractive ⇒ version bump.
    // v3 (2026-09-05): `fields` may carry MORE THAN ONE entry per key — one per
    // occurrence the model emitted (see LensField::occurrence). v1's additions
    // rode in place because they were purely additive and a strict importer
    // stayed safe; this changes a STRUCTURAL invariant that importers can have
    // relied on (fields.size() == key_vocabulary.size(), one field per concept),
    // so it cannot ride. An importer that maps key -> value LAST-wins silently
    // flips from the first repeated value to the last — a wrong value, no error.
    // Same severity class as v2's removal of presence_grounded ⇒ version bump.
    // The first entry for each key is unchanged in value and position, so a
    // first-match importer is unaffected.
    // v4 (2026-09-06, docs/plan-candidate-set.md, architect-approved): adds
    // `key_candidates` + top-level `candidates_error`, additive — a v3 importer
    // that ignores unknown top-level members is unaffected. Bumped anyway
    // (not ridden in place) because the ABSENCE of `key_candidates` is now a
    // load-bearing fact: a v4 response with no `key_candidates` member means
    // "the finder failed on this document"; a v3 response means "this server
    // has no finder." An importer stuck on v3 cannot tell those apart, so the
    // version string is what lets it refuse loud instead of reading absence as
    // silence — see ../qemmi-lens ACCEPTED_FORMAT_VERSIONS, which must add
    // "qemmi-lens/v4" or every extract call fails its own fail-loud gate.
    std::string format_version = "qemmi-lens/v4";

    // ── extraction_origin — who produced the values in this report ───────────
    // "generated": THIS model emitted the JSON (POST /v1/extract). The report
    //              describes where the producing model looked while writing it.
    // "supplied":  the CALLER supplied the JSON and this model only read it
    //              (POST /v1/verify, teacher-forced). The report describes where
    //              THIS model attends to SOMEONE ELSE'S answer.
    //
    // Those are different claims and the format must not blur them. The lens
    // sells "the record is faithful"; a verify report is faithful about this
    // model's reading, NOT about how the values were arrived at — which is
    // exactly the distinction docs/plan-lens-server-shape.md §3.6 flags before
    // cross-model verification is ever offered as a feature.
    //
    // Set by compute_lens_report from LensRun::extraction_supplied — do NOT
    // assign it at a call site. The same flag words the shape-contract refusal
    // (lens_locate_or_throw), so a second place to set the origin is a place
    // for a 422 to disagree with the report it would have produced.
    //
    // ADDITIVE, and deliberately NOT a version bump — unlike v4's
    // `key_candidates`, whose absence was ambiguous. Absence here is not:
    // a payload without this member came from a server that had no /v1/verify,
    // so its values are necessarily "generated". An importer may therefore read
    // absent as "generated" and be right. See docs/lens-format.md.
    std::string extraction_origin = "generated";

    // ── config — the numerical configuration that produced this report ───────
    // `model` names the CALIBRATION ENTRY, which is a coarser thing than it
    // looks: it says "Qwen3.8-9B", not which quantization, not which attention
    // implementation, not which KV element type. Those all move decisions, and
    // two reports that differ because of them would otherwise look like two
    // reports that differ because the document changed — which is exactly the
    // comparison /v1/verify and the client's diff view exist to support.
    //
    // Measured reason this is not hypothetical (plan-lens-server-shape.md §4):
    // a permitted configuration change moves a coverage peak by ~8e-4 on a
    // dense model, against a nearest-line margin of ~1.3e-3. On an MoE it
    // changes the extraction outright on one document in fifteen. A report
    // that cannot say which configuration produced it cannot be compared with
    // another one honestly.
    //
    // Derived in ONE place from the server's own state
    // (http_server.cpp's lens_config_stamp), never assembled at a call site:
    // two builders is a place for two reports to disagree about one server.
    // Empty ⇒ never stamped — which is the case for every in-process caller,
    // including the unit tests — and the member is then omitted from the JSON
    // entirely, leaving those payloads byte-identical to before it existed.
    //
    // ADDITIVE and NOT a version bump, same reasoning as extraction_origin:
    // absence is unambiguous (a server too old to stamp it) and an importer
    // that ignores it reads exactly the payload it read before. It is the
    // reversible choice — a bump can be added later, un-bumping cannot.
    struct RuntimeConfig {
        std::string weights;    // metadata weights_hash, hex — arch+shape+QUANT+layout
        std::string attention;  // "materialized" | "flash-prefill" | "flash"
        std::string kv_type;    // "f32" | "f16" | ...
        bool empty() const { return weights.empty() && attention.empty() && kv_type.empty(); }
    };

    // ── What `attention` must say, and why it is NOT just the server flags ──
    //
    // The contract (lens-format.md, `config`) is "the numerical configuration
    // that produced THIS report", and "two reports are only comparable when
    // this matches". A label read straight off the server's flags breaks both
    // halves of that on ONE route, which is why this is a named rule with
    // tests rather than a ternary at the stamp site.
    //
    // /v1/locate has exactly one pass, and it is the TAPPED prefill.
    // run_lens_locate forces it to Materialized unconditionally, because flash
    // never writes kq_soft and kq_soft is the whole of what locate reads. So a
    // locate report from a --flash-attn server is byte-comparable with one
    // from a plain server. Stamping it "flash-prefill" — which the flags alone
    // would do — tells a reader to discard a comparison that is in fact valid.
    //
    // /v1/extract and /v1/verify are genuinely different: each has an UNTAPPED
    // prompt prefill over the document that DOES run under the server's
    // setting (verify forces Materialized only for its second, tapped pass
    // over the extraction tokens). For those two the flag is the truth.
    enum class RoutePrefill {
        HonoursServerFlag,   // extract, verify — untapped prompt prefill runs as configured
        AlwaysMaterialized,  // locate — its only prefill is the tapped one
    };

    RuntimeConfig config;

    // ── Routing digest (MoE only; docs/plan-lens-only-engine.md §3) ──────────
    //
    // WHICH EXPERTS RAN, as a fingerprint rather than a payload. A MoE router's
    // top-k is an argmax with no margin, so the same document routed on a
    // different build, driver or prefill shape selects different experts —
    // measured 1-6% of selections, changing 1 extraction in 15. Dense models
    // have none of this and emit nothing here.
    //
    // WHAT IT IS FOR. Same configuration, comparing two reports' `digest` is an
    // exact, nearly free regression detector: did this refactor or ggml bump
    // change which experts run? That is the question it answers well.
    //
    // WHAT IT IS NOT. A cross-configuration pass/fail gate. Because 1-6% of
    // selections flip under any perturbation, the whole-trace digest differs on
    // essentially EVERY cross-config comparison while the extraction changes
    // about 1 time in 15 — a gate on equality would cry wolf ~14 times out of
    // 15. `per_layer` exists so a comparison can report HOW MANY layers
    // diverged and WHERE, which is a magnitude a reader can weigh, instead of a
    // boolean that is almost always "differs". Carrying the selections
    // themselves (~1.3 MB) would permit exact replay; that is a later slice and
    // deliberately not this one.
    //
    // ADDITIVE, absent on dense models and on any server too old to stamp it —
    // same reversible reasoning as RuntimeConfig above, and no version bump.
    struct RoutingDigest {
        // The result of comparing THIS pass's routing against one the caller
        // supplied (normally `routing` lifted straight out of an earlier
        // report). Absent unless the caller supplied one.
        //
        // READ `identical` CAREFULLY — it answers a narrower question than it
        // looks. It is meaningful only between LIKE passes: verify vs verify
        // across two machines or builds, or extract vs extract. An EXTRACT
        // digest compared against a VERIFY digest differs by CONSTRUCTION and
        // not because anything is wrong: extract decodes the JSON one token at
        // a time while verify teacher-forces the same tokens as one prefill, so
        // the same positions are computed at different batch shapes — which is
        // precisely the perturbation a top-k argmax turns into a different
        // expert. Expect `identical: false` on every honest extract→verify
        // pair; `layers_diverged` is the number to read there, not the flag.
        struct Expected {
            std::string digest;                  // what the caller supplied
            bool identical             = false;
            int  layers_diverged       = 0;
            int  layers_compared       = 0;
            // Index into per_layer, NOT a block index: per_layer is ordered by
            // ascending MoE layer, and a recipe whose MoE sits on alternate
            // blocks has index != block.
            int  first_diverged_index  = -1;
            bool empty() const { return digest.empty(); }
        };

        std::string              digest;      // FNV-1a over the whole trace, hex
        std::vector<std::string> per_layer;   // one hex digest per MoE layer
        // The selections themselves, base64, carried ONLY when the caller asks
        // (`include_routing_trace`) because it is ~100s of KB. Truncated to the
        // layers a verify pass can reach — deeper ones are unreachable there by
        // construction, so shipping them would be shipping what nobody can use.
        // Bound to the token ids it was captured over; replaying it onto
        // different tokens is refused rather than guessed.
        std::string              trace;
        // True when THIS report's routing was replayed from a supplied trace
        // rather than chosen by the router. The receipt should say which.
        bool                     replayed = false;
        int    layers    = 0;
        int    top_k     = 0;
        size_t positions = 0;
        Expected expected;
        bool empty() const { return digest.empty(); }
    };
    RoutingDigest routing;

    std::string model;
    bool        validated_envelope = true;
    LensConstants k;

    int prompt_len = 0, doc_lo = 0, doc_hi = 0;
    int n_messages = 0;                     // 0 ⇒ the request sent a plain document
    std::string document_text;              // the user's raw document
    std::string raw_json;                   // exactly what the model emitted

    std::vector<LensField>            fields;    // structured, importer-facing
    std::vector<LensPromptToken>      prompt;    // viewer: token stream + region
    std::vector<std::string>          gen;       // viewer: gen token texts
    std::vector<std::vector<LensCitation>> hover; // viewer: per-gen-token citations
    std::vector<double>               heat;      // viewer: per-prompt-token coverage peak
    std::vector<LensCoverageSpan>     skipped;   // "possibly not incorporated" (peak < used)

    // ── Candidate set (docs/plan-candidate-set.md) — producer + wire ─────────
    // Populated only when LensExtractOptions::want_candidates is true (default
    // false ⇒ pass 2 never runs and these stay default-constructed/empty — the
    // off-path is byte-inert). Keyed by concept key. lens_report_to_json emits
    // this (with `anchor` + `returned_as` derived, not stored) as of v4 — see
    // the header comment on `format_version` above.
    std::map<std::string, std::vector<LensCandidate>> key_candidates;
    // true ⇒ pass 2 generated non-empty output that parsed to ZERO candidate
    // lines across the whole key vocabulary — a PRODUCER failure (the model
    // did not honor the requested `key: "span"` / `key: (none)` shape), and
    // must never be read as "the document offers nothing for every key". This
    // is the fourth state docs/plan-candidate-set.md's own states table was
    // missing (absorbed from the m_en1 probe finding). key_candidates stays
    // empty when this is true.
    bool        candidates_producer_failed = false;
    std::string candidates_error;   // set iff candidates_producer_failed

    // True iff this extract ran a QUESTION vocabulary (docs/plan-question-keys.md).
    // Drives two additive wire members, both absent in ordinary key mode:
    //   "vocabulary_mode":"questions"
    //   "uncalibrated":["badge","coverage"]
    // The probe measured the CITATION head under questions and nothing else, so
    // `badge` (ungrounded_body_mass) and coverage (coverage_used_peak) are
    // running on coordinates measured for identifier extraction. Disclosing that
    // is the whole point: the alternative is a receipt we cannot back, which is
    // the defect class per-model calibration was introduced to kill.
    bool question_vocabulary = false;

    // True iff this extract REUSED a primed document prefix rather than
    // re-prefilling it. Surfaces as `"prefix":"warm"`; absent when cold. A warm
    // citation mass must never be compared against a cold one without the
    // reader knowing which is which — same disclosure discipline as
    // `uncalibrated`.
    bool prefix_warm = false;
    // Set true iff LensExtractOptions::want_candidates was true for this
    // extract — i.e. pass 2 was attempted at all, success or failure. This is
    // the ONLY way lens_report_to_json can tell "candidates were not
    // requested" (this stays false, key_candidates stays empty, no error) apart
    // from "requested and ran to a legitimate empty result" — both leave
    // key_candidates empty, but only the former must render `key_candidates`
    // absent-with-no-error on the wire; the two are otherwise indistinguishable
    // from key_candidates/candidates_producer_failed alone. Not itself
    // serialized (it is a driver-side fact, not a document fact).
    bool        candidates_requested = false;
};

// The one rule for LensReport::RuntimeConfig::attention. Free and pure so the
// rule itself is unit-testable — the defect it fixes was a WRONG RULE, not a
// wrong call site, and a ternary inlined at the stamp had nowhere to assert it.
// See LensReport::RoutePrefill for why locate is not the server's flags.
inline const char* lens_attention_label(LensReport::RoutePrefill route,
                                        bool decode_is_flash,
                                        bool prefill_is_flash) {
    // Locate's only pass is its tapped prefill, forced materialized whatever
    // the server was started with. Checked FIRST and unconditionally: this is
    // a property of the route, not a value the flags get a vote on.
    if (route == LensReport::RoutePrefill::AlwaysMaterialized) return "materialized";
    if (decode_is_flash)  return "flash";
    if (prefill_is_flash) return "flash-prefill";
    return "materialized";
}


// Compute the lens report from one tapped run. Pure; fails loud (throws
// std::runtime_error) only on structurally-impossible input (row width vs n_kv).
LensReport compute_lens_report(const LensRun& run, const LensConstants& k = {});

// ── A5.4: "the lens never lies about where the model looked" ─────────────────
// The product invariant, as a pure predicate over a report — promoted out of
// test_server_lens.cpp so the deterministic unit gate and the LIVE gate
// (QDOCS_S1, free-form output) measure it with the SAME ruler rather than two
// drifting copies. Same reason the tier heuristic is shared, not re-implemented.

// Does [c_lo,c_hi) overlap ANY occurrence of `value` in `document` (± tol bytes)?
// "Any occurrence", not just the first: citing a later duplicate (a conflict's
// second copy) is faithful, not a false receipt — the lens names no conflict
// winner (CF1), so either real source is a faithful receipt.
bool lens_cites_a_real_source(const std::string& document, const std::string& value,
                              size_t c_lo, size_t c_hi, long tol = 2);

// THE GATE. Count fields whose confident receipt is not faithful: a grounded,
// distinctive, verbatim value whose top-1 citation lands on no occurrence of
// that value. Anything counted here is a lie the format would be telling.
// Scoped deliberately — ungrounded ⇒ not a confident claim; short_numeric ⇒ the
// weak class the format makes no citation claim for (plan §1.3); not-verbatim ⇒
// its own disclosure (found_in_document=false). Required to be ZERO on any
// corpus, constrained or free.
int lens_count_confident_false_receipts(const LensReport& r, long tol = 2);

// Serialize a report to the lens-format JSON (a superset of the demo's data
// shape so docs/demo/attention-lens.html renders it; P3 owns the spec).
std::string lens_report_to_json(const LensReport& r);

// ── The shape contract (docs/lens-format.md) ─────────────────────────────────
// Thrown when the model's output for a document cannot be parsed into a JSON
// object. A DISTINCT type, because the endpoint must answer 422
// unparseable_extraction rather than 400 bad_request: the request was fine, the
// model's output was not, and an importer has to tell those apart without
// string-matching a message. Never a partial extraction — an unparseable document
// is a loud refusal, which is strictly better than a constraint that corrupts the
// output to avoid a failure it does not actually prevent.
struct LensUnparseableError : std::runtime_error {
    LensUnparseableError(const std::string& what, std::string raw_output)
        : std::runtime_error(what), raw(std::move(raw_output)) {}
    std::string raw;  // exactly what the model emitted, for the 422 body
};

// TOLERANT on shape: skip a ``` fence, then locate the OUTERMOST {...} in `raw`
// by brace-depth (string- and escape-aware, so a brace inside a value does not
// end it). Returns its byte span [lo,hi) within `raw` — byte offsets, not a
// substring, because the gen-token span math must stay anchored to `raw`.
// false ⇒ no object found. Tolerance is bounded and mechanical: it recovers
// SHAPE and never guesses CONTENT.
bool lens_find_json_object(const std::string& raw, size_t& lo, size_t& hi);

// ── Concepts ─────────────────────────────────────────────────────────────────
// A hinted concept. The complete hint is what holds key names stable (Leg B) —
// dropping the grammar does not reopen the naming zoo; dropping the hint would.
//
// `gloss` is ACCEPTED AND CURRENTLY UNUSED. Its only consumer was the deleted
// presence gate's Pass-A question (where it lifted recall 0.75 → 0.92). It is
// kept in the request shape deliberately: removing it would be a second breaking
// change, and it is a plausible future lever. It is NOT fed into the extraction
// instruction — that would silently change the exact prompt regime Stage 1
// validated, on no measurement. If a glossed instruction is ever wanted, measure
// it first.
// One vocabulary entry. `key` is ALWAYS the join key — `fields` and
// `key_candidates` are keyed by it, never by prose — so a question entry still
// carries a short stable id in `key` (plan-question-keys.md §5: "the sentence is
// never a map key").
//
// `question` is the OPTIONAL question form (docs/plan-question-keys.md). Empty
// ⇒ today's identifier vocabulary, byte-identical prompt, nothing changes.
// Non-empty ⇒ the instruction asks the question instead of naming the key, and
// the model still answers with a VERBATIM SPAN — measured 2026-09-07 at 97%
// extractiveness and 99.5% citation top-3 on Qwen 3.8-9B, marginally BETTER
// than the identifier control (plan §9).
//
// `gloss` remains accepted and UNUSED. Do not repurpose it to carry the
// question: it is deliberately kept out of the instruction so the prompt stays
// byte-identical to the regime Stage 1 measured, and routing prose through it
// would invalidate the calibration silently, with no version bump and no gate.
struct LensConcept { std::string key, gloss, question; };

// ── Warm document (docs/plan-lens-warm-document.md) ──────────────────────────
// The lens prompt is `document + instruction_suffix`, so an edit to the key
// vocabulary changes only a ~40-80 token SUFFIX of a multi-thousand-token
// prompt. Re-prefilling the document on every edit is the dominant cost of the
// UI's key-editing loop: measured 2026-09-07, skipping it saves 92.6% of pass-1
// prefill at 1K tokens and 98.9% at 8K (plan §8.2).
//
// The mechanism is deliberately NOT a snapshot. `--attention-lens` is
// single-slot EXCLUSIVE, so slot 0 belongs to the lens alone; a decode writes
// only at positions >= the split, so the document's KV at [0, prefix_tokens)
// survives untouched between requests. Warming is therefore "do not clear the
// slot, rewind the cache position" — no serialization, no PrefixLibrary, no
// disk. This is why the feature is small.
//
// Held by the server across requests, and reset fail-loud whenever anything it
// depends on could have changed.
struct LensWarmDocument {
    std::string document_id;        // caller's handle; empty ⇒ nothing held
    size_t      document_hash = 0;  // of the document BYTES — see http_server
    uint32_t    prefix_tokens = 0;  // tokens of `document` in the rendered prompt
    bool        valid = false;      // false ⇒ slot 0 holds nothing reusable

    void invalidate() { *this = LensWarmDocument{}; }
};

// ── Driver ───────────────────────────────────────────────────────────────────
struct LensExtractOptions {
    int  max_new_tokens = 512;   // hard cap on the emitted JSON length
    // Carry the expert SELECTIONS, not just their fingerprint, so a later
    // /v1/verify can replay them and reproduce this pass's routing exactly.
    // Off by default: the payload is ~100s of KB, and most callers want the
    // digest only.
    bool include_routing_trace = false;
    // Opt-in warm handle (docs/plan-lens-warm-document.md §2.2). Empty ⇒ today's
    // behaviour exactly: cold prefill, nothing stored. Non-empty ⇒ the caller
    // asserts this is the same document it named last time, and the server
    // reuses the primed prefix if it still holds one for that id.
    //
    // Explicit rather than transparent ON PURPOSE: the lens sells receipts, and
    // a transparent cache would let the same request return different citation
    // masses depending on invisible server state. The caller knows when a
    // document is "the same" across an edit; the server does not.
    std::string document_id;
    // Server-owned warm state, borrowed for this call. nullptr ⇒ no warming
    // regardless of document_id.
    LensWarmDocument* warm = nullptr;
    bool validated_envelope_only = false;  // reserved; false = accept + disclose
    // Per-request toggle for the candidate set (docs/plan-candidate-set.md).
    // Default OFF: pass 1 is untouched either way, and false means run_lens_extract
    // does not run pass 2 at all — no second prefill, no extra decode, no
    // behaviour change of any kind. No server flag / CLI flag exists for this;
    // it is request-scoped only, by design. The wire contract for what
    // want_candidates=true PRODUCES (`key_candidates`, `anchor`, `returned_as`,
    // format_version qemmi-lens/v4) is architect-approved and implemented
    // (docs/plan-candidate-set.md); still out of scope is any route/flag that
    // would let an HTTP caller flip this bit — that remains a separate decision.
    bool want_candidates = false;
    // Message boundaries, as byte offsets into `document`, when the caller sent
    // `messages` instead of a flat `document` (the server joins them and fills
    // this in). Empty ⇒ plain document, and every citation reports message -1.
    // Boundaries buy ATTRIBUTION ("this value was read from message 23"), which
    // is all the lens claims. They deliberately do NOT buy a staleness alarm:
    // measured, a later-message rule cried wolf on 7 of 9 correctly-handled
    // corrections and stayed silent on the real failure, because a later message
    // routinely restates an old value (docs/note-ss3-matched-pairs.md §3).
    std::vector<size_t> message_offsets;
};

// Assemble the ChatML thinking-off prompt from (document, concepts), run the
// FREE tapped decode single-slot, and compute the report — one prefill, one pass.
// Fields come back in `concepts` order; a hinted concept the model did not emit
// comes back absent (value:null, badge:"absent"). Borrows fp/sched/tok by
// reference — owns none of them.
//
// Fails loud: std::runtime_error on empty concepts / empty document / a prompt
// exceeding the model's context; LensUnparseableError (⇒ 422) when the emitted
// output holds no parseable JSON object.
//
// `control_arm_grammar` is a PROBE-ONLY seam and defaults to nullptr — production
// omits it and decodes free. QDOCS_S1 passes the refuted `lens_grammar_gbnf()`
// through it to run the constrained arm against the free one on this same driver,
// which is the only reason the comparison stays honest rather than measuring a
// lookalike. `control_arm_vocab` is the grammar's token table and must be non-null
// exactly when the grammar is. NOT reachable from /v1/extract; unrelated to the
// server's per-request `grammar` field on the OpenAI endpoints.
//
// NOTE: `::Tokenizer` is force-qualified. Tokenizer is a global type, but
// inference_server.h forward-declares a phantom `qinf::Tokenizer`; without
// the `::` this declaration would bind to that phantom inside namespace qinf
// wherever both headers are visible (e.g. http_server.cpp), mismatching the
// definition.
LensReport run_lens_extract(ForwardPassBase* fp, ggml_backend_sched_t sched,
                            ::Tokenizer* tok, const ModelMetadata& meta,
                            uint32_t vocab_size, uint32_t n_ctx_max,
                            const std::string& document,
                            const std::vector<LensConcept>& concepts,
                            const LensExtractOptions& opts,
                            // No default: the constants are per model (see
                            // kLensCalibrations). A defaulted `k` would silently
                            // run Qwen 3.6 coordinates on whatever is loaded —
                            // the false receipt the refusal exists to prevent.
                            const LensConstants& k,
                            GrammarVocab* control_arm_grammar = nullptr,
                            const std::vector<std::string>* control_arm_vocab = nullptr);

// ── Verify: teacher-forced re-audit (docs/plan-lens-server-shape.md §3) ──────
// POST /v1/verify's driver. Given a document, its COMPLETE key vocabulary (same
// shape/order contract as extract's — the report's field ordering and absent-
// by-omission marking both depend on it) and a KNOWN extraction, reproduce the
// lens report WITHOUT generating: one prefill over the prompt, then one
// head-less TAPPED prefill over the extraction text (teacher-forced as the
// assistant's answer) — no decode loop, no sampling. The forward pass is
// truncated after max(citation_layer, coverage_layer): causality means an
// attention layer cannot depend on a layer above it, so the tapped rows are
// IDENTICAL to what an untruncated pass (or the original decode-time
// extraction) would have produced, and the layers above the cutoff never run.
// See docs/plan-lens-server-shape.md §3.2-§3.3.
//
// `extraction` is teacher-forced VERBATIM as the assistant's answer — pass the
// exact text a prior report's `raw` field carried for the tightest
// reproduction. It is tokenized directly; no JSON re-serialization happens
// here, because re-serializing a parsed value does not promise the same token
// boundaries the original run produced.
//
// Not bit-for-bit identical to the report `extraction` was derived from: the
// batch-vs-single-token numerical fork (architecture.md §11, "…except where
// the hardware forbids it") means a teacher-forced multi-row prefill and a
// token-by-token decode take different Metal kernels. Measured at 5.6e-4
// against a 0.019 decision margin (plan §2.1) — decision-for-decision stable,
// not byte-identical.
//
// Fails loud exactly like run_lens_extract: std::runtime_error on empty
// concepts/document/extraction, a concept key empty, a mixed question/
// identifier vocabulary, or a prompt+extraction exceeding the model's context;
// LensUnparseableError (⇒ 422) when `extraction` holds no parseable JSON
// object — verify cannot audit a value it cannot locate, same as extract
// cannot emit one.
//
// Single-slot, exclusive — same discipline as run_lens_extract (slot 0, the
// only correct qwen36 decode KV gather, architecture.md §12); the caller holds
// the model lock for the whole call. No warm-document reuse in this version:
// every call is a cold prefill of (document, extraction) — §4/§5's warm-verify
// archive is client-side and out of scope here.
LensReport run_lens_verify(ForwardPassBase* fp, ggml_backend_sched_t sched,
                           ::Tokenizer* tok, const ModelMetadata& meta,
                           uint32_t n_ctx_max,
                           const std::string& document,
                           const std::string& extraction,
                           const std::vector<LensConcept>& concepts,
                           const std::vector<size_t>& message_offsets,
                           const LensConstants& k,
                           // Optional: a routing fingerprint from an earlier
                           // report, to compare this pass against. Null (the
                           // default) skips the comparison entirely and keeps
                           // every existing caller byte-identical.
                           const LensReport::RoutingDigest* expected = nullptr);

// ═════════════════════════════════════════════════════════════════════════════
// LOCATE — where do these keys look? No generation, no audit.
// (../qemmi-lens/docs/plan-locate-and-cut.md §3; plan-lens-only-engine.md §5)
// ═════════════════════════════════════════════════════════════════════════════
//
// The third verb, and the WEAKEST of the three claims. Extract says where the
// model looked while WRITING values; verify says where it attends while READING
// values it was handed; locate says only **where these key tokens look**. No
// value is produced, none is audited, and `extraction_origin` is deliberately
// ABSENT from the response rather than set to either of the other two — a
// locate report is not an extraction of any origin, and a consumer that reads
// it as one is reading a claim that was never made.
//
// Mechanism: one head-less TAPPED prefill over the ordinary lens prompt
// (document + instruction). The instruction names every key, so each key OWNS a
// token span inside the prompt, and the citation head's rows at those positions
// are a retrieval signal over the document — causally available because the
// document precedes the instruction, so every document position is visible to
// every key token. Nothing is generated, so this is cheaper than verify: it
// truncates after `citation_layer` ALONE (coverage is not read), which is 4
// blocks of 40 on Qwen 3.6-35B against verify's 12.
//
// ── WHAT THIS IS NOT CALIBRATED FOR ─────────────────────────────────────────
// The citation head was selected and validated for key→VALUE extraction: rows
// of GENERATED tokens attending back to their source. Key-as-QUERY retrieval is
// the same head read in a regime no probe has scored. `LensLocateReport::
// uncalibrated` is therefore hardcoded true, and it is not a placeholder to be
// flipped when someone feels confident — it comes off when a probe measures
// this regime, and not before.
//
// Two further properties a consumer must carry rather than smooth over:
//
//   * **Attention sums to 1, so absence does not look like absence.** A key
//     whose answer is not in the document still returns a ranked list. There is
//     no abstention signal here and none may be synthesized from `mass` — that
//     is scalar confidence, which has been refuted three times on this codebase
//     (SCORE2 BAR2, the presence gate, the margin-measures-difficulty finding).
//   * **`mass` is comparable within one hit list only.** Not across keys, not
//     across documents, and never against a report's `body_mass`, which is a
//     different head's mean over different rows.
struct LensLocateHit {
    // DOCUMENT-relative byte span, exactly like LensCitation's — the caller
    // slices its own document with these, never the rendered prompt.
    size_t byte_lo = 0, byte_hi = 0;
    // Two different questions about the same span, both published because a
    // caller cutting a document needs both and they can disagree:
    //   `peak` — the single largest citation-head mass in the span. THIS IS THE
    //            ORDERING KEY: hits come back peak-descending, because that is
    //            what the span finder selects on.
    //   `mass` — the SUM over the span's positions. Emphatically not the
    //            ordering key: a wide flat span routinely outsums a sharp one,
    //            and ranking by it would reorder the list away from the signal
    //            the spans were chosen by. (The smoke gate caught exactly this
    //            disagreement on a real document the first time it ran.)
    double mass = 0.0;
    double peak = 0.0;
    // Prompt token positions, for a caller correlating against `prompt_len` /
    // `doc_lo` / `doc_hi`. Absolute, not document-relative.
    int tok_lo = 0, tok_hi = 0;
};

// ── How a key's own query rows are combined before the span finder ──────────
//
// A key occupies several prompt tokens. Their attention rows must become ONE
// mass profile over the document, and which reduction you pick is not a detail.
//
// MAX (default, and what LOCHEAD measured) is right for the keys locate was
// calibrated on: a field name is 1-4 tokens and a mean dilutes the one row that
// carried the retrieval signal with rows that carried punctuation.
//
// MEAN exists because that stops being true the moment a "key" is a SENTENCE.
// Measured 2026-09-20 (DECIDE1, docs/note-lens-qwen38-probe.md): scoring
// category descriptions like "an invoice, a payment, an amount of money owed"
// under MAX collapsed a 4-way routing task into one category — a single filler
// token spiked (`an` against ` agreement`, 0.507, with the same key's next span
// at 0.047) and descriptions carrying more filler simply won. Averaging over
// the key's rows removes that: +25 points on the same corpus at identical
// latency, and the position sensitivity largely went with it.
//
// So this is a property of the KEY KIND, not a better default: short names want
// MAX, sentences want MEAN. The default stays MAX and is byte-identical to the
// behaviour that predates this enum.
enum class LensKeyAggregation { Max, Mean };

// ── Which JOB's calibrated pair this request reads ──────────────────────────
// Locate ("where does this key's answer sit") and Choice ("which of these
// option descriptions fits this document") are different jobs with different
// heads — see the note above LensConstants::locate_layer. The route reads one
// pair per request and says which on the wire, because the two carry different
// provenance and therefore different rates.
enum class LensHeadRole { Locate, Choice, Absent, Score };

inline const char* lens_head_role_name(LensHeadRole r) {
    switch (r) {
        case LensHeadRole::Choice: return "choice";
        case LensHeadRole::Absent: return "absent";
        case LensHeadRole::Score:  return "score";
        case LensHeadRole::Locate: break;
    }
    return "locate";
}

inline const char* lens_key_aggregation_name(LensKeyAggregation a) {
    return a == LensKeyAggregation::Mean ? "mean" : "max";
}

struct LensLocateReport {
    std::string model;
    LensReport::RuntimeConfig config;
    bool validated_envelope = true;
    // TRUE iff this particular request sits outside what LOCHEAD measured.
    //
    // It is no longer "always true": the sweep scored the key-as-query regime on
    // this model and the pair in LensConstants came out of it, so an ordinary
    // key-mode locate is as calibrated as any other lens number here — same
    // corpus, same bar, same standard as the citation and coverage constants.
    //
    // What LOCHEAD did NOT sweep is the QUESTION form. It ran key mode
    // (lens_build_instruction), so a question vocabulary is a different prompt
    // regime with no measurement behind it, exactly as question mode is already
    // partly uncalibrated on the extract path. That, and only that, sets this
    // now. `locate_provenance` travels beside it so a reader sees the rate
    // rather than a bare boolean.
    bool uncalibrated = false;
    std::string locate_provenance;
    bool question_vocabulary = false;
    // Which reduction produced the mass profile. Serialized, because LOCHEAD
    // measured MAX and a report scored under MEAN is outside that measurement —
    // the same reason `question_vocabulary` is on the wire next to
    // `uncalibrated` rather than left for the caller to remember.
    LensKeyAggregation key_aggregation = LensKeyAggregation::Max;
    // Which calibrated pair produced this report. Serialized beside
    // locate_provenance, which switches with it.
    LensHeadRole head_role = LensHeadRole::Locate;
    int prompt_len = 0, doc_lo = 0, doc_hi = 0;
    // The pair actually read — the LOCATE pair, not the citation pair. Named
    // for what it is: after LOCHEAD these are different heads doing different
    // jobs, and calling this "citation" on the wire would invite a reader to
    // compare it against a citation from a report.
    int locate_layer = 0, locate_head = 0;
    int top_k = 0;
    // Ordered by the caller's key_vocabulary, not by score: the request's order
    // is the one the caller can correlate against, and a key with NO hits keeps
    // its slot with an empty list rather than vanishing. Within one key, hits are
    // ordered by `peak` descending — see LensLocateHit.
    std::vector<std::pair<std::string, std::vector<LensLocateHit>>> hits;
};

std::string lens_locate_to_json(const LensLocateReport& r);

// Single-slot, exclusive — same discipline as run_lens_extract and
// run_lens_verify (slot 0, model lock held by the caller). Fails loud on an
// empty document or vocabulary, a mixed key/question vocabulary, an oversized
// prompt, or a key that cannot be found in the rendered instruction.
//
// `top_k` is the number of SPANS returned per key, not positions. Default 3 in
// the route, because the measured weakness of this signal is twins — adjacent
// near-identical lines — and a caller handed one span cannot see that it was
// contested (RETRIEVE2B: every error was right-family-wrong-variant).
LensLocateReport run_lens_locate(ForwardPassBase* fp, ggml_backend_sched_t sched,
                                 ::Tokenizer* tok, const ModelMetadata& meta,
                                 uint32_t n_ctx_max,
                                 const std::string& document,
                                 const std::vector<LensConcept>& concepts,
                                 const LensConstants& k,
                                 int top_k,
                                 LensKeyAggregation key_agg = LensKeyAggregation::Max,
                                 LensHeadRole head_role = LensHeadRole::Locate);

// Pure span-finder, exposed for tests: turn a per-position mass vector into at
// most `top_k` disjoint token spans, highest peak first.
//
// Greedy by peak: take the largest untaken position, then grow left and right
// while the neighbour holds at least `tail_frac` of that peak, stopping at
// `max_width` positions or at an already-taken one. Growing from the peak
// rather than thresholding globally is what keeps two adjacent values from
// merging into one span that spells neither — the twins case again.
//
// Returns [lo, hi) index pairs into `mass`. A position with mass <= 0 is never
// taken, so a shorter list than `top_k` is a real answer, not a truncation.
std::vector<std::pair<int, int>>
lens_locate_spans(const std::vector<float>& mass, int top_k,
                  double tail_frac, int max_width);

// Pure: order `report.fields` by `concepts` and mark absent-by-omission — a
// hinted concept the model did not emit (or emitted empty) becomes a value-null,
// badge:"absent" field. Keys outside the hint are dropped (the complete hint
// defines the surface; Leg B measured 15/15 key stability at greedy, so this is
// not where data hides). No engine — unit-testable with a synthesized report.
//
// Deliberately does NOT second-guess a value the model DID emit. The presence
// gate's "safety net" (re-mark an ungrounded, non-verbatim value as absent) died
// with the gate: it was only defensible as the second of two independent signals.
// Alone it would be the lens judging a value WRONG — a claim the format refuses
// ("No correctness"). The badges already disclose it; the importer decides.
LensReport apply_absent_by_omission(LensReport report,
                                    const std::vector<LensConcept>& concepts);

// The fixed ChatML instruction that names the complete key vocabulary (plan §1.2
// — a complete hint or the naming zoo returns). Exposed for the startup sanity
// check and tests.
std::string lens_build_instruction(const std::vector<std::string>& key_vocabulary);

// The question-vocabulary instruction (docs/plan-question-keys.md §5). Asks each
// question but still keys the answer by its short id and still demands a
// VERBATIM span — the extractiveness the probe measured is a property of an
// instruction that keeps asking for one, not a property of questions.
// `ids` and `questions` must be the same length; the caller guarantees it.
std::string lens_build_question_instruction(const std::vector<std::string>& ids,
                                            const std::vector<std::string>& questions);

// The REFUTED fixed KV grammar (GBNF text) — docs/note-nogrammar-refutation.md.
// NOT on the product path: it exists solely so the QDOCS_S1 probe can run it as a
// control arm against the free path (see run_lens_extract's control_arm_grammar).
// Kept as exercised, measured history rather than deleted, the same precedent as
// the other refuted machinery. Do not wire this to an endpoint.
const char* lens_grammar_gbnf();

// ── Candidate set — pass 2 (docs/plan-candidate-set.md) ──────────────────────
// Ported from tests/perf/attn_provenance.cpp's CAND=1 probe
// (CAND_PASS2_TASK_PREFIX / cand_parse_pass2), which measured the cheap-kill
// gate: median candidate-set size 1.0 on 75 uncontested keys, 100%
// byte-exactness. Both are PURE (no model, no engine) and exposed for tests;
// the cold decode that produces the raw text pass 2 parses lives in
// server_lens.cpp and is not model-free, so it is not unit-tested here.

// The pass-2 instruction: a fixed task description (asking for every span
// answering each key, quoted verbatim, one per line, "(none)" if absent) plus
// the complete key vocabulary — built the same way as lens_build_instruction.
std::string lens_cand_pass2_instruction(const std::vector<std::string>& keys);

// Tolerant line parser for pass 2's free-form output. Tolerates bullets,
// numbering, and unquoted `key: span` (a strict `key: "span"`-only parser
// silently dropped an entire document's correct output when the model emitted
// every line unquoted, and rendered it as "this document offers no answers" —
// the exact confusion the candidate-set format exists to prevent, reproduced
// one level down). `(none)` is a real ANSWER (no candidate for that key), not
// a malformed line. An unterminated quote IS malformed and is skipped. Returns
// (key, span) pairs in EMISSION order, restricted to `keys` — the caller
// dedups, byte-checks against the document, and sorts into document order.
std::vector<std::pair<std::string, std::string>>
lens_parse_pass2_candidates(const std::string& text, const std::vector<std::string>& keys);

// PURE (no engine): parses pass 2's raw output with lens_parse_pass2_candidates,
// verifies every candidate is a byte-exact slice of `document` (dropping any
// that are not — the format requires exactness, not a disclosure about it),
// dedups per key, sorts each key's candidates into document order (byte_lo
// ascending), and writes the result into `report.key_candidates`.
//
// FAILS LOUD on a producer failure: non-empty `gen_text` that parses to zero
// candidate lines sets report.candidates_producer_failed + candidates_error
// instead of silently leaving key_candidates empty — see the header comment on
// LensReport::candidates_producer_failed for why (docs/plan-candidate-set.md's
// own states table is missing this as its fourth state).
void lens_apply_pass2_candidates(const std::string& document, const std::string& gen_text,
                                 const std::vector<std::string>& keys, LensReport& report);

// A5.3 trust tier of a (present) value, by VALUE SHAPE — deterministic,
// regex-grade: a short bare integer ⇒ "short_numeric" (the weak citation class,
// plan §1.3, coverage-backstopped); anything with structure ⇒ "distinctive"
// (the class the lens claims citations for). Empty ⇒ "" (used for absent).
std::string lens_value_tier(const std::string& value);

}  // namespace qinf
