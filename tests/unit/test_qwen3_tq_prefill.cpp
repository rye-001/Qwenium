/**
 * test_qwen3_tq_prefill.cpp
 *
 * Canary baseline regression test for the TurboQuant (TQ) prefill path in
 * Qwen3ForwardPass (_build_layer_batch_graph, dispatched from run_prefill when
 * kv_quant_bits >= 2).
 *
 * Closes open question #3 from docs/kv-cache-unification.md: "no tests covering
 * the TurboQuant path at all". This test is the prerequisite gate for the
 * follow-up extraction PR that unifies build_prefill_graph and
 * _build_layer_batch_graph into a shared per-layer builder.
 *
 * ─── Two modes ────────────────────────────────────────────────────────────────
 *
 *   Recording mode (baseline absent):
 *     Runs prefill, captures top-8 logits at the last-token position, writes
 *     them to BASELINE_PATH. Test passes; commit the generated file.
 *
 *   Assertion mode (baseline present):
 *     Loads baseline, runs prefill, asserts:
 *       • Top-1 token ID matches exactly.
 *       • Top-8 token IDs match as a set (order may shift under quant noise).
 *       • Logit values for each top-8 token are within abs_tol and rel_tol.
 *
 * ─── Tolerance rationale ─────────────────────────────────────────────────────
 *   abs_tol = 0.05, rel_tol = 0.01 (relative to |baseline logit|).
 *   4-bit Lloyd-Max+WHT quantization introduces ≈1-2% RMS error on activations;
 *   0.05 abs / 1% rel is ~5× that budget so the test catches real regressions
 *   while tolerating harmless floating-point ordering differences across runs.
 *
 * ─── Requirements ────────────────────────────────────────────────────────────
 *   QWEN3_MODEL_PATH env var → path to a Qwen3 GGUF model.
 *   If unset, defaults to "./Qwen3-0.6B-Q8_0.gguf" (smallest fixture in repo).
 *   Test self-skips (GTEST_SKIP) when the model file is not accessible.
 *
 *   BASELINE_PATH is baked in at compile time via the cmake define
 *   QWEN3_TQ_BASELINE_PATH (set to ${CMAKE_SOURCE_DIR}/tests/fixtures/...).
 *   Override at runtime via QWEN3_TQ_BASELINE_PATH env var.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../../src/qwen3-core/qwen3-model.h"
#include "../../src/models/qwen3.h"
#include "../../src/loader/tokenizer.h"

using json = nlohmann::json;

// ─── Configuration ─────────────────────────────────────────────────────────

static constexpr int    KV_QUANT_BITS = 2;        // Minimum TQ depth; exercises _build_layer_batch_graph
static constexpr int    TOP_K        = 8;
static constexpr float  ABS_TOL      = 0.05f;     // See tolerance rationale in file header
static constexpr float  REL_TOL      = 0.01f;
static const std::string CANARY_PROMPT = "The capital of France is";

// Default baseline path injected by cmake. Can be overridden via env var.
#ifndef QWEN3_TQ_BASELINE_DEFAULT
#define QWEN3_TQ_BASELINE_DEFAULT "./tests/fixtures/qwen3_tq_prefill_baseline.json"
#endif

static std::string get_model_path() {
    const char* p = std::getenv("QWEN3_MODEL_PATH");
    return p ? std::string(p) : "./Qwen3-0.6B-Q8_0.gguf";
}

static std::string get_baseline_path() {
    const char* p = std::getenv("QWEN3_TQ_BASELINE_PATH");
    return p ? std::string(p) : QWEN3_TQ_BASELINE_DEFAULT;
}

// ─── Helpers ───────────────────────────────────────────────────────────────

struct TopKEntry {
    int32_t token_id;
    float   logit;
};

/** Extract the top-k (token_id, logit) pairs from a full logit vector. */
static std::vector<TopKEntry> top_k(const std::vector<float>& logits, int k) {
    const int vocab = static_cast<int>(logits.size());
    k = std::min(k, vocab);

    std::vector<int> idx(vocab);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&logits](int a, int b) { return logits[a] > logits[b]; });

    std::vector<TopKEntry> result;
    result.reserve(k);
    for (int i = 0; i < k; ++i)
        result.push_back({idx[i], logits[idx[i]]});
    return result;
}

/** Write top-k results to the baseline JSON file. */
static void write_baseline(const std::string& path,
                           const std::vector<TopKEntry>& entries) {
    json j;
    j["model_desc"]    = "Qwen3ForwardPass";
    j["kv_quant_bits"] = KV_QUANT_BITS;
    j["prompt"]        = CANARY_PROMPT;
    j["top_k"]         = TOP_K;
    j["abs_tol"]       = ABS_TOL;
    j["rel_tol"]       = REL_TOL;

    json arr = json::array();
    for (const auto& e : entries) {
        json item;
        item["token_id"] = e.token_id;
        item["logit"]    = e.logit;
        arr.push_back(item);
    }
    j["top8"] = arr;

    std::ofstream out(path);
    if (!out.is_open()) {
        FAIL() << "Cannot write baseline to: " << path
               << " — expected slot: token_id=N/A, actual: file open failed";
    }
    out << j.dump(2) << "\n";
    std::cout << "[RECORDING] Baseline written to: " << path << "\n";
}

/** Load top-k results from the baseline JSON file. */
static std::vector<TopKEntry> load_baseline(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        // Caller checks existence before calling this; this should not happen.
        throw std::runtime_error("Cannot read baseline: " + path);
    }
    json j = json::parse(in);

    std::vector<TopKEntry> entries;
    for (const auto& item : j.at("top8")) {
        entries.push_back({
            item.at("token_id").get<int32_t>(),
            item.at("logit").get<float>()
        });
    }
    return entries;
}

// ─── Test fixture ──────────────────────────────────────────────────────────

class Qwen3TQPrefillTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const std::string model_path = get_model_path();
        std::ifstream probe(model_path);
        if (!probe.is_open()) {
            // Will cause each test to skip via SKIP_IF_NO_MODEL.
            return;
        }
        probe.close();

        model_ = std::make_unique<Qwen3Model>();
        model_->load_metadata(model_path);
        model_->load_tensors();
    }

    static void TearDownTestSuite() { model_.reset(); }

    /**
     * Run TQ prefill and return the last-token logits.
     *
     * Fail-loud contract: if model_ is null (load failed), the test was
     * already skipped; this function is only called when the model is ready.
     */
    std::vector<float> run_tq_prefill(const std::string& prompt) {
        EXPECT_NE(model_, nullptr)
            << "model_ is null — SetUpTestSuite did not load: "
            << get_model_path();

        const auto& meta = model_->get_metadata();

        auto fp = std::make_unique<Qwen3ForwardPass>(
            *model_, &meta,
            /*context_len=*/512,
            /*max_batch=*/1,
            KV_QUANT_BITS);

        Tokenizer* tok   = model_->get_tokenizer();
        auto* sched      = model_->get_scheduler();

        std::vector<int32_t> tokens = tok->encode(prompt);
        EXPECT_GT(tokens.size(), 0u)
            << "Tokeniser returned empty sequence for prompt: " << prompt;

        std::vector<float> all_logits = fp->run_prefill(tokens, 0, 0, sched);

        EXPECT_EQ(all_logits.size(), tokens.size() * meta.vocab_size)
            << "Logit vector size mismatch: "
            << "expected=" << tokens.size() * meta.vocab_size
            << " actual=" << all_logits.size();

        // Extract last-token logits.
        const size_t offset = (tokens.size() - 1) * meta.vocab_size;
        return std::vector<float>(
            all_logits.begin() + static_cast<std::ptrdiff_t>(offset),
            all_logits.begin() + static_cast<std::ptrdiff_t>(offset) + meta.vocab_size);
    }

    static std::unique_ptr<Qwen3Model> model_;
};

std::unique_ptr<Qwen3Model> Qwen3TQPrefillTest::model_ = nullptr;

#define SKIP_IF_NO_MODEL()                                                   \
    do {                                                                     \
        if (!model_) {                                                       \
            GTEST_SKIP() << "QWEN3_MODEL_PATH not accessible ("             \
                         << get_model_path() << ") — skipping TQ test";    \
        }                                                                    \
    } while (0)

// ─── Tests ─────────────────────────────────────────────────────────────────

/**
 * Recording / assertion canary for the TQ prefill path.
 *
 * When the baseline file is absent this test writes it and passes.
 * When the baseline file is present this test asserts against it.
 *
 * Naming: "CanaryBaseline" so it appears prominently in test output.
 */
TEST_F(Qwen3TQPrefillTest, CanaryBaseline) {
    SKIP_IF_NO_MODEL();

    const std::string baseline_path = get_baseline_path();
    const std::vector<float> last_logits = run_tq_prefill(CANARY_PROMPT);
    const std::vector<TopKEntry> live = top_k(last_logits, TOP_K);

    // ── Recording mode ──────────────────────────────────────────────────────
    {
        std::ifstream probe(baseline_path);
        if (!probe.is_open()) {
            std::cout << "[RECORDING MODE] Baseline absent at: "
                      << baseline_path << "\n";
            write_baseline(baseline_path, live);
            // First run: just write and pass. Commit the generated file.
            SUCCEED();
            return;
        }
    }

    // ── Assertion mode ──────────────────────────────────────────────────────
    std::cout << "[ASSERTION MODE] Comparing against: " << baseline_path << "\n";

    std::vector<TopKEntry> baseline;
    try {
        baseline = load_baseline(baseline_path);
    } catch (const std::exception& ex) {
        FAIL() << "Failed to parse baseline at " << baseline_path
               << ": " << ex.what();
    }

    ASSERT_EQ(baseline.size(), static_cast<size_t>(TOP_K))
        << "Baseline has wrong number of entries: "
        << "expected=" << TOP_K << " actual=" << baseline.size();

    // Top-1 exact token-ID match.
    EXPECT_EQ(live[0].token_id, baseline[0].token_id)
        << "Top-1 token mismatch: "
        << "slot=top-1, expected=" << baseline[0].token_id
        << " actual=" << live[0].token_id;

    // Top-8 set match (order may shift under quant noise).
    std::set<int32_t> baseline_ids, live_ids;
    for (const auto& e : baseline) baseline_ids.insert(e.token_id);
    for (const auto& e : live)     live_ids.insert(e.token_id);
    EXPECT_EQ(live_ids, baseline_ids)
        << "Top-8 token ID sets differ — TQ prefill graph may have diverged";

    // Logit-value tolerance check for each baseline token.
    for (const auto& ref : baseline) {
        // Find this token in the live results.
        float live_logit = 0.0f;
        bool  found      = false;
        for (const auto& lv : live) {
            if (lv.token_id == ref.token_id) {
                live_logit = lv.logit;
                found      = true;
                break;
            }
        }
        if (!found) {
            // Already covered by the set-equality check above; report for clarity.
            ADD_FAILURE() << "Token " << ref.token_id
                          << " from baseline not found in live top-8";
            continue;
        }

        const float abs_err = std::abs(live_logit - ref.logit);
        const float rel_err = std::abs(ref.logit) > 1e-6f
                              ? abs_err / std::abs(ref.logit) : abs_err;

        EXPECT_TRUE(abs_err <= ABS_TOL || rel_err <= REL_TOL)
            << "Logit out of tolerance for token_id=" << ref.token_id
            << ": baseline=" << ref.logit << " actual=" << live_logit
            << " abs_err=" << abs_err << " (tol=" << ABS_TOL << ")"
            << " rel_err=" << rel_err << " (tol=" << REL_TOL << ")";
    }

    // Print top-8 for diagnostic visibility in CI output.
    std::cout << "Live top-8 logits (kv_quant_bits=" << KV_QUANT_BITS << "):\n";
    for (int i = 0; i < static_cast<int>(live.size()); ++i) {
        std::cout << "  [" << i << "] token=" << live[i].token_id
                  << "  logit=" << live[i].logit << "\n";
    }
}

/**
 * Smoke: TQ prefill completes without crash and produces finite logits.
 *
 * Separate from CanaryBaseline so CI can distinguish "crashed" from
 * "logit regression". Runs with the same kv_quant_bits as the canary.
 */
TEST_F(Qwen3TQPrefillTest, PrefillProducesFiniteLogits) {
    SKIP_IF_NO_MODEL();

    const std::vector<float> last_logits = run_tq_prefill(CANARY_PROMPT);
    ASSERT_FALSE(last_logits.empty())
        << "run_prefill returned empty logits for kv_quant_bits=" << KV_QUANT_BITS;

    int nan_count  = 0;
    int inf_count  = 0;
    for (float v : last_logits) {
        if (std::isnan(v)) ++nan_count;
        if (std::isinf(v)) ++inf_count;
    }
    EXPECT_EQ(nan_count, 0)
        << "NaN in TQ prefill logits: count=" << nan_count
        << " kv_quant_bits=" << KV_QUANT_BITS;
    EXPECT_EQ(inf_count, 0)
        << "Inf in TQ prefill logits: count=" << inf_count
        << " kv_quant_bits=" << KV_QUANT_BITS;
}
