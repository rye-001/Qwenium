// PR 4.0 — Performance baseline tool.
//
// Runs the canary prompt set through both MoE models (Qwen 3.6 and Gemma 4)
// and records wall-clock throughput metrics. Output JSON is the baseline that
// every Phase 4 PR must beat (or match, for kernels not changed).
//
// Usage:
//   QWEN36_MODEL_PATH=/path/to/qwen36.gguf \
//   GEMMA4_MODEL_PATH=/path/to/gemma4.gguf \
//   ./bin/perf-baseline > tests/fixtures/perf_baseline_post_gemma.json
//
// Self-skips any model whose env var is unset — outputs a partial JSON with
// the missing model marked as "skipped".
//
// Per-layer GPU timing and kernel launch counts are NOT captured here;
// those require a Metal GPU capture session in Xcode Instruments. Run this
// binary inside Instruments (Product → Profile → Metal System Trace) and
// inspect the per-dispatch timing there. The wall-clock numbers here are
// the programmatic gate; the GPU trace is the diagnostic tool.
//
// Decode run length: PERF_DECODE_TOKENS (default 32). Override via env var.

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/forward_pass_base.h"
#include "../../src/loader/tokenizer.h"
#include "../../src/sampling/sampling.h"

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------

static std::vector<std::string> load_canary_prompts(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "perf_baseline: cannot open canary prompts: " << path << "\n";
        std::exit(1);
    }
    std::vector<std::string> prompts;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        prompts.push_back(line);
    }
    return prompts;
}

struct PromptResult {
    std::string prompt;
    int         prompt_tokens;
    double      prefill_ms;
    int         decode_tokens;
    double      decode_ms;
    double      tokens_per_sec;
};

static PromptResult run_one(
    ForwardPassBase&          fp,
    ggml_backend_sched_t      sched,
    Tokenizer&                tok,
    size_t                    vocab_size,
    const std::string&        prompt,
    int                       decode_n)
{
    qinf::GreedySampler sampler;
    const auto vocab = tok.get_vocabulary();

    std::vector<int32_t> tokens = tok.encode(prompt);

    // Prefill
    auto t0 = Clock::now();
    std::vector<float> logits = fp.run_prefill(tokens, 0, 0, sched);
    auto t1 = Clock::now();

    std::vector<float> last_logits(logits.end() - (long)vocab_size, logits.end());
    int next = sampler.sample(last_logits, tokens, vocab);

    // Decode
    auto td0 = Clock::now();
    int decoded = 0;
    for (int i = 0; i < decode_n; i++) {
        std::vector<int32_t> cur = {next};
        int pos = fp.get_cache_pos(0);
        std::vector<float> dl = fp.run_prefill(cur, pos, 0, sched);
        next = sampler.sample(dl, tokens, vocab);
        decoded++;
        if (next == -1) break;
    }
    auto td1 = Clock::now();

    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double decode_ms  = std::chrono::duration<double, std::milli>(td1 - td0).count();
    double tps        = (decode_ms > 0) ? (decoded / (decode_ms / 1000.0)) : 0.0;

    return {prompt, (int)tokens.size(), prefill_ms, decoded, decode_ms, tps};
}

static json benchmark_model(
    const std::string&              model_path,
    const std::vector<std::string>& prompts,
    int                             decode_n)
{
    if (model_path.empty()) {
        return {{"status", "skipped"}, {"reason", "env var not set"}};
    }

    register_builtin_models();   // must be before load_metadata (gguf_loader validates architecture on load)

    std::cerr << "Loading: " << model_path << "\n";
    Model model;
    model.load_metadata(model_path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    const uint32_t ctx_len = 2048;   // cap KV cache for baseline — full context_length OOMs on 32GB
    auto fp = create_forward_pass(model, &meta, ctx_len, 1);
    ggml_backend_sched_t sched = model.get_scheduler();
    Tokenizer* tok = model.get_tokenizer();

    std::vector<PromptResult> results;
    for (const auto& p : prompts) {
        fp->clear_slot(0);
        fp->set_cache_pos(0, 0);
        std::cerr << "  prompt: \"" << p << "\"\n";
        results.push_back(run_one(*fp, sched, *tok, meta.vocab_size, p, decode_n));
    }

    // Summary
    double mean_tps = 0, mean_prefill = 0;
    for (const auto& r : results) {
        mean_tps     += r.tokens_per_sec;
        mean_prefill += r.prefill_ms;
    }
    mean_tps     /= results.size();
    mean_prefill /= results.size();

    json jresults = json::array();
    for (const auto& r : results) {
        jresults.push_back({
            {"prompt",        r.prompt},
            {"prompt_tokens", r.prompt_tokens},
            {"prefill_ms",    r.prefill_ms},
            {"decode_tokens", r.decode_tokens},
            {"decode_ms",     r.decode_ms},
            {"tokens_per_sec",r.tokens_per_sec}
        });
    }

    return {
        {"status",              "ok"},
        {"model_path",          model_path},
        {"architecture",        meta.architecture},
        {"block_count",         meta.block_count},
        {"results",             jresults},
        {"summary", {
            {"mean_tokens_per_sec", mean_tps},
            {"mean_prefill_ms",     mean_prefill}
        }},
        {"_gpu_capture_note",
            "Per-layer timing and kernel launch counts require a Metal GPU "
            "capture session (Xcode: Product → Profile → Metal System Trace). "
            "Run this binary inside Instruments and inspect per-dispatch timing."}
    };
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    const char* qwen36_path = std::getenv("QWEN36_MODEL_PATH");
    const char* gemma4_path = std::getenv("GEMMA4_MODEL_PATH");
    const char* canary_path_env = std::getenv("CANARY_PROMPTS_PATH");
    const char* decode_n_env    = std::getenv("PERF_DECODE_TOKENS");

    std::string canary_path = canary_path_env
        ? canary_path_env
        : CANARY_PROMPTS_DEFAULT;

    int decode_n = decode_n_env ? std::atoi(decode_n_env) : 32;

    if (!qwen36_path && !gemma4_path) {
        std::cerr << "perf_baseline: set QWEN36_MODEL_PATH and/or GEMMA4_MODEL_PATH\n";
        std::cerr << "  example:\n";
        std::cerr << "  QWEN36_MODEL_PATH=/path/to/qwen36.gguf \\\n";
        std::cerr << "  GEMMA4_MODEL_PATH=/path/to/gemma4.gguf \\\n";
        std::cerr << "  ./bin/perf-baseline > tests/fixtures/perf_baseline_post_gemma.json\n";
        return 1;
    }

    auto prompts = load_canary_prompts(canary_path);
    std::cerr << "Loaded " << prompts.size() << " canary prompts from " << canary_path << "\n";
    std::cerr << "Decode tokens per prompt: " << decode_n << "\n\n";

    // ISO 8601 timestamp
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&tt));

    json out = {
        {"_note",            "Generated by tests/perf/perf_baseline.cpp (PR 4.0). "
                             "This file is the Phase 4 performance gate — every PR must "
                             "beat or match these numbers on the changed kernel types."},
        {"generated_at",     ts_buf},
        {"decode_tokens",    decode_n},
        {"canary_prompts",   prompts},
        {"qwen36",           benchmark_model(qwen36_path ? qwen36_path : "", prompts, decode_n)},
        {"gemma4",           benchmark_model(gemma4_path ? gemma4_path : "", prompts, decode_n)}
    };

    std::cout << out.dump(2) << "\n";
    return 0;
}
