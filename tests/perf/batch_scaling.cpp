// Batch-scaling probe — the falsifiable test for the "empty seats" hypothesis.
//
// Claim: on the qwen36 hybrid, one decode step is dominated by kernel-launch
// overhead (DeltaNet) and weight reads, NOT per-sequence math — so decoding B
// sequences in one batched step costs far less than B× one. If true, "N futures
// per step, priced as one" is a real primitive (best-of-N, tree speculation,
// ghost-slot prefill). If batch-8 ≈ 8× batch-1, the hypothesis is dead.
//
// Method: prefill one prompt into slot 0, clone it O(1) into slots 1..B-1
// (clone_slot), then time D batched decode steps at B = 1,2,4,8,10. Each lane
// is fed a DIFFERENT token id per step so MoE expert routing diverges across
// lanes (the realistic, weight-read-heavy case — identical tokens would route
// every lane to the same experts and flatter the result).
//
// Reports, per B: ms/step (all B lanes) and ms/step ÷ B (amortized per-token).
// The headline is ms_step(8) / ms_step(1).
//
//   QWEN36_MODEL_PATH=models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf ./bin/batch-scaling

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/forward_pass_base.h"
#include "../../src/loader/tokenizer.h"

using Clock = std::chrono::steady_clock;

static int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::atoi(v) : fallback;
}

int main() {
    const char* env = std::getenv("QWEN36_MODEL_PATH");
    std::string path = env ? env : "models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf";

    // Batch list, e.g. QINF_BS_BATCHES=1,2,4,8
    std::vector<uint32_t> batches;
    if (const char* bl = std::getenv("QINF_BS_BATCHES")) {
        std::string s(bl), cur;
        for (char c : s + ",") {
            if (c == ',') { if (!cur.empty()) batches.push_back((uint32_t)std::atoi(cur.c_str())); cur.clear(); }
            else cur += c;
        }
    }
    if (batches.empty()) batches = {1, 2, 4, 8, 10};

    uint32_t max_b = 1;
    for (uint32_t b : batches) max_b = std::max(max_b, b);

    const uint32_t MAXB   = (uint32_t)env_int("QINF_BS_MAXB", (int)max_b);
    const uint32_t CTX    = (uint32_t)env_int("QINF_BS_CTX", 1024);
    const int      WARMUP = env_int("QINF_BS_WARMUP", 6);
    const int      TIMED  = env_int("QINF_BS_TIMED", 24);
    const int      PASSES = env_int("QINF_BS_PASSES", 1);
    // 0 => use the historical text prompt; N>0 => N synthetic ids (tokenizer-free,
    // so prompt length is exactly equal across model families).
    const int      PTOK   = env_int("QINF_BS_PROMPT_TOKENS", 0);

    register_builtin_models();
    std::cerr << "Loading " << path << " ...\n";
    Model model;
    model.load_metadata(path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    std::cerr << "arch=" << meta.architecture << " ctx=" << CTX << " maxb=" << MAXB
              << " warmup=" << WARMUP << " timed=" << TIMED << " passes=" << PASSES << "\n";
    auto fp = create_forward_pass(model, &meta, CTX, MAXB);
    ggml_backend_sched_t sched = model.get_scheduler();
    Tokenizer* tok = model.get_tokenizer();

    // A short prompt — decode cost is what we measure, prefill is one-off.
    std::vector<int32_t> prompt;
    if (PTOK > 0) { for (int i = 0; i < PTOK; ++i) prompt.push_back((int32_t)((i * 7 + 3) % 1000)); }
    else          { prompt = tok->encode("Summarize the following order:"); }
    const auto vsz = (int32_t)meta.vocab_size;
    std::fprintf(stderr, "prompt_tokens=%zu\n", prompt.size());

    volatile long long sink = 0;
    std::printf("\n pass | batch |  ms/step | ms/step/lane | vs B=1 (step) | speedup vs serial\n");
    std::printf("------+-------+----------+--------------+---------------+------------------\n");

    for (int pass = 0; pass < PASSES; ++pass) {
    double ms_step_b1 = 0.0;
    for (uint32_t B : batches) {
        // Fresh state: clear every slot, prefill slot 0, clone into 1..B-1.
        for (uint32_t s = 0; s < MAXB; ++s) { fp->clear_slot(s); fp->set_cache_pos(0, s); }
        fp->run_prefill(prompt, 0, 0, sched);
        const int P = (int)fp->get_cache_pos(0);
        for (uint32_t s = 1; s < B; ++s) fp->clone_slot(0, s, (uint32_t)P);

        std::vector<uint32_t> slots(B);
        for (uint32_t s = 0; s < B; ++s) slots[s] = s;

        // Phase 0.1 of plan-deltanet-batched-decode: the decode graph's node
        // count as a function of B. Free — the graph is built anyway — and it
        // is the structural, thermally-immune half of that plan's evidence.
        // QINF_BS_NODES_ONLY=1 stops after this, so the sweep costs one graph
        // build per B instead of a full timed cell.
        {
            std::vector<int32_t> tk(B), ps(B);
            for (uint32_t s = 0; s < B; ++s) { tk[s] = (int32_t)((1000 + s * 131) % vsz); ps[s] = P; }
            ggml_backend_sched_reset(sched);
            ggml_cgraph* gn = fp->build_decoding_graph(tk, slots, ps);
            std::printf("NODES,%s,%u,%d\n", meta.architecture.c_str(), B, ggml_graph_n_nodes(gn));
            // Per-op node histogram, so the O(B) growth can be attributed to
            // specific ops rather than inferred from a total.
            if (std::getenv("QINF_BS_NODE_OPS")) {
                std::map<std::string, int> hist;
                for (int i = 0; i < ggml_graph_n_nodes(gn); ++i) {
                    ggml_tensor* n = ggml_graph_node(gn, i);
                    std::string op = ggml_op_name(n->op);
                    if (n->op == GGML_OP_UNARY) op += std::string("/") + ggml_unary_op_name(ggml_get_unary_op(n));
                    hist[op]++;
                }
                for (const auto& kv : hist)
                    std::printf("NODEOP,%u,%s,%d\n", B, kv.first.c_str(), kv.second);
            }
            std::fflush(stdout);
        }
        if (std::getenv("QINF_BS_NODES_ONLY")) continue;

        auto run_step = [&](int step) {
            std::vector<int32_t> tokens(B);
            std::vector<int32_t> positions(B);
            const int pos = P + step;
            // QINF_BATCH_IDENTICAL=1 feeds every lane the SAME token → maximal
            // MoE expert-read sharing (best case). Default: distinct per lane →
            // maximal routing diversity (worst case). The two bracket real
            // best-of-N, which shares a context and so lands in between.
            static const bool identical = std::getenv("QINF_BATCH_IDENTICAL");
            for (uint32_t s = 0; s < B; ++s) {
                tokens[s]    = identical ? (int32_t)((1000 + step * 7) % vsz)
                                         : (int32_t)((1000 + step * 7 + s * 131) % vsz);
                positions[s] = pos;
            }
            ggml_backend_sched_reset(sched);
            ggml_cgraph* gf = fp->build_decoding_graph(tokens, slots, positions);
            ggml_backend_sched_alloc_graph(sched, gf);
            fp->set_decode_inputs(gf, tokens, slots, positions);
            ggml_backend_sched_graph_compute(sched, gf);
            // QINF_BS_SERVERLIKE=1 adds the per-lane work the *product* loop does
            // and this probe historically did not: a full-vocabulary device->host
            // logits readback plus a full-vocabulary CPU argmax, once per lane
            // (http_server.cpp run_batched_decode). Phase 0.3 of
            // plan-deltanet-batched-decode names this as the reason the server's
            // marginal lane cost must exceed the probe's; this switch is how that
            // claim is tested here rather than only against a live server.
            static const bool serverlike = std::getenv("QINF_BS_SERVERLIKE");
            if (serverlike) {
                for (uint32_t s = 0; s < B; ++s) {
                    std::vector<float> lg = fp->get_output_logits_for_slot(gf, s);
                    int best = 0; float bv = lg.empty() ? 0.0f : lg[0];
                    for (size_t k = 1; k < lg.size(); ++k) if (lg[k] > bv) { bv = lg[k]; best = (int)k; }
                    sink += best;
                }
            }
            for (uint32_t s = 0; s < B; ++s) fp->advance_cache(1, s);
        };

        for (int i = 0; i < WARMUP; ++i) run_step(i);
        auto t0 = Clock::now();
        for (int i = 0; i < TIMED; ++i) run_step(WARMUP + i);
        auto t1 = Clock::now();

        double ms_step = std::chrono::duration<double, std::milli>(t1 - t0).count() / TIMED;
        if (B == 1) ms_step_b1 = ms_step;
        std::printf(" %4d | %5u | %8.2f | %12.2f | %13.2fx | %15.2fx\n",
                    pass, B, ms_step, ms_step / B, ms_step / ms_step_b1,
                    (ms_step_b1 * B) / ms_step);
        std::printf("RESULT,%s,%d,%u,%.4f\n", meta.architecture.c_str(), pass, B, ms_step);
        std::fflush(stdout);
    }
    }
    std::printf("\nHeadline: ms/step(B) grows as ~1x if seats are free, ~Bx if full.\n"
                "'speedup vs serial' = tokens/wall you'd get running B futures batched\n"
                "instead of one at a time.\n");
    return 0;
}
