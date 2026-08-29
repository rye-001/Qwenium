// Decode-step phase decomposition — where does the ~20ms fixed intercept live?
//
// The batch-scaling probe fit ms/step ≈ 20 + 25.6·B. The ~20ms fixed part is
// paid every single-user token (44% of B=1 latency) and is what graph-reuse
// would attack. But only if it's CPU-side REPLANNING (rebuild the identical
// graph + re-plan scratch every step). If instead it lives inside GPU compute
// (launch bubbles between hundreds of tiny DeltaNet kernels), reuse won't help
// and the answer is fusion. This probe times each phase separately to decide.
//
// Phases (one B=1 decode step, the single-user case that matters):
//   build   — build_decoding_graph: re-derive ~2400 ggml nodes (CPU)
//   alloc   — sched_reset + alloc_graph: re-plan scratch memory (CPU)
//   set     — set_decode_inputs: fill typed input slots + upload (CPU→GPU)
//   compute — sched_graph_compute: encode + dispatch + GPU run + sync
//   read    — get_output_logits: readback logits (GPU→CPU sync)
//
//   QWEN36_MODEL_PATH=... ./bin/decode-breakdown
//
// Optional env:
//   QINF_BD_CTX=N            KV context to allocate (default 1024)
//   QINF_BD_PROMPT_TOKENS=N  prefill N synthetic tokens before timing, so the
//                            step is profiled at a realistic n_kv. Decode cost
//                            is not flat in n_kv -- attention scales with it --
//                            so profiling at n_kv~6 understates the GPU share
//                            of any real workload.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <map>
#include <vector>

#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/forward_pass_base.h"
#include "../../src/loader/tokenizer.h"

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

int main() {
    const char* env = std::getenv("QWEN36_MODEL_PATH");
    std::string path = env ? env : "models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf";
    const char* ctx_env = std::getenv("QINF_BD_CTX");
    const uint32_t CTX = ctx_env ? (uint32_t)std::atoi(ctx_env) : 1024;
    const int WARMUP = 8, TIMED = 40;

    register_builtin_models();
    std::cerr << "Loading " << path << " ...\n";
    Model model;
    model.load_metadata(path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    auto fp = create_forward_pass(model, &meta, CTX, 1);
    ggml_backend_sched_t sched = model.get_scheduler();
    Tokenizer* tok = model.get_tokenizer();

    const int32_t vsz = (int32_t)meta.vocab_size;
    std::vector<int32_t> prompt = tok->encode("Summarize the following order:");
    if (const char* pt = std::getenv("QINF_BD_PROMPT_TOKENS")) {
        const int n = std::atoi(pt);
        prompt.clear();
        for (int i = 0; i < n; ++i) prompt.push_back((int32_t)((i * 7 + 3) % 1000));
    }
    std::cerr << "prefill " << prompt.size() << " tokens (ctx " << CTX << ")\n";
    fp->run_prefill(prompt, 0, 0, sched);

    std::vector<double> t_build, t_alloc, t_set, t_compute, t_read;
    int n_nodes_seen = 0;
    const int base_pos = (int)fp->get_cache_pos(0);

    for (int step = 0; step < WARMUP + TIMED; ++step) {
        const std::vector<int32_t>  tokens    = {(int32_t)((1000 + step * 7) % vsz)};
        const std::vector<uint32_t> slots     = {0};
        const std::vector<int32_t>  positions = {(int)fp->get_cache_pos(0)};

        auto t0 = Clock::now();
        ggml_cgraph* gf = fp->build_decoding_graph(tokens, slots, positions);
        double d_build = ms_since(t0);
        n_nodes_seen = ggml_graph_n_nodes(gf);

        auto t1 = Clock::now();
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf);
        double d_alloc = ms_since(t1);

        auto t2 = Clock::now();
        fp->set_decode_inputs(gf, tokens, slots, positions);
        double d_set = ms_since(t2);

        auto t3 = Clock::now();
        ggml_backend_sched_graph_compute(sched, gf);
        double d_compute = ms_since(t3);

        auto t4 = Clock::now();
        volatile std::vector<float> logits = fp->get_output_logits(gf);
        double d_read = ms_since(t4);

        fp->advance_cache(1, 0);

        if (step >= WARMUP) {
            t_build.push_back(d_build);   t_alloc.push_back(d_alloc);
            t_set.push_back(d_set);       t_compute.push_back(d_compute);
            t_read.push_back(d_read);
        }
        (void)base_pos;
    }

    auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    double b = med(t_build), a = med(t_alloc), s = med(t_set),
           c = med(t_compute), r = med(t_read);
    double total = b + a + s + c + r;
    double cpu_fixed = b + a + s + r;  // everything graph-reuse could remove

    std::printf("\n  decode graph nodes: %d\n", n_nodes_seen);

    // Op-type histogram of the decode graph. Node *count* alone cannot say
    // whether a step is matmul-dominated or small-op-dominated; the histogram
    // can, and it is what tells graph-shape work where to aim. Rebuilt once
    // outside the timing loop so it costs nothing measured.
    if (std::getenv("QINF_BD_OPS")) {
        const std::vector<int32_t>  tokens    = {(int32_t)1000};
        const std::vector<uint32_t> slots     = {0};
        const std::vector<int32_t>  positions = {(int)fp->get_cache_pos(0)};
        ggml_cgraph* gh = fp->build_decoding_graph(tokens, slots, positions);
        std::map<std::string, int> hist;
        for (int i = 0; i < ggml_graph_n_nodes(gh); ++i) {
            hist[ggml_op_name(ggml_graph_node(gh, i)->op)]++;
        }
        std::vector<std::pair<std::string,int>> rows(hist.begin(), hist.end());
        std::sort(rows.begin(), rows.end(),
                  [](const auto& x, const auto& y){ return x.second > y.second; });
        std::printf("\n  op-type histogram (decode graph)\n");
        std::printf("  %-20s %6s\n", "op", "nodes");
        for (const auto& kv : rows) std::printf("  %-20s %6d\n", kv.first.c_str(), kv.second);
    }

    // Node listing of the decode graph (QINF_BD_NODES=1): index, op, name.
    // Names are what QINF_BD_CUTS below takes -- not indices, because an index
    // is only meaningful against one particular build. The listing re-builds
    // and reports the first divergence, so the assumption is checked, not
    // assumed.
    if (std::getenv("QINF_BD_NODES")) {
        const std::vector<int32_t>  tokens    = {(int32_t)1000};
        const std::vector<uint32_t> slots     = {0};
        const std::vector<int32_t>  positions = {(int)fp->get_cache_pos(0)};
        ggml_cgraph* gh = fp->build_decoding_graph(tokens, slots, positions);
        std::vector<std::pair<std::string,std::string>> seq;
        std::printf("\n  decode graph nodes (idx op name)\n");
        for (int i = 0; i < ggml_graph_n_nodes(gh); ++i) {
            ggml_tensor* t = ggml_graph_node(gh, i);
            std::printf("  %5d %-20s %s\n", i, ggml_op_name(t->op), ggml_get_name(t));
            seq.emplace_back(ggml_op_name(t->op), ggml_get_name(t));
        }
        // Build-to-build stability of node order, across a full step cycle
        // (build -> reset -> alloc -> build) rather than two bare builds: the
        // scheduler pass in between is what a real decode loop does, and it is
        // where the order was observed to move. It matters beyond this probe --
        // anything that caches a decode graph across steps assumes it.
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gh);
        ggml_cgraph* gh2 = fp->build_decoding_graph(tokens, slots, positions);
        const int n2 = ggml_graph_n_nodes(gh2);
        int diff = -1;
        for (int i = 0; i < (int)seq.size() && i < n2; ++i) {
            ggml_tensor* t = ggml_graph_node(gh2, i);
            if (seq[i].first != ggml_op_name(t->op) || seq[i].second != ggml_get_name(t)) { diff = i; break; }
        }
        if ((int)seq.size() != n2) {
            std::printf("\n  node order: UNSTABLE -- node count %d then %d\n", (int)seq.size(), n2);
        } else if (diff >= 0) {
            ggml_tensor* t = ggml_graph_node(gh2, diff);
            std::printf("\n  node order: UNSTABLE -- first divergence at index %d:"
                        " build 1 = %s %s, build 2 = %s %s\n",
                        diff, seq[diff].first.c_str(), seq[diff].second.c_str(),
                        ggml_op_name(t->op), ggml_get_name(t));
        } else {
            std::printf("\n  node order: stable across two consecutive builds (%d nodes)\n", n2);
        }
    }
    // ── Per-region GPU profile of the REAL decode graph (QINF_BD_PROFILE=1) ──
    //
    // Truncation via the scheduler's eval callback: report "need this node"
    // only at the cut, so ggml batches every node up to it into ONE
    // compute_async + one sync, then stops. The graph, its allocation and its
    // node ordering are the real ones — nothing is rebuilt or re-created in
    // isolation. Region cost is the difference between consecutive cuts.
    //
    // Why not sub-graphs rooted at each node, and why not
    // deltanet_substage_breakdown's method: a dependency-cone sub-graph is not
    // always a well-formed graph (attention's mask can drop out of the cone),
    // and the substage probe subtracts a near-no-op baseline, which dumps the
    // whole fixed submit + under-saturation cost onto its first stage. Here
    // every measured prefix is hundreds of nodes, so that fixed cost is the
    // same in both terms of every difference and cancels. Cumulative numbers
    // still carry it; only the deltas are attributable.
    if (std::getenv("QINF_BD_PROFILE")) {
        const int stride = std::getenv("QINF_BD_STRIDE")
                         ? std::atoi(std::getenv("QINF_BD_STRIDE")) : 32;
        const int iters  = std::getenv("QINF_BD_PROF_ITERS")
                         ? std::atoi(std::getenv("QINF_BD_PROF_ITERS")) : 12;

        const std::vector<int32_t>  tk = {(int32_t)1000};
        const std::vector<uint32_t> sl = {0};
        const std::vector<int32_t>  ps = {(int)fp->get_cache_pos(0)};

        static ggml_tensor* g_cut = nullptr;
        auto cb = [](ggml_tensor* t, bool ask, void*) -> bool {
            // ask: "do you need this node's data?" -> true only at the cut, so
            //      everything before it is computed as one batched dispatch.
            // post: returning false breaks the eval loop -> truncation.
            return ask ? (t == g_cut) : false;
        };

        // Rebuild the graph every iteration, exactly as the timing loop above
        // does, and never advance the cache: every rebuild is then identical in
        // shape, the scheduler stays in the state it was allocated for, and the
        // node pointers we cut on always belong to the graph we are computing.
        ggml_cgraph* probe = fp->build_decoding_graph(tk, sl, ps);
        const int n_full = ggml_graph_n_nodes(probe);
        std::fprintf(stderr, "[profile] decode graph: %d nodes\n", n_full);

        // Cut points: a comma-separated list of node *names* (QINF_BD_CUTS) when
        // the regions of interest are structural -- layer boundaries, say --
        // and a fixed stride otherwise. Explicit cuts are what turn "diffuse
        // tax" into "this layer type"; a stride only ever lands where it lands.
        // Names, not indices: a name identifies the same cut in any build of
        // this graph, an index only in the build it was read from.
        std::vector<int> cut_idx;
        if (const char* cuts = std::getenv("QINF_BD_CUTS")) {
            std::string spec(cuts);
            size_t pos = 0;
            while (pos < spec.size()) {
                size_t comma = spec.find(',', pos);
                if (comma == std::string::npos) comma = spec.size();
                const std::string want = spec.substr(pos, comma - pos);
                pos = comma + 1;
                if (want.empty()) continue;
                int found = -1;
                for (int i = 0; i < n_full; ++i) {
                    if (want == ggml_get_name(ggml_graph_node(probe, i))) { found = i; break; }
                }
                if (found < 0) {
                    std::fprintf(stderr,
                        "QINF_BD_CUTS: node name not in decode graph: expected a name from"
                        " QINF_BD_NODES=1, got '%s'\n", want.c_str());
                    return 1;
                }
                cut_idx.push_back(found);
            }
            std::sort(cut_idx.begin(), cut_idx.end());
            cut_idx.erase(std::unique(cut_idx.begin(), cut_idx.end()), cut_idx.end());
            if (cut_idx.empty()) {
                std::fprintf(stderr, "QINF_BD_CUTS: expected at least one node name, got none\n");
                return 1;
            }
            if (cut_idx.back() != n_full - 1) cut_idx.push_back(n_full - 1);
        } else {
            for (int i = stride - 1; i < n_full - 1; i += stride) cut_idx.push_back(i);
            cut_idx.push_back(n_full - 1);
        }

        // Resolve each cut to a NAME as well. The per-iteration graph is a fresh
        // build; the index is only a label, the name is what identifies the cut
        // (and the loop below re-finds it by name if the index ever moves).
        std::vector<std::string> cut_name;
        for (int ci : cut_idx) cut_name.push_back(ggml_get_name(ggml_graph_node(probe, ci)));

        const std::string cut_desc = std::getenv("QINF_BD_CUTS")
                                   ? std::string("explicit cuts")
                                   : "stride " + std::to_string(stride);
        std::printf("\n  per-region GPU profile of the real decode graph"
                    " (%s, %d cuts, %d iters)\n",
                    cut_desc.c_str(), (int)cut_idx.size(), iters);
        std::printf("  %-6s %-18s %10s %10s   %s\n",
                    "upto", "op at cut", "cum ms", "delta ms", "name at cut");
        std::fflush(stdout);

        double prev_ms = 0.0;
        int    prev_i  = 0;
        for (size_t c = 0; c < cut_idx.size(); ++c) {
            const int ci = cut_idx[c];
            const char* op_at_cut = "?";
            const char* name_at_cut = "";
            std::vector<double> samples;
            for (int it = 0; it < iters + 3; ++it) {
                ggml_cgraph* gf2 = fp->build_decoding_graph(tk, sl, ps);
                ggml_backend_sched_reset(sched);
                ggml_backend_sched_alloc_graph(sched, gf2);
                fp->set_decode_inputs(gf2, tk, sl, ps);
                g_cut = ggml_graph_node(gf2, ci);
                if (cut_name[c] != ggml_get_name(g_cut)) {
                    // Node order moved between builds: find the cut by name so
                    // the region boundary stays where it was asked for.
                    g_cut = nullptr;
                    for (int i = 0; i < ggml_graph_n_nodes(gf2); ++i) {
                        ggml_tensor* t = ggml_graph_node(gf2, i);
                        if (cut_name[c] == ggml_get_name(t)) { g_cut = t; break; }
                    }
                    if (!g_cut) {
                        std::fprintf(stderr,
                            "profile: cut node vanished from rebuilt graph: expected '%s',"
                            " got '%s' at index %d\n",
                            cut_name[c].c_str(), ggml_get_name(ggml_graph_node(gf2, ci)), ci);
                        return 1;
                    }
                }
                op_at_cut   = ggml_op_name(g_cut->op);
                name_at_cut = ggml_get_name(g_cut);
                ggml_backend_sched_set_eval_callback(sched, cb, nullptr);
                auto t0 = Clock::now();
                ggml_backend_sched_graph_compute(sched, gf2);
                ggml_backend_sched_synchronize(sched);
                const double d = ms_since(t0);
                ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
                if (it >= 3) samples.push_back(d);
            }
            const double m = med(samples);
            std::printf("  %-6d %-18s %10.3f %10.3f   %s\n",
                        ci, op_at_cut, m, m - prev_ms, name_at_cut);
            std::fflush(stdout);
            prev_ms = m; prev_i = ci + 1;
        }
        g_cut = nullptr;
        ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
        std::fflush(stdout);
    }

    std::printf("\n  phase                 median ms    %% of step\n");
    std::printf("  --------------------- ---------    ---------\n");
    std::printf("  build_decoding_graph  %8.2f    %6.1f%%\n", b, 100*b/total);
    std::printf("  sched alloc (replan)  %8.2f    %6.1f%%\n", a, 100*a/total);
    std::printf("  set_decode_inputs     %8.2f    %6.1f%%\n", s, 100*s/total);
    std::printf("  graph_compute (GPU)   %8.2f    %6.1f%%\n", c, 100*c/total);
    std::printf("  logits readback       %8.2f    %6.1f%%\n", r, 100*r/total);
    std::printf("  --------------------- ---------    ---------\n");
    std::printf("  TOTAL / step          %8.2f    100.0%%\n", total);
    std::printf("\n  CPU-side fixed (build+alloc+set+read) = %.2f ms (%.1f%%)\n",
                cpu_fixed, 100*cpu_fixed/total);
    std::printf("  GPU compute                           = %.2f ms (%.1f%%)\n",
                c, 100*c/total);
    std::printf("\n  Verdict: if CPU-fixed ~ the 20ms intercept, graph-reuse wins.\n"
                "           if GPU compute holds the intercept, it's launch bubbles → fusion.\n");
    return 0;
}
