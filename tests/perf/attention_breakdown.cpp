// Phase 4 investigation — total per-layer Qwen 3.6 gated-attention cost on
// the decode path (n_batch = 1), swept over a few KV-history depths.
//
// Why: the MoE breakdown showed the fusable MoE-side dispatch is too small
// for a 3× target (~1.13× ceiling). Before picking the next Phase 4 PR, we
// need to know whether attention or DeltaNet has more reachable speedup.
// This binary supplies the attention number.
//
// Method: coarse two-point timing per n_kv_len. Stage 0 builds an empty
// scheduled graph (dup of input) — isolates per-iter scheduler launch
// overhead. Stage 1 calls production `build_gated_batched_attention`
// (src/layers/attention.cpp) for one layer at decode shape. Per-layer cost =
// median(Stage 1) − median(Stage 0). No sub-stage attribution: if attention
// ranks high in the budget table, the follow-up is a sub-stage breakdown
// that cuts the production graph at known stage boundaries.
//
// We deliberately do NOT replicate the production graph inline. The earlier
// inline version (now removed) silently diverged from production on permute
// argument semantics and KV cache dtype, both of which produced shape /
// allocator failures. Calling the production function eliminates that class
// of bug.
//
// Usage:
//   QWEN36_MODEL_PATH=/path/to/qwen36.gguf \
//   ATTN_KV_LENS=128,512,2048 \
//   ./bin/attention-breakdown > tests/fixtures/attention_breakdown.json
//
// Env knobs: ATTN_WARMUP (default 10), ATTN_ITERS (default 200),
// ATTN_LAYER (default 3 — first attention layer in qwen35moe layout),
// ATTN_KV_LENS (CSV, default "512").

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ggml.h"
#include "ggml-backend.h"

#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/state/kv_cache_simple.h"
#include "../../src/layers/attention.h"

#include "breakdown_harness.h"

using json = nlohmann::json;
using breakdown::BuiltGraph;
using breakdown::make_graph;
using breakdown::time_subgraph;

struct AttnHparams {
    int   n_embd;
    int   n_head;
    int   n_head_kv;
    int   n_embd_head;
    int   n_rot;
    float freq_base;
    int   context_len;
    float rms_norm_eps;
};

static AttnHparams extract_hparams(const ModelMetadata& meta) {
    AttnHparams hp{};
    hp.n_embd       = static_cast<int>(meta.embedding_length);
    hp.n_head       = static_cast<int>(meta.attention_head_count);
    hp.n_head_kv    = static_cast<int>(meta.attention_head_count_kv);
    // Qwen 3.6: attention_key_length=256 with n_embd=2048, n_head=16 — head_dim
    // is decoupled from n_embd/n_head. Always read attention_key_length.
    hp.n_embd_head  = static_cast<int>(meta.attention_key_length);
    if (hp.n_embd_head == 0 && hp.n_head > 0) hp.n_embd_head = hp.n_embd / hp.n_head;
    hp.freq_base    = meta.rope_freq_base;
    hp.context_len  = static_cast<int>(meta.context_length);
    hp.rms_norm_eps = meta.rms_norm_eps;
    hp.n_rot        = hp.n_embd_head;  // sane default; partial RoPE overrides.
    try {
        hp.n_rot = static_cast<int>(meta.raw_kv.get_uint32("qwen35moe.rope.dimension_count"));
    } catch (...) {}
    return hp;
}

static std::vector<uint32_t> parse_kv_lens(const char* env_csv) {
    std::vector<uint32_t> out;
    std::string s = env_csv ? env_csv : "512";
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        out.push_back(static_cast<uint32_t>(std::atoi(tok.c_str())));
    }
    if (out.empty()) out.push_back(512);
    return out;
}

int main(int /*argc*/, char** /*argv*/) {
    const char* qwen36_path = std::getenv("QWEN36_MODEL_PATH");
    const char* warmup_env  = std::getenv("ATTN_WARMUP");
    const char* iters_env   = std::getenv("ATTN_ITERS");
    const char* layer_env   = std::getenv("ATTN_LAYER");
    const char* kv_env      = std::getenv("ATTN_KV_LENS");

    const int warmup = warmup_env ? std::atoi(warmup_env) : 10;
    const int iters  = iters_env  ? std::atoi(iters_env)  : 200;
    const int layer  = layer_env  ? std::atoi(layer_env)  : 3;
    auto kv_lens     = parse_kv_lens(kv_env);

    if (!qwen36_path) {
        std::cerr << "attention_breakdown: QWEN36_MODEL_PATH not set\n";
        return 1;
    }

    register_builtin_models();
    Model model;
    model.load_metadata(qwen36_path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    const auto& blk  = model.get_block(layer);

    if (!blk.attn_q_weight || !blk.attn_k_weight || !blk.attn_v_weight ||
        !blk.attn_output_weight || !blk.attn_q_norm_weight || !blk.attn_k_norm_weight) {
        std::cerr << "attention_breakdown: layer " << layer
                  << " is not a (gated) attention layer — try ATTN_LAYER=3,7,11,...\n";
        return 1;
    }

    const AttnHparams hp = extract_hparams(meta);

    // Standalone F32 KV cache to match production qwen36 (qwen36.cpp:134).
    uint32_t max_kv = 0;
    for (auto k : kv_lens) max_kv = std::max(max_kv, k);
    const int n_embd_k = hp.n_head_kv * hp.n_embd_head;
    const int n_embd_v = hp.n_head_kv * hp.n_embd_head;

    ggml_backend_t backend = const_cast<ggml_backend_t>(model.get_backend_metal());
    if (!backend) backend = const_cast<ggml_backend_t>(model.get_backend_cpu());

    auto kv = std::make_unique<simple_kv_cache>(
        /*n_layers=*/1, /*n_ctx_max=*/max_kv,
        /*n_batch_max=*/1,
        static_cast<uint32_t>(n_embd_k), static_cast<uint32_t>(n_embd_v),
        GGML_TYPE_F32, GGML_TYPE_F32, backend);

    std::cerr << "  layer=" << layer
              << " n_embd=" << hp.n_embd
              << " n_head=" << hp.n_head
              << " n_head_kv=" << hp.n_head_kv
              << " n_embd_head=" << hp.n_embd_head
              << " n_rot=" << hp.n_rot
              << " kv_lens=[";
    for (size_t i = 0; i < kv_lens.size(); ++i)
        std::cerr << (i ? "," : "") << kv_lens[i];
    std::cerr << "]\n";

    ggml_backend_sched_t sched = model.get_scheduler();

    // We construct `cur` via ggml_get_rows on the model's token embedding so
    // it lives on the same backend as the weights (Metal). A standalone
    // ggml_new_tensor_2d + ggml_set_input would end up CPU-side and force the
    // scheduler into a backend-swap that triggered backend_ids_changed
    // re-reservation and a downstream Metal mul_mat assert.
    ggml_tensor* tok_embd_w = model.get_token_embedding_weight();
    if (!tok_embd_w) {
        std::cerr << "attention_breakdown: token embedding weight missing\n";
        return 1;
    }
    std::vector<int32_t> tok_data{ 0 };  // arbitrary token id; we don't care about numerics

    json sweeps = json::array();

    for (uint32_t n_kv_len : kv_lens) {
        // Mask + gather inputs (constant across iterations once we set them).
        std::vector<int32_t> pos_data{ static_cast<int32_t>(n_kv_len) - 1 };
        std::vector<float>   mask_data(n_kv_len, 0.0f);   // unmask all of [0, pos]
        std::vector<int32_t> gather_data(n_kv_len);
        for (uint32_t j = 0; j < n_kv_len; ++j)
            gather_data[j] = static_cast<int32_t>(j);    // slot=0, positions 0..n_kv-1

        // ── Stage 0: baseline (per-iter scheduler floor) ────────────────────
        auto build_baseline = [&]() {
            BuiltGraph bg = make_graph(/*meta_bytes=*/16 * 1024 * 1024,
                                       /*max_nodes=*/4096);
            ggml_tensor* tokens = ggml_new_tensor_1d(bg.ctx, GGML_TYPE_I32, 1);
            ggml_set_input(tokens);
            ggml_set_name(tokens, "tokens");
            bg.input = ggml_get_rows(bg.ctx, tok_embd_w, tokens);
            ggml_set_name(bg.input, "cur");
            bg.output = ggml_dup(bg.ctx, bg.input);
            ggml_build_forward_expand(bg.gf, bg.output);
            return bg;
        };

        // ── Stage 1: full gated attention via production code ──────────────
        std::vector<uint32_t> slots{ 0u };
        std::vector<int32_t>  positions{ static_cast<int32_t>(n_kv_len) - 1 };

        auto build_full = [&]() {
            BuiltGraph bg = make_graph();
            ggml_tensor* tokens = ggml_new_tensor_1d(bg.ctx, GGML_TYPE_I32, 1);
            ggml_set_input(tokens);
            ggml_set_name(tokens, "tokens");
            bg.input = ggml_get_rows(bg.ctx, tok_embd_w, tokens);
            ggml_set_name(bg.input, "cur");

            ggml_tensor* inp_pos = ggml_new_tensor_1d(bg.ctx, GGML_TYPE_I32, 1);
            ggml_set_input(inp_pos);
            ggml_set_name(inp_pos, "inp_pos");

            ggml_tensor* kq_mask = ggml_new_tensor_4d(bg.ctx, GGML_TYPE_F32,
                                                      n_kv_len, 1, 1, 1);
            ggml_set_input(kq_mask);
            ggml_set_name(kq_mask, "kq_mask");

            ggml_tensor* gather_indices = ggml_new_tensor_1d(bg.ctx, GGML_TYPE_I32,
                                                              static_cast<int64_t>(n_kv_len));
            ggml_set_input(gather_indices);
            ggml_set_name(gather_indices, "gather_indices");

            bg.output = build_gated_batched_attention(
                bg.ctx, bg.gf, kv.get(),
                bg.input, inp_pos,
                kq_mask, gather_indices,
                /*kv_cache_layer=*/0,
                slots, positions,
                /*il=*/0,
                blk.attn_q_weight, blk.attn_q_norm_weight,
                blk.attn_k_weight, blk.attn_k_norm_weight,
                blk.attn_v_weight, blk.attn_output_weight,
                hp.n_embd_head, hp.n_head, hp.n_head_kv,
                hp.n_rot, hp.freq_base, hp.context_len, hp.rms_norm_eps);

            ggml_build_forward_expand(bg.gf, bg.output);
            return bg;
        };

        auto run = [&](const char* name, BuiltGraph& bg) {
            ggml_tensor* tok_t  = ggml_graph_get_tensor(bg.gf, "tokens");
            ggml_tensor* pos_t  = ggml_graph_get_tensor(bg.gf, "inp_pos");
            ggml_tensor* mask_t = ggml_graph_get_tensor(bg.gf, "kq_mask");
            ggml_tensor* gi_t   = ggml_graph_get_tensor(bg.gf, "gather_indices");

            auto upload = [&]() {
                if (tok_t)
                    ggml_backend_tensor_set(tok_t, tok_data.data(), 0,
                        tok_data.size() * sizeof(int32_t));
                if (pos_t)
                    ggml_backend_tensor_set(pos_t, pos_data.data(), 0,
                        pos_data.size() * sizeof(int32_t));
                if (mask_t)
                    ggml_backend_tensor_set(mask_t, mask_data.data(), 0,
                        mask_data.size() * sizeof(float));
                if (gi_t)
                    ggml_backend_tensor_set(gi_t, gather_data.data(), 0,
                        gather_data.size() * sizeof(int32_t));
            };
            return time_subgraph(bg, sched, upload, warmup, iters, name);
        };

        // Run the larger graph first: ggml_backend_sched buffer pools size up
        // on first reserve; running bg0 first leaves kq_mask / gather_indices
        // unaccounted for and the subsequent reserve trips a size_max assert
        // when bg1 brings them in.
        BuiltGraph bg1 = build_full();

        if (std::getenv("ATTN_DUMP_GRAPH")) {
            const int n_nodes = ggml_graph_n_nodes(bg1.gf);
            std::cerr << "  [graph dump, " << n_nodes << " nodes]\n";
            for (int n = 0; n < n_nodes; ++n) {
                ggml_tensor* t = ggml_graph_node(bg1.gf, n);
                std::cerr << "    [" << n << "] op=" << ggml_op_name(t->op)
                          << " name=" << (t->name[0] ? t->name : "(unnamed)")
                          << " ne=[" << t->ne[0] << "," << t->ne[1]
                          << "," << t->ne[2] << "," << t->ne[3] << "]";
                if (t->op == GGML_OP_MUL_MAT) {
                    ggml_tensor* a = t->src[0];
                    ggml_tensor* b = t->src[1];
                    std::cerr << " src0=[" << a->ne[0] << "," << a->ne[1]
                              << "," << a->ne[2] << "," << a->ne[3] << "]"
                              << " src1=[" << b->ne[0] << "," << b->ne[1]
                              << "," << b->ne[2] << "," << b->ne[3] << "]";
                }
                std::cerr << "\n";
            }
        }

        double full_ms = run("attn_decode_full", bg1);
        ggml_free(bg1.ctx);

        BuiltGraph bg0 = build_baseline();
        double base_ms = run("baseline", bg0);
        ggml_free(bg0.ctx);

        const double layer_ms = full_ms - base_ms;

        std::cerr << "    [n_kv=" << n_kv_len << "] baseline " << base_ms
                  << " ms, full " << full_ms << " ms, per-layer " << layer_ms << " ms\n";

        sweeps.push_back({
            {"n_kv_len",        n_kv_len},
            {"baseline_ms",     base_ms},
            {"attn_decode_ms",  full_ms},
            {"per_layer_ms",    layer_ms},
        });
    }

    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&tt));

    json out = {
        {"_note",
            "Phase 4 investigation: coarse total per-layer gated-attention "
            "decode cost. Calls production build_gated_batched_attention "
            "(src/layers/attention.cpp) — no inline replication. per_layer_ms "
            "is full−baseline to cancel scheduler launch overhead."},
        {"generated_at",   ts_buf},
        {"model_path",     qwen36_path},
        {"architecture",   meta.architecture},
        {"layer_idx",      layer},
        {"n_embd",         hp.n_embd},
        {"n_head",         hp.n_head},
        {"n_head_kv",      hp.n_head_kv},
        {"n_embd_head",    hp.n_embd_head},
        {"n_rot",          hp.n_rot},
        {"warmup_iters",   warmup},
        {"timed_iters",    iters},
        {"kv_sweeps",      sweeps},
    };
    std::cout << out.dump(2) << "\n";
    return 0;
}
