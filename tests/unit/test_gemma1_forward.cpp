// test_gemma1_forward.cpp — PR G1.5
//
// Integration smoke test for the Gemma 1 model recipe: load a real Gemma-1
// GGUF and verify prefill produces finite logits with the expected shape.
//
// The HF reference-logit comparison lives in PR G1.0 / a separate fixture-
// driven test once `scripts/gen_gemma_ref_logits.py` has been run. This test
// asserts the structural plumbing — load → tokenize → prefill — works
// end-to-end before we tighten to numeric agreement.
//
// Self-skips when the canonical Gemma file is not present.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "../../src/loader/tokenizer.h"
#include "../../src/models/forward_pass_base.h"
#include "../../src/models/model_registry.h"
#include "../../src/core/model.h"

namespace {

std::string find_gemma_path() {
    if (const char* p = std::getenv("GEMMA1_MODEL_PATH")) return p;
    // Default: alongside the project root.
    return "Gemma_2b_it_v1p1.gguf";
}

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

} // namespace

class Gemma1ModelFile : public ::testing::Test {
protected:
    static std::string path_;

    static void SetUpTestSuite() {
        path_ = find_gemma_path();
    }

    void SetUp() override {
        if (!file_exists(path_)) {
            GTEST_SKIP() << "GEMMA1_MODEL_PATH not set and " << path_
                         << " not found — skipping";
        }
    }
};
std::string Gemma1ModelFile::path_;

TEST_F(Gemma1ModelFile, LoadMetadataAndArchitecture) {
    

    Model model;
    model.load_metadata(path_);
    EXPECT_EQ(model.get_metadata().architecture, "gemma");
    EXPECT_EQ(model.get_metadata().tokenizer_type, "llama");
    EXPECT_EQ(model.get_metadata().bos_token_id, 2);
    EXPECT_EQ(model.get_metadata().eos_token_id, 1);
    EXPECT_EQ(model.get_metadata().padding_token_id, 0);
    EXPECT_EQ(model.get_metadata().attention_head_count, 8u);
    EXPECT_EQ(model.get_metadata().attention_head_count_kv, 1u);  // GQA 8:1
    EXPECT_EQ(model.get_metadata().embedding_length, 2048u);
    EXPECT_GT(model.get_metadata().scores.size(), 0u);
}

TEST_F(Gemma1ModelFile, TokenizerRoundTripsSimpleText) {
    

    Model model;
    model.load_metadata(path_);
    Tokenizer tok(&model.get_metadata());

    const std::string text = "Hello world";
    auto ids = tok.encode(text);
    ASSERT_FALSE(ids.empty());
    // Decoding the encoded ids must reproduce the input byte-for-byte after
    // ▁→space substitution. Note: the leading-space heuristic here depends
    // on Gemma's specific vocab — we only require the round-trip identity.
    EXPECT_EQ(tok.decode(ids), text);
}

// Diagnostic: print the tokenization of a representative chat segment so we
// can compare against the HF reference. Until we have ref fixtures from
// PR G1.0, this is the cheapest way to spot tokenizer drift.
TEST_F(Gemma1ModelFile, DiagnoseTokenizationOfChatSegment) {
    Model model;
    model.load_metadata(path_);
    Tokenizer tok(&model.get_metadata());

    auto print_ids = [&](const std::string& label, const std::vector<int32_t>& ids) {
        std::cerr << label << " (" << ids.size() << "): ";
        for (int32_t id : ids) {
            std::cerr << id << "='" << tok.decode(id) << "'  ";
        }
        std::cerr << "\n";
    };

    print_ids("'Who are you?'", tok.encode("Who are you?"));
    print_ids("' Who are you?'", tok.encode(" Who are you?"));
    print_ids("'<start_of_turn>user\\nWho are you?<end_of_turn>'",
              tok.encode("<start_of_turn>user\nWho are you?<end_of_turn>"));

    // Vocab existence probe: does the GGUF actually contain ▁Who, ▁are, ▁user?
    const std::string under = "\xE2\x96\x81";
    auto& meta = model.get_metadata();
    auto find = [&](const std::string& s) -> int {
        for (size_t i = 0; i < meta.id_to_token.size(); ++i) {
            if (meta.id_to_token[i] == s) return static_cast<int>(i);
        }
        return -1;
    };
    auto score_of = [&](int id) -> float {
        if (id < 0 || static_cast<size_t>(id) >= meta.scores.size()) return 0.0f;
        return meta.scores[id];
    };
    // Probe byte tokens and a representative ▁W token to compute
    // expected Viterbi costs.
    for (const std::string& tok : {"\xE2\x96\x81W", "\xE2\x96\x81Wh", "ho", "<0x00>", "<0x41>", "<0x57>"}) {
        int id = find(tok);
        std::cerr << "raw probe '";
        for (char c : tok) {
            if (c >= 32 && c < 127) std::cerr << c;
            else std::cerr << "\\x" << std::hex << (int)(unsigned char)c << std::dec;
        }
        std::cerr << "' → id=" << id;
        if (id >= 0) std::cerr << " score=" << score_of(id);
        std::cerr << "\n";
    }

    for (const std::string& probe : {"<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>", "<pad>", "<unk>"}) {
        int id = find(probe);
        std::cerr << "special probe '" << probe << "' → id=" << id;
        if (id >= 0 && static_cast<size_t>(id) < meta.token_types.size()) {
            std::cerr << " type=" << static_cast<int>(meta.token_types[id]);
        }
        std::cerr << "\n";
    }

    for (const std::string& probe : {"Who", "are", "user", " Who", " are", " user"}) {
        const std::string sp = (probe[0] == ' ') ? (under + probe.substr(1)) : probe;
        int id = find(sp);
        std::cerr << "vocab probe '" << probe << "' (sp form bytes=" << sp.size()
                  << ") → id=" << id;
        if (id >= 0) std::cerr << " score=" << score_of(id);
        std::cerr << "\n";
    }
}

// Dumps the last-token slice of intermediate tensors via a manually-driven
// graph build/compute. Lets us compare our post-embed-scale values against
// the HF reference (which gives norm=203.14, sample[:5]=[10.16,0.82,-3.67,
// 0.44,0.23] for the last token 'is' (id 603) of "The capital of France is").
TEST_F(Gemma1ModelFile, DumpInpLScaledForCompareWithHF) {
    Model model;
    model.load_metadata(path_);
    model.load_tensors();

    Tokenizer tok(&model.get_metadata());
    auto tokens = tok.encode("The capital of France is");
    tokens.insert(tokens.begin(), 2);  // BOS
    std::cerr << "tokens: [";
    for (auto t : tokens) std::cerr << t << " ";
    std::cerr << "]\n";

    register_builtin_models();
    auto fp = create_forward_pass(model, &model.get_metadata(),
                                  /*ctx=*/256, /*bs=*/1, /*kvb=*/0);

    ggml_backend_sched_t sched = model.get_scheduler();
    ggml_backend_sched_reset(sched);
    ggml_cgraph* gf = fp->build_prefill_graph(tokens, 0, 0);
    ggml_backend_sched_alloc_graph(sched, gf);
    fp->set_inputs(gf, tokens, 0);
    ggml_backend_sched_graph_compute(sched, gf);

    auto dump_last_token = [&](const char* name) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        if (!t) { std::cerr << "TENSOR '" << name << "' NOT FOUND\n"; return; }
        const int hidden = t->ne[0];
        const int n_tokens = t->ne[1];
        std::vector<float> buf(static_cast<size_t>(hidden) * n_tokens);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        // Last-token slice = buf[(n_tokens-1)*hidden .. n_tokens*hidden]
        const float* last = buf.data() + (n_tokens - 1) * hidden;
        double sumsq = 0.0;
        double sumabs = 0.0;
        for (int i = 0; i < hidden; ++i) {
            sumsq += static_cast<double>(last[i]) * last[i];
            sumabs += std::abs(last[i]);
        }
        std::cerr << name
                  << " norm=" << std::sqrt(sumsq)
                  << " mean_abs=" << (sumabs / hidden)
                  << " sample[:5]=[" << last[0] << ", " << last[1] << ", "
                  << last[2] << ", " << last[3] << ", " << last[4] << "]\n";
    };

    auto dump_token_at = [&](const char* name, int col) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        if (!t) { std::cerr << "TENSOR '" << name << "' NOT FOUND\n"; return; }
        std::cerr << name << " ne=[" << t->ne[0] << "," << t->ne[1] << ","
                  << t->ne[2] << "," << t->ne[3] << "] nb=[" << t->nb[0]
                  << "," << t->nb[1] << "] type=" << t->type << "\n";
        const int hidden = t->ne[0];
        const int n_tokens = t->ne[1];
        if (col >= n_tokens) return;
        std::vector<float> buf(static_cast<size_t>(hidden) * n_tokens);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        const float* row = buf.data() + col * hidden;
        double sumsq = 0.0;
        for (int i = 0; i < hidden; ++i) sumsq += static_cast<double>(row[i]) * row[i];
        std::cerr << name << "[col=" << col << "] norm=" << std::sqrt(sumsq)
                  << " sample[:5]=[" << row[0] << ", " << row[1] << ", "
                  << row[2] << ", " << row[3] << ", " << row[4] << "]\n";
    };

    // Read blk.0.attn_norm.weight directly to check whether the GGUF
    // stores `weight` (HF expects -1.0 at index 0) or `(1+weight)` (already
    // shifted; would store 0.0 at index 0).
    auto attn_norm_w = model.get_block(0).attn_norm_weight;
    if (attn_norm_w) {
        std::vector<float> w(attn_norm_w->ne[0]);
        ggml_backend_tensor_get(attn_norm_w, w.data(), 0, w.size() * sizeof(float));
        std::cerr << "blk.0.attn_norm.weight[:5] = [" << w[0] << ", " << w[1] << ", "
                  << w[2] << ", " << w[3] << ", " << w[4] << "]\n";
        std::cerr << "  if HF stores w directly, values are around 0; if GGUF stores (1+w), values are around 1.\n";
    }

    dump_last_token("inpL_scaled");
    dump_last_token("cur_normed.0");
    dump_last_token("attn_proj_out.0");
    dump_last_token("ffn_inp.0");
    for (uint32_t i = 0; i < 18; ++i) {
        char n[32]; std::snprintf(n, sizeof(n), "layer_out.%u", i);
        dump_last_token(n);
    }
    dump_last_token("post_layers");
    dump_last_token("final_norm");

    // Direct read of token_embd weight for comparison
    ggml_tensor* embd = ggml_graph_get_tensor(gf, "token_embd.weight");
    if (!embd) {
        // fallback: it's owned by the model not the graph
        embd = model.get_token_embedding_weight();
    }
    if (embd) {
        std::cerr << "token_embd shape = [" << embd->ne[0] << ", " << embd->ne[1] << "] type=" << embd->type << "\n";
        // Read row for token 603 (= 'is') and 2 (= BOS)
        for (int tid : {2, 603, 651}) {
            std::vector<float> row(embd->ne[0]);
            ggml_backend_tensor_get(embd, row.data(),
                                    static_cast<size_t>(tid) * embd->ne[0] * sizeof(float),
                                    row.size() * sizeof(float));
            double sumsq = 0.0;
            for (float v : row) sumsq += static_cast<double>(v) * v;
            std::cerr << "  embed[" << tid << "] norm=" << std::sqrt(sumsq)
                      << " sample[:5]=[" << row[0] << ", " << row[1] << ", "
                      << row[2] << ", " << row[3] << ", " << row[4] << "]\n";
        }
    }
}

// Sanity check: with the canonical "The capital of France is" prompt and
// greedy decoding, the top-1 predicted token should decode to text that
// contains "Paris" (or its leading-space variant). This is a coarse test
// — it doesn't require numeric logit agreement, just that the model is
// pointing at the right ballpark answer. If the forward pass is wired
// correctly, Gemma 1 2B-it answers this trivially.
TEST_F(Gemma1ModelFile, GreedyTop1OnTrivialPromptIsSensible) {
    Model model;
    model.load_metadata(path_);
    model.load_tensors();

    Tokenizer tok(&model.get_metadata());
    auto tokens = tok.encode("The capital of France is");
    tokens.insert(tokens.begin(), 2);  // explicit BOS

    register_builtin_models();
    auto fp = create_forward_pass(model, &model.get_metadata(),
                                  /*context_len=*/256, /*max_batch=*/1, /*kvb=*/0);

    std::vector<float> logits = fp->run_prefill(
        tokens, /*pos=*/0, /*slot_idx=*/0, model.get_scheduler());

    const size_t vocab = model.get_metadata().vocab_size;
    ASSERT_GE(logits.size(), vocab);

    std::vector<float> last(logits.end() - vocab, logits.end());

    // Print top-10 predictions so we can see if "Paris" is anywhere.
    std::vector<int> idx(last.size());
    for (int i = 0; i < (int)last.size(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(),
        [&](int a, int b) { return last[a] > last[b]; });
    std::cerr << "top-10 predictions for 'The capital of France is':\n";
    for (int k = 0; k < 10; ++k) {
        std::cerr << "  " << k << ": id=" << idx[k]
                  << " logit=" << last[idx[k]]
                  << " text='" << tok.decode(idx[k]) << "'\n";
    }
    int top1 = idx[0];
    std::string top1_text = tok.decode(top1);

    // Loose assertion: anything mentioning Paris counts.
    EXPECT_NE(top1_text.find("Paris"), std::string::npos)
        << "model didn't predict Paris for the trivial prompt — logits look wrong";
}

TEST_F(Gemma1ModelFile, PrefillProducesFiniteLogits) {
    

    Model model;
    model.load_metadata(path_);
    model.load_tensors();

    Tokenizer tok(&model.get_metadata());
    auto tokens = tok.encode("Hello");
    ASSERT_FALSE(tokens.empty());
    // Add Gemma's BOS (id=2) at the start, mirroring HF's add_bos_token=true.
    tokens.insert(tokens.begin(), 2);

    register_builtin_models();
    auto fp = create_forward_pass(model, &model.get_metadata(),
                                  /*context_len=*/256, /*max_batch=*/1, /*kvb=*/0);
    ASSERT_NE(fp, nullptr);

    std::vector<float> logits = fp->run_prefill(
        tokens, /*pos=*/0, /*slot_idx=*/0, model.get_scheduler());

    const size_t vocab = model.get_metadata().vocab_size;
    ASSERT_GE(logits.size(), vocab);

    // Last-token logits — the slice the sampler would consume.
    std::vector<float> last(logits.end() - vocab, logits.end());
    bool any_nonzero = false;
    for (float v : last) {
        ASSERT_TRUE(std::isfinite(v)) << "non-finite logit";
        if (v != 0.0f) any_nonzero = true;
    }
    EXPECT_TRUE(any_nonzero) << "all logits zero — graph likely disconnected";
}
