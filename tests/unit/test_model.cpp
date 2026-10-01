/**
 * test_model.cpp — src/engine/model.cpp
 *
 * Covers Model::load_tensors' partial-loading parameter (`max_blocks`), the
 * mechanism behind --lens-verify-only (docs/architecture.md §6,
 * src/server/http_server.cpp, src/server/server_lens.h). A truncated
 * /v1/verify pass never reads a block at or past its cutoff (causality: an
 * attention layer cannot depend on one above it), so loading only
 * [0, max_blocks) is provably safe, not a guess -- these tests pin the two
 * load-bearing properties that guarantee: (1) the omitted blocks really are
 * left unloaded (nullptr, not silently loaded anyway), and (2) the ONE thing
 * that must NOT change when fewer blocks load: weights_hash, the identity a
 * lens report's config.weights stamp relies on for cross-server comparison
 * (docs/lens-format.md: "two reports are only comparable when this matches").
 * Hashing the loaded subset instead of the full inventory would make a
 * verify-only report incomparable with a full-server one -- the exact trap
 * this file exists to catch a regression into.
 *
 * Requires: QWEN35_MODEL_PATH env var pointing to Qwen3.5-0.8B-BF16.gguf
 * (same model + convention as test_qwen35_tensor_loading.cpp). Self-skips
 * when absent, per architecture.md's model-file-test convention.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <string>

#include "engine/model.h"
#include "models/model_registry.h"

static std::string get_qwen35_model_path() {
    const char* path = std::getenv("QWEN35_MODEL_PATH");
    return path ? std::string(path) : "";
}

// GGUFLoader::load_model validates general.architecture against the
// registry's allow-list, so it must be populated before ANY Model in this
// file loads metadata. Other test files in the same binary (e.g.
// test_gguf_loader.cpp) happen to do this too and registration is
// idempotent, but a suite filtered to just this file (gtest_filter) must not
// depend on load order across translation units -- register here as well,
// same discipline as test_gguf_loader.cpp / test_decode_graph_cache.cpp.
namespace {
struct RegisterModelsOnce {
    RegisterModelsOnce() { register_builtin_models(); }
} g_register_models_once;
}  // namespace

#define SKIP_IF_NO_MODEL()                                          \
    do {                                                            \
        if (get_qwen35_model_path().empty()) {                     \
            GTEST_SKIP() << "QWEN35_MODEL_PATH not set, skipping"; \
        }                                                          \
    } while (0)

// ============================================================
// Test: weights_hash is identical whether load_tensors loaded every block
// or only a leading subset -- it must come from the FULL parsed metadata,
// never from what actually got copied.
// ============================================================
TEST(ModelPartialLoad, WeightsHashUnaffectedByPartialLoad) {
    SKIP_IF_NO_MODEL();

    Model full;
    full.load_metadata(get_qwen35_model_path());
    full.load_tensors();
    ASSERT_GT(full.get_metadata().block_count, 5u)
        << "test needs a model with more than 5 blocks to exercise a partial load";

    Model partial;
    partial.load_metadata(get_qwen35_model_path());
    partial.load_tensors(/*max_blocks=*/5);

    EXPECT_NE(full.get_metadata().weights_hash, 0u);
    EXPECT_EQ(full.get_metadata().weights_hash, partial.get_metadata().weights_hash)
        << "weights_hash must be computed from the FULL tensor inventory "
           "(GGUFLoader::load_model, before load_tensors ever runs) regardless "
           "of how many blocks actually got copied -- otherwise a "
           "--lens-verify-only report's config.weights would never match a "
           "full server's, breaking the one thing that stamp exists to let a "
           "caller check (docs/lens-format.md).";
}

// ============================================================
// Test: blocks [0, max_blocks) are loaded (every pointer a real tensor);
// blocks [max_blocks, block_count) are left default-constructed (every
// pointer null) -- not partially populated, not silently loaded anyway.
// ============================================================
TEST(ModelPartialLoad, LoadsOnlyRequestedBlocks) {
    SKIP_IF_NO_MODEL();

    Model m;
    m.load_metadata(get_qwen35_model_path());
    const uint32_t needed = 5;
    ASSERT_GT(m.get_metadata().block_count, needed);
    m.load_tensors(needed);

    for (uint32_t i = 0; i < needed; ++i) {
        const auto& blk = m.get_block(i);
        EXPECT_NE(blk.attn_norm_weight, nullptr) << "block " << i << " should be loaded (< needed)";
        EXPECT_NE(blk.ffn_gate_weight, nullptr) << "block " << i << " should be loaded (< needed)";
    }
    for (uint32_t i = needed; i < m.get_metadata().block_count; ++i) {
        const auto& blk = m.get_block(i);
        // qwen35 blocks are either attention- or SSM-typed (never both), so
        // checking one field from each family pins "nothing at all loaded"
        // rather than just "the field this family happens not to use".
        EXPECT_EQ(blk.attn_norm_weight, nullptr) << "block " << i << " should NOT be loaded (>= needed)";
        EXPECT_EQ(blk.attn_q_weight, nullptr)    << "block " << i << " should NOT be loaded (>= needed)";
        EXPECT_EQ(blk.ssm_a, nullptr)            << "block " << i << " should NOT be loaded (>= needed)";
    }
}

// ============================================================
// Test: token_embd.weight is always loaded (every forward pass embeds its
// input regardless of truncation) -- and output_norm.weight/output.weight
// are NOT loaded on a partial request, matching run_lens_verify's
// want_logits=false (verify never decodes, so nothing reads them).
// ============================================================
TEST(ModelPartialLoad, EmbeddingLoadedHeadAndFinalNormAreNot) {
    SKIP_IF_NO_MODEL();

    Model m;
    m.load_metadata(get_qwen35_model_path());
    ASSERT_GT(m.get_metadata().block_count, 3u);
    m.load_tensors(/*max_blocks=*/3);

    EXPECT_NE(m.get_token_embedding_weight(), nullptr)
        << "token_embd.weight must load even on a partial request -- every "
           "forward pass embeds its input tokens regardless of truncation";
    EXPECT_EQ(m.get_output_norm_weight(), nullptr)
        << "output_norm.weight has no reader on a truncated pass "
           "(build_prefill_graph is always called with want_logits=false)";
    EXPECT_EQ(m.get_output_weight(), nullptr)
        << "output.weight has no reader on a truncated pass";
}

// ============================================================
// Test: keep_output_head on a partial load keeps the final norm and the
// output weight (a truncated server that reads logits after its last loaded
// block — --lens-verdict) and still skips every block from max_blocks on.
// ============================================================
TEST(ModelPartialLoad, KeepOutputHeadLoadsHeadAndFinalNorm) {
    SKIP_IF_NO_MODEL();

    Model m;
    m.load_metadata(get_qwen35_model_path());
    ASSERT_GT(m.get_metadata().block_count, 3u);
    m.load_tensors(/*max_blocks=*/3, /*keep_output_head=*/true);

    EXPECT_NE(m.get_token_embedding_weight(), nullptr);
    EXPECT_NE(m.get_output_norm_weight(), nullptr)
        << "keep_output_head expected output_norm.weight on a partial load";
    EXPECT_EQ(m.get_output_weight() != nullptr, m.get_metadata().tensor_inventory.count("output.weight") > 0)
        << "keep_output_head expected output.weight exactly when the file carries one (tied models do not)";
    for (uint32_t i = 3; i < m.get_metadata().block_count; ++i)
        EXPECT_EQ(m.get_block(i).attn_norm_weight, nullptr) << "block " << i << " should NOT be loaded (>= 3)";
}

// ============================================================
// Test: an ordinary full load (no max_blocks argument, i.e. the pre-existing
// call shape) is unaffected -- every block loads, output_norm.weight loads.
// Guards against the partial-load branch leaking into the default path.
// ============================================================
TEST(ModelPartialLoad, DefaultCallLoadsEverything) {
    SKIP_IF_NO_MODEL();

    Model m;
    m.load_metadata(get_qwen35_model_path());
    m.load_tensors();  // no argument -- must behave exactly as before this feature existed

    EXPECT_NE(m.get_output_norm_weight(), nullptr);
    for (uint32_t i = 0; i < m.get_metadata().block_count; ++i) {
        EXPECT_NE(m.get_block(i).attn_norm_weight, nullptr) << "block " << i;
    }
}

// ============================================================
// Test: passing block_count explicitly is the same "everything" path as the
// default argument -- the partial/full split is a threshold, not a
// has-an-argument check.
// ============================================================
TEST(ModelPartialLoad, ExplicitFullBlockCountLoadsEverything) {
    SKIP_IF_NO_MODEL();

    Model m;
    m.load_metadata(get_qwen35_model_path());
    m.load_tensors(m.get_metadata().block_count);

    EXPECT_NE(m.get_output_norm_weight(), nullptr);
    for (uint32_t i = 0; i < m.get_metadata().block_count; ++i) {
        EXPECT_NE(m.get_block(i).attn_norm_weight, nullptr) << "block " << i;
    }
}
