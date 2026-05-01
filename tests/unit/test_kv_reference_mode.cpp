// test_kv_reference_mode.cpp — PR G4.4
//
// Synthetic state-isolation tests for simple_kv_cache reference-mode slots.
//
// Reference-mode is the mechanism Gemma 4's "shared KV across layers" feature
// uses: the last `num_kv_shared_layers` layers reuse the K and V tensors from
// a producing layer.  In 26B-A4B `shared_kv_layers == 0`, so the path is
// dormant in production; this test is the gate the dense 31B follow-up will
// promote to a canary.
//
// Tests:
//   1. EmptySharingMatchesPrimary — passing an empty sharing vector to the
//      G4.4 constructor must produce the same per-layer tensor layout as
//      the primary constructor (regression gate for the existing path).
//   2. AliasesSourceTensor       — a reference layer's k_cache / v_cache
//      pointers are *identical* to the source layer's pointers (zero
//      allocation; same backing storage).
//   3. WritesVisibleAcrossAlias  — set bytes through the source layer's
//      tensor; read them back through the reference layer's tensor.  Round
//      trips via the backend tensor get/set API exercise the backing
//      memory, not just pointer equality.
//   4. RejectsForwardReference   — referencing a later layer is rejected at
//      construction with the spec-mandated fail-loud message.
//   5. RejectsTransitiveReference — referencing another reference layer is
//      rejected (sources must be owning layers).
//
// No model file required.
// Build target: kv-reference-mode-tests (see tests/CMakeLists.txt).

#include <gtest/gtest.h>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "../../src/state/kv_cache_simple.h"
#include "../../src/state/layer_state.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

namespace {

constexpr uint32_t N_LAYERS    = 8;
constexpr uint32_t N_CTX_MAX   = 16;
constexpr uint32_t N_BATCH_MAX = 1;
constexpr uint32_t N_EMBD_K    = 4;
constexpr uint32_t N_EMBD_V    = 4;

simple_kv_cache make_owning_cache(ggml_backend_t be) {
    return simple_kv_cache(N_LAYERS, N_CTX_MAX, N_BATCH_MAX,
                            N_EMBD_K, N_EMBD_V,
                            GGML_TYPE_F32, GGML_TYPE_F32, be);
}

simple_kv_cache make_sharing_cache(ggml_backend_t be,
                                   const std::vector<KvSharingSpec>& sharing) {
    return simple_kv_cache(N_LAYERS, N_CTX_MAX, N_BATCH_MAX,
                            N_EMBD_K, N_EMBD_V,
                            GGML_TYPE_F32, GGML_TYPE_F32, be, sharing);
}

} // namespace

// 1. Empty sharing vector ⇒ behavior identical to the primary constructor.
//    Each layer must own a distinct tensor (no accidental aliasing).
TEST(KvReferenceModeTest, EmptySharingMatchesPrimary) {
    ggml_backend_t be = ggml_backend_cpu_init();
    ASSERT_NE(be, nullptr);

    {
        auto cache = make_sharing_cache(be, /*sharing=*/{});
        for (uint32_t i = 0; i < N_LAYERS; ++i) {
            EXPECT_FALSE(cache.is_reference_layer(i))
                << "layer " << i << " misreported as reference under empty sharing";
            EXPECT_NE(cache.get_k_cache_tensor(i), nullptr);
            EXPECT_NE(cache.get_v_cache_tensor(i), nullptr);
            for (uint32_t j = i + 1; j < N_LAYERS; ++j) {
                EXPECT_NE(cache.get_k_cache_tensor(i),
                          cache.get_k_cache_tensor(j))
                    << "layers " << i << " and " << j
                    << " share K storage under empty sharing";
            }
        }
    }

    ggml_backend_free(be);
}

// 2. Reference layer's tensor pointers identically equal the source's.
TEST(KvReferenceModeTest, AliasesSourceTensor) {
    ggml_backend_t be = ggml_backend_cpu_init();
    ASSERT_NE(be, nullptr);

    // Last two layers reference layer 4.
    std::vector<KvSharingSpec> sharing(N_LAYERS);
    sharing[6] = { /*is_reference=*/true,  /*source_layer=*/4 };
    sharing[7] = { /*is_reference=*/true,  /*source_layer=*/4 };

    {
        auto cache = make_sharing_cache(be, sharing);

        EXPECT_TRUE(cache.is_reference_layer(6));
        EXPECT_TRUE(cache.is_reference_layer(7));
        EXPECT_FALSE(cache.is_reference_layer(4));

        // Pointer identity — the load-bearing claim.
        EXPECT_EQ(cache.get_k_cache_tensor(6), cache.get_k_cache_tensor(4));
        EXPECT_EQ(cache.get_v_cache_tensor(6), cache.get_v_cache_tensor(4));
        EXPECT_EQ(cache.get_k_cache_tensor(7), cache.get_k_cache_tensor(4));
        EXPECT_EQ(cache.get_v_cache_tensor(7), cache.get_v_cache_tensor(4));

        // Owning layers other than 4 must remain distinct from layer 4.
        for (uint32_t i : {0u, 1u, 2u, 3u, 5u}) {
            EXPECT_NE(cache.get_k_cache_tensor(i),
                      cache.get_k_cache_tensor(4));
        }

        EXPECT_EQ(cache.reference_source(6), 4);
        EXPECT_EQ(cache.reference_source(7), 4);
        EXPECT_EQ(cache.reference_source(4), -1);
    }

    ggml_backend_free(be);
}

// 3. Bytes written through the source layer's tensor are observable through
//    the reference layer's tensor — the alias points at the same backing
//    memory, not a parallel copy.
TEST(KvReferenceModeTest, WritesVisibleAcrossAlias) {
    ggml_backend_t be = ggml_backend_cpu_init();
    ASSERT_NE(be, nullptr);

    std::vector<KvSharingSpec> sharing(N_LAYERS);
    sharing[6] = { true, 4 };

    {
        auto cache = make_sharing_cache(be, sharing);

        // Write a recognizable pattern into layer 4's K tensor.
        ggml_tensor* k_src = cache.get_k_cache_tensor(4);
        ASSERT_NE(k_src, nullptr);
        const size_t byte_count = ggml_nbytes(k_src);
        ASSERT_GE(byte_count, sizeof(float) * N_EMBD_K);

        std::vector<float> pattern(N_EMBD_K, 0.0f);
        for (uint32_t i = 0; i < N_EMBD_K; ++i) {
            pattern[i] = 1.0f + static_cast<float>(i) * 0.5f;
        }
        // Write only the first row (n_embd_k floats) at offset 0.
        ggml_backend_tensor_set(k_src, pattern.data(), 0,
                                sizeof(float) * N_EMBD_K);

        // Read the first row back through the reference layer's tensor.
        ggml_tensor* k_ref = cache.get_k_cache_tensor(6);
        ASSERT_NE(k_ref, nullptr);
        std::vector<float> readback(N_EMBD_K, 0.0f);
        ggml_backend_tensor_get(k_ref, readback.data(), 0,
                                sizeof(float) * N_EMBD_K);

        for (uint32_t i = 0; i < N_EMBD_K; ++i) {
            EXPECT_FLOAT_EQ(readback[i], pattern[i])
                << "alias readback mismatched source write at index " << i;
        }
    }

    ggml_backend_free(be);
}

// 4. A reference whose source_layer is >= il must be rejected at
//    construction with a fail-loud message naming the field.
TEST(KvReferenceModeTest, RejectsForwardReference) {
    ggml_backend_t be = ggml_backend_cpu_init();
    ASSERT_NE(be, nullptr);

    std::vector<KvSharingSpec> sharing(N_LAYERS);
    sharing[3] = { true, 5 };  // source after self — invalid

    EXPECT_THROW({
        auto _ = make_sharing_cache(be, sharing);
        (void)_;
    }, std::runtime_error);

    ggml_backend_free(be);
}

// 5. Sources must be owning layers.  Transitive references are rejected so
//    the alias resolves in a single hop.
TEST(KvReferenceModeTest, RejectsTransitiveReference) {
    ggml_backend_t be = ggml_backend_cpu_init();
    ASSERT_NE(be, nullptr);

    std::vector<KvSharingSpec> sharing(N_LAYERS);
    sharing[3] = { true, 1 };
    sharing[5] = { true, 3 };  // 3 is itself a reference — invalid

    EXPECT_THROW({
        auto _ = make_sharing_cache(be, sharing);
        (void)_;
    }, std::runtime_error);

    ggml_backend_free(be);
}
