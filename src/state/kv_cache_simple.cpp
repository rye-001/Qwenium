#include "kv_cache_simple.h"

#include <stdexcept>
#include <string>

simple_kv_cache::simple_kv_cache(
        uint32_t n_layers,
        uint32_t n_ctx_max,
        uint32_t n_batch_max,
        uint32_t n_embd_k,
        uint32_t n_embd_v,
        ggml_type type_k,
        ggml_type type_v,
        ggml_backend_t backend) :
    n_layers(n_layers),
    n_ctx_max(n_ctx_max),
    n_batch_max(n_batch_max),
    n_embd_k(n_embd_k),
    n_embd_v(n_embd_v),
    type_k(type_k),
    type_v(type_v),
    backend(backend),
    buf(nullptr, ggml_backend_buffer_free),
    ctx(nullptr, ggml_free) {

    positions.resize(n_batch_max, 0);
    init_cache();
}

// reference-mode constructor ─────────────────────────────────────────
simple_kv_cache::simple_kv_cache(
        uint32_t n_layers,
        uint32_t n_ctx_max,
        uint32_t n_batch_max,
        uint32_t n_embd_k,
        uint32_t n_embd_v,
        ggml_type type_k,
        ggml_type type_v,
        ggml_backend_t backend,
        const std::vector<KvSharingSpec>& sharing) :
    n_layers(n_layers),
    n_ctx_max(n_ctx_max),
    n_batch_max(n_batch_max),
    n_embd_k(n_embd_k),
    n_embd_v(n_embd_v),
    type_k(type_k),
    type_v(type_v),
    backend(backend),
    buf(nullptr, ggml_backend_buffer_free),
    ctx(nullptr, ggml_free),
    sharing_(sharing) {

    if (!sharing_.empty() && sharing_.size() != n_layers) {
        throw std::runtime_error(
            "simple_kv_cache: field \"sharing.size\" expected " +
            std::to_string(n_layers) + ", got " +
            std::to_string(sharing_.size()));
    }

    // Fail-loud: every reference must point to an in-range, owning layer.
    // Catching the malformed spec at construction beats discovering it at
    // graph-build time when k_cache[il] would silently be nullptr.
    for (size_t il = 0; il < sharing_.size(); ++il) {
        if (!sharing_[il].is_reference) continue;
        const int src = sharing_[il].source_layer;
        if (src < 0 || src >= static_cast<int>(il)) {
            throw std::runtime_error(
                "simple_kv_cache: layer " + std::to_string(il) +
                " field \"source_layer\" expected in [0, " +
                std::to_string(il) + "), got " + std::to_string(src));
        }
        if (sharing_[src].is_reference) {
            throw std::runtime_error(
                "simple_kv_cache: layer " + std::to_string(il) +
                " references layer " + std::to_string(src) +
                " which is itself a reference; expected an owning layer "
                "(transitive aliasing not supported)");
        }
    }

    positions.resize(n_batch_max, 0);
    init_cache();
}

void simple_kv_cache::init_cache() {
    // Calculate memory for tensor metadata, including headroom for views
    const size_t ctx_size = (n_layers * 2 + 512) * ggml_tensor_overhead();

    // Create context with no_alloc=true
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,  // Don't allocate data
    };
    ctx.reset(ggml_init(params));

    // Create tensors (metadata only, no data allocated yet).
    // reference layers do NOT allocate their own tensors — their
    // k_cache[il] / v_cache[il] are bound to the source layer's tensors
    // after the buffer is allocated below.  Skipping allocation here is
    // what makes reference mode "free" memory-wise.
    k_cache.resize(n_layers);
    v_cache.resize(n_layers);

    for (uint32_t il = 0; il < n_layers; ++il) {
        const bool is_ref = !sharing_.empty() && sharing_[il].is_reference;
        if (is_ref) continue;  // bind below
        k_cache[il] = ggml_new_tensor_3d(ctx.get(), type_k, n_embd_k, n_ctx_max, n_batch_max);
        v_cache[il] = ggml_new_tensor_3d(ctx.get(), type_v, n_embd_v, n_ctx_max, n_batch_max);
    }

    // Choose buffer type based on backend availability
    // If Metal backend provided, use it; otherwise fall back to CPU
    ggml_backend_buffer_type_t buffer_type;
    if (backend) {
        buffer_type = ggml_backend_get_default_buffer_type(backend);
    } else {
        buffer_type = ggml_backend_cpu_buffer_type();
    }

    // Allocate buffer and assign all tensors to it
    buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buffer_type));

    // now that the source layers' tensors are backed by real storage,
    // alias every reference layer's slot to its source.  Reads through
    // get_k / get_v on the reference layer hit the same backing memory the
    // source's writes go to, so cross-layer KV sharing is "free" at runtime.
    for (uint32_t il = 0; il < sharing_.size(); ++il) {
        if (!sharing_[il].is_reference) continue;
        const int src = sharing_[il].source_layer;
        k_cache[il] = k_cache[src];
        v_cache[il] = v_cache[src];
    }




    // DEBUG: Verify cache buffer backend
    if (buf) {
        ggml_backend_buffer_type_t buf_type = ggml_backend_buffer_get_type(buf.get());
        const char* buf_name = ggml_backend_buft_name(buf_type);
        printf("KV cache allocated on: %s\n", buf_name);
        printf("KV cache size: %.2f MB\n", 
               ggml_backend_buffer_get_size(buf.get()) / (1024.0 * 1024.0));
    } else {
        printf("ERROR: KV cache buffer is NULL!\n");
    }

}

size_t simple_kv_cache::memory_bytes() const {
    return buf ? ggml_backend_buffer_get_size(buf.get()) : 0;
}

ggml_tensor * simple_kv_cache::get_k(ggml_context * ctx_compute, int32_t il, uint32_t n_kv, uint32_t slot_idx) {
    return ggml_view_2d(ctx_compute, 
        k_cache[il],
        n_embd_k, 
        n_kv,
        k_cache[il]->nb[1],
        slot_idx * k_cache[il]->nb[2]);
}

ggml_tensor * simple_kv_cache::get_v(ggml_context * ctx_compute, int32_t il, uint32_t n_kv, uint32_t slot_idx) {
    return ggml_view_2d(ctx_compute, 
        v_cache[il],
        n_embd_v, 
        n_kv,
        v_cache[il]->nb[1],
        slot_idx * v_cache[il]->nb[2]);
}

ggml_tensor * simple_kv_cache::cpy_k(ggml_context * ctx_compute, ggml_tensor * k_cur, int32_t il, uint32_t slot_idx) {
    const uint32_t n_tokens = k_cur->ne[2];

    // Create view at current position: [n_embd_k, n_tokens]
    ggml_tensor * k_dst = ggml_view_2d(ctx_compute, k_cache[il],
        n_embd_k, n_tokens,
        k_cache[il]->nb[1],
        slot_idx * k_cache[il]->nb[2] + positions[slot_idx] * k_cache[il]->nb[1]);

    return ggml_cpy(ctx_compute, k_cur, k_dst);
}

ggml_tensor * simple_kv_cache::cpy_v(ggml_context * ctx_compute, ggml_tensor * v_cur, int32_t il, uint32_t slot_idx) {
    const uint32_t n_tokens = v_cur->ne[2];

    // Create view at current position: [n_embd_v, n_tokens]
    ggml_tensor * v_dst = ggml_view_2d(ctx_compute, v_cache[il],
        n_embd_v, n_tokens,
        v_cache[il]->nb[1],
        slot_idx * v_cache[il]->nb[2] + positions[slot_idx] * v_cache[il]->nb[1]);

    return ggml_cpy(ctx_compute, v_cur, v_dst);
}

void simple_kv_cache::advance(uint32_t n_tokens, uint32_t slot_idx) {
    positions[slot_idx] += n_tokens;
    GGML_ASSERT(positions[slot_idx] <= n_ctx_max);
}

void simple_kv_cache::clear_slot(uint32_t slot_idx) {
    positions[slot_idx] = 0;
}

void simple_kv_cache::clear_all() {
    std::fill(positions.begin(), positions.end(), 0);
}

void simple_kv_cache::set_pos(uint32_t p, uint32_t slot_idx) {
    GGML_ASSERT(p <= n_ctx_max);
    positions[slot_idx] = p;
}

void simple_kv_cache::compact(uint32_t slot_idx,
                              const std::vector<uint32_t>& retained_positions) {
    GGML_ASSERT(slot_idx < n_batch_max);

    const uint32_t n_retained = retained_positions.size();
    if (n_retained == 0) {
        positions[slot_idx] = 0;
        return;
    }

    // Read retained rows into a temp buffer, then write them back contiguously.
    // Each row = n_embd floats (we use element-size-agnostic byte copies via
    // ggml_backend_tensor_get/set which handle any type).
    const size_t k_row_bytes = n_embd_k * ggml_type_size(type_k);
    const size_t v_row_bytes = n_embd_v * ggml_type_size(type_v);
    const size_t max_row_bytes = std::max(k_row_bytes, v_row_bytes);

    std::vector<uint8_t> tmp(n_retained * max_row_bytes);

    for (uint32_t il = 0; il < n_layers; ++il) {
        const size_t slot_offset_k = slot_idx * k_cache[il]->nb[2];
        const size_t slot_offset_v = slot_idx * v_cache[il]->nb[2];

        // Compact K
        for (uint32_t i = 0; i < n_retained; ++i) {
            size_t src_off = slot_offset_k + retained_positions[i] * k_cache[il]->nb[1];
            ggml_backend_tensor_get(k_cache[il], tmp.data() + i * k_row_bytes,
                                    src_off, k_row_bytes);
        }
        for (uint32_t i = 0; i < n_retained; ++i) {
            size_t dst_off = slot_offset_k + i * k_cache[il]->nb[1];
            ggml_backend_tensor_set(k_cache[il], tmp.data() + i * k_row_bytes,
                                    dst_off, k_row_bytes);
        }

        // Compact V
        for (uint32_t i = 0; i < n_retained; ++i) {
            size_t src_off = slot_offset_v + retained_positions[i] * v_cache[il]->nb[1];
            ggml_backend_tensor_get(v_cache[il], tmp.data() + i * v_row_bytes,
                                    src_off, v_row_bytes);
        }
        for (uint32_t i = 0; i < n_retained; ++i) {
            size_t dst_off = slot_offset_v + i * v_cache[il]->nb[1];
            ggml_backend_tensor_set(v_cache[il], tmp.data() + i * v_row_bytes,
                                    dst_off, v_row_bytes);
        }
    }

    positions[slot_idx] = n_retained;
}

void simple_kv_cache::clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) {
    GGML_ASSERT(src_slot < n_batch_max);
    GGML_ASSERT(dst_slot < n_batch_max);
    GGML_ASSERT(n_tokens <= n_ctx_max);

    // Scratch context for temporary views - freed at end of function
    std::vector<uint8_t> scratch_buf(1024 * 1024);
    struct ggml_init_params params = {
        .mem_size   = scratch_buf.size(),
        .mem_buffer = scratch_buf.data(),
        .no_alloc   = true,
    };
    ggml_context* scratch_ctx = ggml_init(params);

    for (uint32_t il = 0; il < n_layers; ++il) {
        ggml_tensor * k_src = ggml_view_2d(scratch_ctx, k_cache[il], n_embd_k, n_tokens, k_cache[il]->nb[1], src_slot * k_cache[il]->nb[2]);
        ggml_tensor * k_dst = ggml_view_2d(scratch_ctx, k_cache[il], n_embd_k, n_tokens, k_cache[il]->nb[1], dst_slot * k_cache[il]->nb[2]);
        
        k_src->buffer = k_cache[il]->buffer;
        k_dst->buffer = k_cache[il]->buffer;

        ggml_tensor * v_src = ggml_view_2d(scratch_ctx, v_cache[il], n_embd_v, n_tokens, v_cache[il]->nb[1], src_slot * v_cache[il]->nb[2]);
        ggml_tensor * v_dst = ggml_view_2d(scratch_ctx, v_cache[il], n_embd_v, n_tokens, v_cache[il]->nb[1], dst_slot * v_cache[il]->nb[2]);

        v_src->buffer = v_cache[il]->buffer;
        v_dst->buffer = v_cache[il]->buffer;

        ggml_backend_tensor_copy(k_src, k_dst);
        ggml_backend_tensor_copy(v_src, v_dst);
    }
    
    ggml_free(scratch_ctx);
    positions[dst_slot] = n_tokens;
}

ggml_tensor* simple_kv_cache::gather_k(ggml_context* ctx_compute, ggml_cgraph* gf, int32_t il, ggml_tensor* indices, uint32_t n_active, uint32_t n_kv) {
    // 1. Reshape Cache to Flat [n_embd, n_ctx_max * n_batch_max]
    int32_t flat_size = n_ctx_max * n_batch_max;
    ggml_tensor* flat_cache = ggml_reshape_2d(ctx_compute, k_cache[il], n_embd_k, flat_size);

    // 2. Gather
    // src[ne0, ne1], indices[n] -> dst[ne0, n]
    ggml_tensor* gathered_flat = ggml_get_rows(ctx_compute, flat_cache, indices);

    // 3. Reshape to [n_embd, n_kv, n_active]
    ggml_tensor* dst = ggml_reshape_3d(ctx_compute, gathered_flat, n_embd_k, n_kv, n_active);
    
    return dst;
}

ggml_tensor* simple_kv_cache::gather_v(ggml_context* ctx_compute, ggml_cgraph* gf, int32_t il, ggml_tensor* indices, uint32_t n_active, uint32_t n_kv) {
    int32_t flat_size = n_ctx_max * n_batch_max;
    ggml_tensor* flat_cache = ggml_reshape_2d(ctx_compute, v_cache[il], n_embd_v, flat_size);

    ggml_tensor* gathered_flat = ggml_get_rows(ctx_compute, flat_cache, indices);

    ggml_tensor* dst = ggml_reshape_3d(ctx_compute, gathered_flat, n_embd_v, n_kv, n_active);
    
    return dst;
}
