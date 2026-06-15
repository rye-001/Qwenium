#include "kv_cache_simple.h"
#include "snapshot_io.h"

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

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

    if (slot_idx >= n_batch_max) {
        throw std::runtime_error(
            "simple_kv_cache::cpy_k: slot_idx (" + std::to_string(slot_idx) +
            ") exceeds n_batch_max (" + std::to_string(n_batch_max) + ")");
    }
    if (positions[slot_idx] + n_tokens > n_ctx_max) {
        throw std::runtime_error(
            "simple_kv_cache::cpy_k: KV cache context overflow on slot " + std::to_string(slot_idx) +
            ": current pos=" + std::to_string(positions[slot_idx]) +
            ", adding " + std::to_string(n_tokens) + " tokens exceeds n_ctx_max=" +
            std::to_string(n_ctx_max));
    }

    // Create view at current position: [n_embd_k, n_tokens]
    ggml_tensor * k_dst = ggml_view_2d(ctx_compute, k_cache[il],
        n_embd_k, n_tokens,
        k_cache[il]->nb[1],
        slot_idx * k_cache[il]->nb[2] + positions[slot_idx] * k_cache[il]->nb[1]);

    return ggml_cpy(ctx_compute, k_cur, k_dst);
}

ggml_tensor * simple_kv_cache::cpy_v(ggml_context * ctx_compute, ggml_tensor * v_cur, int32_t il, uint32_t slot_idx) {
    const uint32_t n_tokens = v_cur->ne[2];

    if (slot_idx >= n_batch_max) {
        throw std::runtime_error(
            "simple_kv_cache::cpy_v: slot_idx (" + std::to_string(slot_idx) +
            ") exceeds n_batch_max (" + std::to_string(n_batch_max) + ")");
    }
    if (positions[slot_idx] + n_tokens > n_ctx_max) {
        throw std::runtime_error(
            "simple_kv_cache::cpy_v: KV cache context overflow on slot " + std::to_string(slot_idx) +
            ": current pos=" + std::to_string(positions[slot_idx]) +
            ", adding " + std::to_string(n_tokens) + " tokens exceeds n_ctx_max=" +
            std::to_string(n_ctx_max));
    }

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

// --- L2 session snapshot (AppendKV lane) -----------------------------------

uint64_t simple_kv_cache::path_tag() const {
    const char* buft = buf
        ? ggml_backend_buft_name(ggml_backend_buffer_get_type(buf.get()))
        : "none";
    const std::string s = std::string(buft) + "|tk" +
        std::to_string(static_cast<int>(type_k)) + "|tv" +
        std::to_string(static_cast<int>(type_v)) + "|ctx" +
        std::to_string(n_ctx_max);
    return static_cast<uint64_t>(std::hash<std::string>{}(s));
}

void simple_kv_cache::serialize_slot(qinf::session::SnapshotWriter& w,
                                     uint32_t slot) const {
    if (slot >= n_batch_max) {
        throw std::runtime_error(
            "kv_cache: serialize slot expected < " + std::to_string(n_batch_max) +
            ", got " + std::to_string(slot));
    }
    const uint32_t N = positions[slot];

    w.put_u32(n_layers);
    w.put_u32(n_embd_k);
    w.put_u32(n_embd_v);
    w.put_u32(static_cast<uint32_t>(type_k));
    w.put_u32(static_cast<uint32_t>(type_v));
    w.put_u32(n_ctx_max);
    w.put_u32(N);

    for (uint32_t il = 0; il < n_layers; ++il) {
        if (is_reference_layer(il)) continue;  // aliased; source carries the data
        ggml_tensor* k = k_cache[il];
        ggml_tensor* v = v_cache[il];
        const size_t k_bytes = static_cast<size_t>(N) * k->nb[1];
        const size_t v_bytes = static_cast<size_t>(N) * v->nb[1];
        std::vector<uint8_t> kbuf(k_bytes), vbuf(v_bytes);
        ggml_backend_tensor_get(k, kbuf.data(),
                                static_cast<size_t>(slot) * k->nb[2], k_bytes);
        ggml_backend_tensor_get(v, vbuf.data(),
                                static_cast<size_t>(slot) * v->nb[2], v_bytes);
        w.put_bytes(kbuf.data(), k_bytes);
        w.put_bytes(vbuf.data(), v_bytes);
    }
}

void simple_kv_cache::deserialize_slot(qinf::session::SnapshotReader& r,
                                       uint32_t slot) {
    if (slot >= n_batch_max) {
        throw std::runtime_error(
            "kv_cache: restore slot expected < " + std::to_string(n_batch_max) +
            ", got " + std::to_string(slot));
    }
    auto require = [](const char* field, uint64_t expected, uint64_t actual) {
        if (expected != actual) {
            throw std::runtime_error(
                std::string("kv_cache: field \"") + field + "\" expected " +
                std::to_string(expected) + ", got " + std::to_string(actual));
        }
    };
    require("n_layers", n_layers, r.get_u32());
    require("n_embd_k", n_embd_k, r.get_u32());
    require("n_embd_v", n_embd_v, r.get_u32());
    require("type_k", static_cast<uint32_t>(type_k), r.get_u32());
    require("type_v", static_cast<uint32_t>(type_v), r.get_u32());
    r.get_u32();  // producer n_ctx_max — not required to match, only N must fit
    const uint32_t N = r.get_u32();
    if (N > n_ctx_max) {
        throw std::runtime_error(
            "kv_cache: restore span length expected <= n_ctx_max " +
            std::to_string(n_ctx_max) + ", got " + std::to_string(N));
    }

    for (uint32_t il = 0; il < n_layers; ++il) {
        if (is_reference_layer(il)) continue;
        ggml_tensor* k = k_cache[il];
        ggml_tensor* v = v_cache[il];
        const size_t k_bytes = static_cast<size_t>(N) * k->nb[1];
        const size_t v_bytes = static_cast<size_t>(N) * v->nb[1];
        std::vector<uint8_t> kbuf, vbuf;
        r.get_bytes(kbuf);
        r.get_bytes(vbuf);
        if (kbuf.size() != k_bytes || vbuf.size() != v_bytes) {
            throw std::runtime_error(
                "kv_cache: layer " + std::to_string(il) +
                " block size expected k=" + std::to_string(k_bytes) + " v=" +
                std::to_string(v_bytes) + ", got k=" + std::to_string(kbuf.size()) +
                " v=" + std::to_string(vbuf.size()));
        }
        ggml_backend_tensor_set(k, kbuf.data(),
                                static_cast<size_t>(slot) * k->nb[2], k_bytes);
        ggml_backend_tensor_set(v, vbuf.data(),
                                static_cast<size_t>(slot) * v->nb[2], v_bytes);
    }
    positions[slot] = N;
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
