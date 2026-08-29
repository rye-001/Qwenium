#pragma once
// graph_arena.h — the per-forward-pass ggml context and its metadata buffer.
//
// Responsibility: own the two things every recipe needs to build a graph and
//   nothing else — a fixed metadata buffer, and the ggml_context re-initialized
//   over it once per forward pass. One responsibility, so it is a value a recipe
//   HOLDS rather than a base it inherits.
// Public surface:
//   ctx()        — the live context; graph-building code passes this to ggml
//   new_graph()  — an empty cgraph in that context
//   reset()      — free and re-init the context over the same buffer, which is
//                  what makes a forward pass repeatable without reallocating
// State owned: the buffer (heap, allocated once) and the context (freed and
//   recreated by reset(), freed in the destructor).
// Invariants:
//   - The buffer is allocated ONCE and reused forever; reset() re-inits over it
//     rather than reallocating, so steady-state decode does no heap churn.
//   - no_alloc = true: this context holds graph METADATA only. Tensor data lives
//     in backend buffers the scheduler allocates. Never assume ctx() memory
//     holds numbers.
//   - Every ggml_tensor* built from ctx() dies at the next reset(). Anything a
//     caller retains across a reset — a cached graph, a saved node pointer — is
//     dangling. This is exactly why DecodeGraphCache runs on its own scheduler
//     and invalidates when the recipe's context is reset.
//   - Non-copyable, non-movable: the raw ggml_context* has single ownership and
//     the buffer address is baked into it, so moving would dangle.
//
// Why this exists as a type: it is the first extraction out of ForwardPassBase
//   toward composition-over-inheritance (docs/modular-layer-architecture.md;
//   architecture.md §12). Context lifetime was previously entangled with the
//   base class, which meant it could not be reasoned about — or tested —
//   without constructing a whole model.
//
// Unit test: tests/unit/test_graph_arena.cpp

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ggml.h"

// Graph metadata buffer size, and the node capacity of one graph. Both were
// ForwardPassBase constants; they describe the arena, so they live here.
constexpr size_t FP_GRAPH_SIZE_METADATA = 128 * 1024 * 1024;
constexpr size_t FP_GRAPH_SIZE          = 16384;

class GraphArena {
public:
    GraphArena() {
        buffer_.resize(FP_GRAPH_SIZE_METADATA);
        init();
    }

    ~GraphArena() {
        if (ctx_) ggml_free(ctx_);
    }

    GraphArena(const GraphArena&)            = delete;
    GraphArena& operator=(const GraphArena&) = delete;
    GraphArena(GraphArena&&)                 = delete;
    GraphArena& operator=(GraphArena&&)      = delete;

    ggml_context* ctx() const { return ctx_; }

    // Free and re-init over the SAME buffer. Every tensor built since the last
    // reset is invalid afterwards.
    void reset() {
        if (ctx_) ggml_free(ctx_);
        init();
    }

    ggml_cgraph* new_graph() {
        return ggml_new_graph_custom(ctx_, FP_GRAPH_SIZE, false);
    }

    size_t buffer_bytes() const { return buffer_.size(); }

private:
    void init() {
        struct ggml_init_params params = {
            .mem_size   = buffer_.size(),
            .mem_buffer = buffer_.data(),
            .no_alloc   = true,
        };
        ctx_ = ggml_init(params);
        if (!ctx_)
            throw std::runtime_error(
                "GraphArena: slot \"ggml_init\" expected a context, actual: "
                "nullptr (metadata buffer " + std::to_string(buffer_.size()) +
                " bytes)");
    }

    std::vector<uint8_t> buffer_;
    ggml_context*        ctx_ = nullptr;
};
