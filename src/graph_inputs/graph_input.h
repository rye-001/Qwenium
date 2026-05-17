#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;

// One step's worth of host-side data that typed inputs read to populate their
// graph tensors. Replaces the four overlapping parameter lists (set_inputs,
// set_batched_inputs, the TQ per-layer loop, the prefill sub-graphs) with one.
//
// Position model: prefill is contiguous (row r -> pos + r). Batched decode
// passes an explicit per-row vector. row_pos() folds both so a single
// PositionsInput / AttnMaskInput body serves every recipe.
struct StepContext {
    ggml_cgraph* gf = nullptr;

    const std::vector<int32_t>*  tokens     = nullptr;
    int                          pos        = 0;        // base pos, contiguous mode
    const std::vector<int32_t>*  positions  = nullptr;  // explicit per-row; null => pos+r
    const std::vector<uint32_t>* slots      = nullptr;  // batched-decode slot ids
    const std::vector<int32_t>*  sparse_ids = nullptr;  // grammar valid set; null/empty => dense

    // n_tokens == n_batch: the number of query rows this step computes.
    size_t n_rows() const { return tokens ? tokens->size() : 0; }

    int32_t row_pos(size_t r) const {
        return positions ? (*positions)[r] : (pos + static_cast<int32_t>(r));
    }
};

// A self-describing input over one (or, via composition, several) graph
// slot(s). It knows how to populate its tensor and whether that tensor is
// still valid versus the previous step. Verbose and explicit on purpose: no
// CRTP, no template metaprogramming (CLAUDE.md cleverness rule). Flat — five
// concrete classes, no base-of-base.
class GraphInput {
public:
    virtual ~GraphInput() = default;

    // Populate this input's tensor(s) for the given step. Called after
    // ggml_backend_sched_alloc_graph, before compute.
    virtual void set_input(const StepContext& step) = 0;

    // True iff nothing this input depends on changed since the last
    // set_input. Conservative: default false. Only an input that can
    // *prove* invariance returns true. (Phase 2 implements honest answers;
    // Phase 1 leaves every input at the safe default.)
    virtual bool can_reuse(const StepContext& step) const { return false; }

    // Name of the tensor slot this input owns, for fail-loud diagnostics.
    virtual const char* slot_name() const = 0;

protected:
    // Fail-loud tensor lookup. Throws naming the slot, the expected
    // dtype/shape, and the actual ("absent"), in that order. Strictly
    // stronger than ggml_graph_get_tensor returning null and segfaulting
    // downstream.
    static ggml_tensor* require_tensor(const StepContext& step,
                                       const char* slot,
                                       int expected_type);
};

// An ordered collection of typed inputs for one graph. set_input fans out to
// every member; can_reuse ANDs them (llama.cpp llm_graph_result::can_reuse
// shape). A future HybridMemoryInput is just another GraphInput added here —
// composition, not a unified base (CLAUDE.md KV != RS invariant).
class GraphInputSet {
public:
    void add(std::unique_ptr<GraphInput> input) {
        inputs_.push_back(std::move(input));
    }

    void clear() { inputs_.clear(); }
    bool empty() const { return inputs_.empty(); }

    void set_input(const StepContext& step) const {
        for (const auto& in : inputs_) in->set_input(step);
    }

    // Conservative AND. Phase 1: always false (every member defaults false).
    bool can_reuse(const StepContext& step) const {
        for (const auto& in : inputs_)
            if (!in->can_reuse(step)) return false;
        return !inputs_.empty();
    }

private:
    std::vector<std::unique_ptr<GraphInput>> inputs_;
};
