#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;

// One step's data that typed inputs read to populate their graph tensors.
// Position model: prefill is contiguous (row r -> pos + r). Batched decode
// passes an explicit per-row vector.
struct StepContext {
    ggml_cgraph* gf = nullptr;

    const std::vector<int32_t>*  tokens     = nullptr;
    int                          pos        = 0;        // base pos, contiguous mode
    const std::vector<int32_t>*  positions  = nullptr;  // explicit per-row; null => pos+r
    const std::vector<uint32_t>* slots      = nullptr;  // batched-decode slot ids
    const std::vector<int32_t>*  sparse_ids = nullptr;  // grammar valid set; null/empty => dense

    // P4 (docs/plan-qwen35-vision-impl.md) — M-RoPE image span.
    //
    // > 0 means THIS BATCH IS AN IMAGE CHUNK of `img_grid_w` soft-token
    // columns. Image prefill is chunked (text | image | text), so an image
    // never shares a batch with text and one width describes the whole batch.
    //
    // Under M-RoPE the four position components stop being equal inside an
    // image: t = pos, h = pos + row, w = pos + col, e = 0. Only
    // MRopePositionsInput reads this; scalar-position recipes never set it.
    uint32_t img_grid_w = 0;

    // Number of KV ROWS already written for this sequence before this batch.
    //
    // -1 means "same as pos", which is true for every scalar-position recipe
    // and was true for ALL of them until M-RoPE: one token, one row, one
    // position. An image span breaks that — it writes nx·ny rows but advances
    // the position by only max(nx, ny) — so anything indexing the KV must use
    // this, and anything computing a rotation angle must use `pos`.
    //
    // Conflating the two silently truncates attention: a causal test written as
    // (row_index <= rope_position) masks off every KV row beyond the position,
    // which after an image is most of the image.
    int kv_base = -1;

    // n_tokens == n_batch: the number of query rows this step computes.
    size_t n_rows() const { return tokens ? tokens->size() : 0; }

    int32_t row_pos(size_t r) const {
        return positions ? (*positions)[r] : (pos + static_cast<int32_t>(r));
    }

    // The KV row this batch's row r occupies. Equals row_pos(r) unless an
    // image span has made rows and positions diverge (see kv_base).
    int64_t row_kv(size_t r) const {
        // kv_base < 0 must reproduce the OLD behaviour exactly, which means
        // deferring to row_pos — not to `pos + r`. Batched decode leaves `pos`
        // at 0 and supplies an explicit per-row positions vector, so `pos + r`
        // would silently be wrong for every decode step.
        return kv_base >= 0 ? static_cast<int64_t>(kv_base) + static_cast<int64_t>(r)
                            : static_cast<int64_t>(row_pos(r));
    }
};

// A self-describing input over one (or, via composition, several) graph
// slot(s). It knows how to populate its tensor and whether that tensor is
// still valid versus the previous step.
class GraphInput {
public:
    virtual ~GraphInput() = default;

    // Populate this input's tensor(s) for the given step.
    virtual void set_input(const StepContext& step) = 0;

    // True iff nothing in this input depends on changed since the last
    // set_input. Conservative: default false.
    virtual bool can_reuse(const StepContext& step) const { return false; }

    // Name of the tensor slot this input owns, for fail-loud diagnostics.
    virtual const char* slot_name() const = 0;

protected:
    // Fail-loud tensor lookup
    static ggml_tensor* require_tensor(const StepContext& step,
                                       const char* slot,
                                       int expected_type);
};

// An ordered collection of typed inputs for one graph.
class GraphInputSet {
public:
    void add(std::unique_ptr<GraphInput> input) {
        inputs_.push_back(std::move(input));
    }

    void clear() { inputs_.clear(); }
    bool empty() const { return inputs_.empty(); }

    // True when some input in this set owns the named graph slot. Exists for
    // fail-loud assertions that a required input actually survived to
    // set_input time — see ForwardPassBase::set_prefill_inputs.
    bool has_slot(const char* name) const {
        for (const auto& in : inputs_)
            if (std::strcmp(in->slot_name(), name) == 0) return true;
        return false;
    }

    void set_input(const StepContext& step) const {
        for (const auto& in : inputs_) in->set_input(step);
    }

    // Conservative AND
    bool can_reuse(const StepContext& step) const {
        for (const auto& in : inputs_)
            if (!in->can_reuse(step)) return false;
        return !inputs_.empty();
    }

private:
    std::vector<std::unique_ptr<GraphInput>> inputs_;
};
