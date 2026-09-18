#include "routing_replay_input.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <stdexcept>
#include <string>
#include <vector>

void RoutingReplayInput::set_input(const StepContext& step) {
    if (!trace_)
        throw std::runtime_error(
            "RoutingReplayInput: slot 'moe_routing': expected a borrowed RoutingTrace, "
            "actual: null — this input must only be registered on the replay path");
    if (trace_->empty())
        throw std::runtime_error(
            "RoutingReplayInput: slot 'moe_routing': expected a non-empty RoutingTrace, "
            "actual: 0 layers recorded");

    const size_t n_rows = step.n_rows();
    const int    top_k  = trace_->top_k();
    std::vector<int32_t> buf;
    int filled = 0;

    // Driven by the TRACE, looked up BY NAME. Two reasons it is not a scan of
    // graph->nodes: a graph input has no sources, so ggml files it under the
    // graph's LEAFS and a node walk finds exactly zero of them (found the hard
    // way — the fail-loud below is what caught it); and the trace is the right
    // authority anyway, because a truncated verify prefill builds FEWER layers
    // than were captured and the missing ones are correct absences, not errors.
    //
    // The contract this leaves: the trace must come from the same recipe at
    // full depth, which is what the capture path produces. A graph layer with
    // no entry in the trace would go unfilled rather than fail here.
    for (const auto& kv : trace_->by_layer()) {
        const std::string name = "moe_routing." + std::to_string(kv.first);
        ggml_tensor* t = ggml_graph_get_tensor(step.gf, name.c_str());
        if (!t) continue;   // this pass did not build that layer

        if (t->type != GGML_TYPE_I32)
            throw std::runtime_error(
                "RoutingReplayInput: slot '" + name + "' expected type I32, actual: " +
                ggml_type_name(t->type));
        if (t->ne[0] != top_k || (size_t)t->ne[1] != n_rows)
            throw std::runtime_error(
                "RoutingReplayInput: slot '" + name + "' expected shape [" +
                std::to_string(top_k) + ", " + std::to_string(n_rows) +
                "] (trace top_k x this batch's rows), actual: [" +
                std::to_string(t->ne[0]) + ", " + std::to_string(t->ne[1]) + "]");

        // Absolute positions, not row indices: prefill may arrive in chunks and
        // decode one row at a time, and the trace is indexed by where a token
        // sits in the sequence, not by where it sits in this batch.
        buf.resize(n_rows * (size_t)top_k);
        for (size_t r = 0; r < n_rows; ++r) {
            const int32_t* sel = trace_->at(kv.first, (size_t)step.row_pos(r));
            std::copy(sel, sel + top_k, buf.begin() + r * (size_t)top_k);
        }
        ggml_backend_tensor_set(t, buf.data(), 0, ggml_nbytes(t));
        filled++;
    }

    if (filled == 0)
        throw std::runtime_error(
            "RoutingReplayInput: slot 'moe_routing': expected at least one "
            "'moe_routing.<il>' tensor in the graph, actual: 0 — the graph was built "
            "with RoutingSource::Router while a replay trace was supplied");
}
