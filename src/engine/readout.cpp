// readout.cpp — see readout.h. The sequence below is the one the lens wrote
// by hand (../Qemmi-lens-server/src/server/server_lens.cpp, run_lens_locate's
// tapped pass and run_lens_tapped_decode's step), moved here verbatim in order:
// arm → build → mark taps → mark routing → alloc_readback_graph → inputs →
// compute → read routing → read taps → read logits → advance → disarm.

#include "readout.h"

#include <stdexcept>
#include <string>

#include "graph_compute.h"

namespace qinf::engine::readout {

namespace {

// Every override a Spec can set, saved on entry and restored on exit — the
// one copy of what the lens's four EngineRestore structs each re-implemented.
// The slot itself is not restored: writing the slot is the point of the call.
struct Scope {
    ForwardPassBase& fp;
    int truncate;
    std::vector<int> tap_layers, tap_heads;
    ForwardPassBase::AttnImpl prefill_attn;
    RoutingTrace* capture;
    const RoutingTrace* replay;

    explicit Scope(ForwardPassBase& f)
        : fp(f), truncate(f.truncate_after_layer()), tap_layers(f.attention_taps()),
          tap_heads(f.attention_tap_heads()), prefill_attn(f.prefill_attn_impl()),
          capture(f.routing_capture()), replay(f.routing_replay()) {}

    void arm(const Spec& s) {
        fp.set_truncate_after_layer(s.last_layer);
        fp.set_attention_taps(s.tap_layers, s.tap_heads);
        if (s.prefill_attn) fp.set_prefill_attn_impl(*s.prefill_attn);
        fp.set_routing_capture(s.capture);
        fp.set_routing_replay(s.replay);
    }

    ~Scope() {
        fp.set_truncate_after_layer(truncate);
        fp.set_attention_taps(tap_layers, tap_heads);
        fp.set_prefill_attn_impl(prefill_attn);
        fp.set_routing_capture(capture);
        fp.set_routing_replay(replay);
        fp.set_sparse_decode_ids({});
    }
};

void check_taps(const Spec& s, const std::vector<ForwardPassBase::AttentionTap>& taps, const char* who) {
    if (taps.size() != s.tap_layers.size())
        throw std::runtime_error(std::string(who) + ": slot 'taps': expected " +
                                 std::to_string(s.tap_layers.size()) + " (one per tap layer), actual " +
                                 std::to_string(taps.size()));
    for (size_t i = 0; i < taps.size(); ++i)
        if (taps[i].layer != s.tap_layers[i])
            throw std::runtime_error(std::string(who) + ": slot 'taps[" + std::to_string(i) +
                                     "].layer': expected " + std::to_string(s.tap_layers[i]) +
                                     " (request order), actual " + std::to_string(taps[i].layer));
}

void read_logits(ForwardPassBase& fp, ggml_cgraph* gf, Result& r, const char* who) {
    const ggml_tensor* lt = ggml_graph_get_tensor(gf, "logits");
    if (!lt)
        throw std::runtime_error(std::string(who) + ": slot 'logits': expected an output head in the graph, "
                                 "actual none");
    r.logits = fp.get_output_logits(gf);
    r.logits_width = static_cast<size_t>(lt->ne[0]);
    if (r.logits_width == 0 || r.logits.size() < r.logits_width || r.logits.size() % r.logits_width != 0)
        throw std::runtime_error(std::string(who) + ": slot 'logits': expected whole rows of width " +
                                 std::to_string(r.logits_width) + ", actual " + std::to_string(r.logits.size()) +
                                 " floats");
}

}  // namespace

Result prefill(ForwardPassBase& fp, ggml_backend_sched_t sched, uint32_t slot, int pos,
               const std::vector<int32_t>& tokens, const Spec& spec) {
    if (tokens.empty())
        throw std::runtime_error("readout::prefill: slot 'tokens': expected at least one, actual 0");
    Scope scope(fp);
    scope.arm(spec);
    Result r;
    ggml_cgraph* gf = fp.build_prefill_graph(tokens, pos, slot, /*want_logits=*/spec.logits);
    fp.mark_attention_taps(gf);
    if (spec.capture) fp.mark_moe_routing(gf);
    fp.alloc_readback_graph(sched, gf);   // a plan made for THIS graph; plain alloc when nothing is marked
    fp.set_prefill_inputs(gf, tokens, pos);
    require_compute_success(ggml_backend_sched_graph_compute(sched, gf), "readout::prefill");
    if (spec.capture) fp.read_moe_routing(gf, *spec.capture, pos);
    if (!spec.tap_layers.empty()) {
        r.taps = fp.get_attention_taps(gf);
        check_taps(spec, r.taps, "readout::prefill");
    }
    if (spec.logits) read_logits(fp, gf, r, "readout::prefill");
    fp.advance_cache(static_cast<uint32_t>(tokens.size()), slot);
    return r;
}

Result decode(ForwardPassBase& fp, ggml_backend_sched_t sched, uint32_t slot, int32_t position,
              int32_t token, const Spec& spec) {
    if (spec.last_layer != -1)
        throw std::runtime_error("readout::decode: slot 'last_layer': expected -1 (no recipe cuts a decode), "
                                 "actual " + std::to_string(spec.last_layer));
    Scope scope(fp);
    scope.arm(spec);
    fp.set_sparse_decode_ids(spec.sparse_ids);   // always set: clears any stale ids
    Result r;
    const std::vector<int32_t> tks = {token};
    const std::vector<uint32_t> slots = {slot};
    const std::vector<int32_t> positions = {position};
    ggml_cgraph* gf = fp.build_decoding_graph(tks, slots, positions);
    fp.mark_attention_taps(gf);
    if (spec.capture) fp.mark_moe_routing(gf);
    fp.alloc_readback_graph(sched, gf);
    fp.set_decode_inputs(gf, tks, slots, positions);
    require_compute_success(ggml_backend_sched_graph_compute(sched, gf), "readout::decode");
    if (spec.capture) fp.read_moe_routing(gf, *spec.capture, position);
    if (!spec.tap_layers.empty()) {
        r.taps = fp.get_attention_taps(gf);
        check_taps(spec, r.taps, "readout::decode");
    }
    if (spec.logits) read_logits(fp, gf, r, "readout::decode");
    fp.advance_cache(1, slot);
    return r;
}

}  // namespace qinf::engine::readout
