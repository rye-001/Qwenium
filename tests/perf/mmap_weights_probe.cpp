// mmap-backed weights probe.
//
// Compares the two weight-load paths on one model:
//   copy  (default)  — ggml_backend_alloc_ctx_tensors + a memcpy per tensor,
//                      then release_file_mapping(). Peak ~= 2x model size.
//   mmap  (opt-in)   — ggml_backend_dev_buffer_from_host_ptr over the GGUF's
//                      mapped pages; no copy, mapping retained.
//
// Reports peak RSS, steady RSS, and a checksum of one prefill's logits so the
// two paths can be proven byte-identical (same bytes, different provenance).
//
//   [QINF_MMAP_WEIGHTS=1] ./bin/mmap-weights-probe [model.gguf]
//
// Model path: argv[1] if given, else QWEN36_MODEL_PATH, else the qwen35moe
// default. The resolved path is printed — a probe that silently benchmarks a
// different model than you asked for is worse than one that refuses to run.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include <mach/mach.h>
#include <mach/task_info.h>
#include <sys/resource.h>

#include "ggml.h"
#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/forward_pass_base.h"

namespace {

size_t rss_now_mb() {
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) != KERN_SUCCESS) return 0;
    return (size_t)(info.resident_size / (1024 * 1024));
}

// phys_footprint is what Activity Monitor calls "Memory" — it counts dirty
// anonymous pages AND the process's share of resident file-backed pages, so it
// does NOT let mmap'd weights hide the way plain RSS does. This is the honest
// number for a copy-vs-mmap comparison.
size_t phys_footprint_mb() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  (task_info_t)&info, &count) != KERN_SUCCESS) return 0;
    return (size_t)(info.phys_footprint / (1024 * 1024));
}

size_t rss_peak_mb() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return (size_t)(ru.ru_maxrss / (1024 * 1024));  // macOS reports bytes
}

}  // namespace

int main(int argc, char** argv) {
    std::string path;
    if (argc > 1) path = argv[1];
    else if (const char* e = std::getenv("QWEN36_MODEL_PATH")) path = e;
    else path = "models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf";

    const bool use_mmap = std::getenv("QINF_MMAP_WEIGHTS") != nullptr;

    std::cerr << "=== mmap-weights probe ===\n"
              << "model : " << path << "\n"
              << "path  : " << (use_mmap ? "MMAP (buffer_from_host_ptr)"
                                         : "COPY (alloc_ctx_tensors + memcpy)") << "\n";

    register_builtin_models();

    const size_t rss_before = rss_now_mb();
    auto t0 = std::chrono::steady_clock::now();
    Model model;
    model.set_mmap_weights(use_mmap);
    model.load_metadata(path);
    model.load_tensors();

    auto t1 = std::chrono::steady_clock::now();
    const size_t rss_after = rss_now_mb();
    const size_t peak_load = rss_peak_mb();

    const auto& meta = model.get_metadata();
    auto fp = create_forward_pass(model, &meta, 2048, 1);
    ggml_backend_sched_t sched = model.get_scheduler();

    // Fixed synthetic prompt — same token ids on both paths, no tokenizer in
    // the comparison (the decode-gap ledger's reproduction convention).
    std::vector<int32_t> tokens;
    for (int i = 0; i < 64; ++i) tokens.push_back((int32_t)((i * 7 + 3) % 1000));

    // Cold prefill: on the mmap path this is where weight pages fault in.
    fp->clear_slot(0);
    fp->set_cache_pos(0, 0);
    auto p0 = std::chrono::steady_clock::now();
    std::vector<float> logits = fp->run_prefill(tokens, 0, 0, sched);
    auto p1 = std::chrono::steady_clock::now();

    // Warm prefills: pages already faulted, so this is the steady-state cost.
    double warm_ms = 0.0;
    for (int r = 0; r < 5; ++r) {
        fp->clear_slot(0);
        fp->set_cache_pos(0, 0);
        auto w0 = std::chrono::steady_clock::now();
        volatile auto l2 = fp->run_prefill(tokens, 0, 0, sched);
        auto w1 = std::chrono::steady_clock::now();
        (void)l2;
        warm_ms += std::chrono::duration<double, std::milli>(w1 - w0).count();
    }
    warm_ms /= 5.0;

    // Checksum: order-sensitive, exact-bit. Two paths must agree exactly.
    uint64_t h = 1469598103934665603ULL;
    const unsigned char* raw = (const unsigned char*)logits.data();
    for (size_t i = 0; i < logits.size() * sizeof(float); ++i) {
        h ^= raw[i]; h *= 1099511628211ULL;
    }
    size_t best = 0;
    for (size_t i = 1; i < logits.size(); ++i) if (logits[i] > logits[best]) best = i;

    std::cerr << "rss before load : " << rss_before << " MB\n"
              << "rss after load  : " << rss_after  << " MB\n"
              << "PEAK rss        : " << peak_load  << " MB\n"
              << "peak after fwd  : " << rss_peak_mb() << " MB\n"
              << "load ms         : " << std::chrono::duration<double,std::milli>(t1-t0).count() << "\n"
              << "cold prefill ms : " << std::chrono::duration<double,std::milli>(p1-p0).count() << "\n"
              << "warm prefill ms : " << warm_ms << "\n"
              << "phys footprint  : " << phys_footprint_mb() << " MB\n"
              << "logits fnv1a    : " << h << "\n"
              << "logits argmax   : " << best << "\n"
              << "logits count    : " << logits.size() << "\n";
    return 0;
}
