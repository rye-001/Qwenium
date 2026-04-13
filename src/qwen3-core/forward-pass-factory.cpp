// forward-pass-factory.cpp
#include "forward-pass-factory.h"
#include "forward-pass.h"
#include "forward-pass-qwen35.h"

std::unique_ptr<ForwardPassBase> create_forward_pass(
    const Qwen3Model& model,
    const Qwen3Metadata* metadata,
    uint32_t context_len,
    uint32_t max_batch_size,
    int kv_quant_bits,
    uint32_t snapkv_budget,
    uint32_t snapkv_window)
{
    std::unique_ptr<ForwardPassBase> fp;
    if (metadata->architecture == "qwen35") {
        fp = std::make_unique<Qwen35ForwardPass>(model, metadata, context_len, max_batch_size, kv_quant_bits);
    } else {
        fp = std::make_unique<Qwen3ForwardPass>(model, metadata, context_len, max_batch_size, kv_quant_bits);
    }
    fp->set_snapkv_config(snapkv_budget, snapkv_window);
    return fp;
}