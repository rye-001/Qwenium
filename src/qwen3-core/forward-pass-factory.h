// forward-pass-factory.h
#pragma once
#include "forward-pass-base.h"
#include <memory>

std::unique_ptr<ForwardPassBase> create_forward_pass(
    const Qwen3Model& model,
    const Qwen3Metadata* metadata,
    uint32_t context_len,
    uint32_t max_batch_size = 1,
    int kv_quant_bits = 0,
    uint32_t snapkv_budget = 0,
    uint32_t snapkv_window = 32);