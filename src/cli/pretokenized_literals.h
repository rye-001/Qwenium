#pragma once
#include "pretokenized_literals_qwen2.h"
#include "pretokenized_literals_qwen35.h"

inline const std::map<std::string, std::vector<int32_t>>&
get_pretokenized_literals(const std::string& architecture) {
    if (architecture == "qwen35") {
        return pretokenized_literals_qwen35;
    }
    return pretokenized_literals_qwen2;
}