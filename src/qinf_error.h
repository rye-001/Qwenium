#pragma once
// qinf_error.h — error-contract macros for module boundaries.
//
// Every error at a module boundary must name the slot/parameter, the expected
// value, and the actual value, in that order.  These macros enforce that format.
// Silent fallbacks and best-effort recovery are forbidden at module boundaries.
//
// Usage:
//   QINF_ASSERT(condition, "context: what went wrong")
//   QINF_SLOT_ERROR("q_proj", "shape [4096,4096] dtype f16", "shape [4096,2048] dtype f16")
//
// Unit test: tests/unit/test_weight_binding.cpp (exercises error messages)

#include <stdexcept>
#include <string>

// Throw std::runtime_error with msg if cond is false.
#define QINF_ASSERT(cond, msg)                                    \
    do {                                                          \
        if (!(cond)) {                                            \
            throw std::runtime_error(std::string(msg));           \
        }                                                         \
    } while (0)

// Throw the canonical slot-error: "weight_binding: slot <slot> expected <exp>, got <got>"
#define QINF_SLOT_ERROR(slot, expected, got)                                         \
    throw std::runtime_error(                                                         \
        std::string("weight_binding: slot \"") + (slot) +                            \
        "\" expected " + (expected) + ", got " + (got))
