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
// QINF_ASSERT is used by the qwen35/qwen36 config validators. QINF_SLOT_ERROR
// has NO caller as of 2026-08-29: its only one was loader/weight_binding, the
// declarative binder that the blueprint named canonical but nothing ever used
// (deleted). The live weight path is Model::assign_tensor_pointers, which
// formats its errors by hand. Kept pending a decision on which of the two the
// contract should be expressed through — see docs/plan-post-vision-consolidation.md.

#include <stdexcept>
#include <string>

// Throw std::runtime_error with msg if cond is false.
#define QINF_ASSERT(cond, msg)                                    \
    do {                                                          \
        if (!(cond)) {                                            \
            throw std::runtime_error(std::string(msg));           \
        }                                                         \
    } while (0)

// Throw the canonical slot-error: "<module>: slot <slot> expected <exp>, got <got>".
// NOTE: the literal prefix below still says "weight_binding" — stale now that the
// module is gone; fixing it changes error text, so it is a separate decision.
#define QINF_SLOT_ERROR(slot, expected, got)                                         \
    throw std::runtime_error(                                                         \
        std::string("weight_binding: slot \"") + (slot) +                            \
        "\" expected " + (expected) + ", got " + (got))
