#pragma once
// qinf_error.h — error-contract macros for module boundaries.
//
// Every error at a module boundary must name the slot/parameter, the expected
// value, and the actual value, in that order.  These macros enforce that format.
// Silent fallbacks and best-effort recovery are forbidden at module boundaries.
//
// Usage:
//   QINF_ASSERT(condition, "context: what went wrong")
//
// Scope: QINF_ASSERT, used by the qwen35/qwen36 config validators.
//
// There was also a QINF_SLOT_ERROR macro emitting
//   "weight_binding: slot <slot> expected <exp>, got <got>"
// Its only caller was loader/weight_binding — the declarative binder the
// blueprint named canonical and nothing ever adopted — so when that was deleted
// the macro had none, and its hardcoded "weight_binding:" prefix named a module
// that no longer existed. Deleted 2026-08-29. The contract it encoded is not
// gone: errors still name the slot, the expected value and the actual one, in
// that order — see Model::assign_tensor_pointers' require() and the fail-loud
// messages throughout src/. The format is the rule; the macro was one unused
// way of spelling it.

#include <stdexcept>
#include <string>

// Throw std::runtime_error with msg if cond is false.
#define QINF_ASSERT(cond, msg)                                    \
    do {                                                          \
        if (!(cond)) {                                            \
            throw std::runtime_error(std::string(msg));           \
        }                                                         \
    } while (0)

