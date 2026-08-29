// test_graph_compute.cpp — the fail-loud compute check.
//
// The behaviour under test is the one that was missing for the entire life of
// the text path: a non-SUCCESS ggml_status must STOP the pass, not be discarded
// so the caller reads an uncomputed buffer and decodes fluent nonsense
// (docs/architecture.md §12, "A Metal command-buffer OOM does not fail loud").
//
// No GPU and no model needed: the check is a pure function of the status, which
// is exactly why it was written as one.

#include <gtest/gtest.h>

#include <string>

#include "../../src/engine/graph_compute.h"

using qinf::engine::ggml_status_name;
using qinf::engine::require_compute_success;

// Success is the only status that continues.
TEST(GraphCompute, SuccessDoesNotThrow) {
    EXPECT_NO_THROW(require_compute_success(GGML_STATUS_SUCCESS, "unit"));
}

// Every failure status throws — including ABORTED, so a future ggml that starts
// returning it cannot slip through as "not really a failure".
TEST(GraphCompute, EveryNonSuccessStatusThrows) {
    for (ggml_status st : {GGML_STATUS_ALLOC_FAILED,
                           GGML_STATUS_FAILED,
                           GGML_STATUS_ABORTED}) {
        EXPECT_THROW(require_compute_success(st, "unit"), std::runtime_error)
            << "status " << ggml_status_name(st) << " must stop the pass";
    }
}

// The fail-loud contract (CLAUDE.md): the message names the site, then the
// expected value, then the actual one, in that order.
TEST(GraphCompute, MessageNamesSiteExpectedThenActual) {
    try {
        require_compute_success(GGML_STATUS_FAILED, "decode_step");
        FAIL() << "expected a throw";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        const size_t site     = msg.find("decode_step");
        const size_t expected = msg.find("expected GGML_STATUS_SUCCESS");
        const size_t actual   = msg.find("actual: GGML_STATUS_FAILED");
        ASSERT_NE(site,     std::string::npos) << msg;
        ASSERT_NE(expected, std::string::npos) << msg;
        ASSERT_NE(actual,   std::string::npos) << msg;
        EXPECT_LT(site, expected) << "site must come first: " << msg;
        EXPECT_LT(expected, actual) << "expected must precede actual: " << msg;
    }
}

// A command-buffer failure is the case operators actually hit, so the message
// has to say what to do about it and that the state is latched — otherwise the
// second failure looks like a new, different bug.
TEST(GraphCompute, BackendFailureExplainsLatchingAndRemedy) {
    for (ggml_status st : {GGML_STATUS_FAILED, GGML_STATUS_ALLOC_FAILED}) {
        try {
            require_compute_success(st, "unit");
            FAIL() << "expected a throw";
        } catch (const std::runtime_error& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("out-of-memory"), std::string::npos) << msg;
            EXPECT_NE(msg.find("latched"), std::string::npos) << msg;
            EXPECT_NE(msg.find("--ctx-size"), std::string::npos) << msg;
        }
    }
}

// Statuses are named, not printed as integers a reader has to decode.
TEST(GraphCompute, StatusNamesAreHumanReadable) {
    EXPECT_STREQ(ggml_status_name(GGML_STATUS_SUCCESS),      "GGML_STATUS_SUCCESS");
    EXPECT_STREQ(ggml_status_name(GGML_STATUS_FAILED),       "GGML_STATUS_FAILED");
    EXPECT_STREQ(ggml_status_name(GGML_STATUS_ALLOC_FAILED), "GGML_STATUS_ALLOC_FAILED");
    EXPECT_STREQ(ggml_status_name(GGML_STATUS_ABORTED),      "GGML_STATUS_ABORTED");
}
