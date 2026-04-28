// test_gguf_kv_bag.cpp — unit tests for GGUFKVBag
//
// Tests: round-trip set/get for each scalar type; missing-key fail-loud;
// wrong-type fail-loud; optional accessor (nullopt for missing, value for
// present, throw for wrong type); contains().
//
// No model file required — pure in-memory bag operations.
//
// Run: ./qwen3-gguf-kv-bag-tests

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "../../src/core/gguf_value.h"

// ── Round-trip: each scalar type survives set → get ──────────────────────────

TEST(GGUFKVBag, RoundTripUint32) {
    GGUFKVBag bag;
    bag.set("u32", uint32_t{42});
    EXPECT_EQ(bag.get_uint32("u32"), 42u);
}

TEST(GGUFKVBag, RoundTripInt32) {
    GGUFKVBag bag;
    bag.set("i32", int32_t{-7});
    EXPECT_EQ(bag.get_int32("i32"), -7);
}

TEST(GGUFKVBag, RoundTripFloat) {
    GGUFKVBag bag;
    bag.set("f32", float{1e7f});
    EXPECT_FLOAT_EQ(bag.get_float("f32"), 1e7f);
}

TEST(GGUFKVBag, RoundTripBoolTrue) {
    GGUFKVBag bag;
    bag.set("b", bool{true});
    EXPECT_TRUE(bag.get_bool("b"));
}

TEST(GGUFKVBag, RoundTripBoolFalse) {
    GGUFKVBag bag;
    bag.set("b", bool{false});
    EXPECT_FALSE(bag.get_bool("b"));
}

TEST(GGUFKVBag, RoundTripString) {
    GGUFKVBag bag;
    bag.set("s", std::string{"gpt2"});
    EXPECT_EQ(bag.get_string("s"), "gpt2");
}

TEST(GGUFKVBag, RoundTripEmptyString) {
    GGUFKVBag bag;
    bag.set("empty", std::string{""});
    EXPECT_EQ(bag.get_string("empty"), "");
}

// ── Overwrite: set replaces an existing entry ─────────────────────────────────

TEST(GGUFKVBag, SetOverwritesPreviousValue) {
    GGUFKVBag bag;
    bag.set("k", uint32_t{1});
    bag.set("k", uint32_t{99});
    EXPECT_EQ(bag.get_uint32("k"), 99u);
}

// ── contains ─────────────────────────────────────────────────────────────────

TEST(GGUFKVBag, ContainsTrueAfterSet) {
    GGUFKVBag bag;
    bag.set("x", uint32_t{0});
    EXPECT_TRUE(bag.contains("x"));
}

TEST(GGUFKVBag, ContainsFalseForAbsentKey) {
    GGUFKVBag bag;
    EXPECT_FALSE(bag.contains("no_such_key"));
}

// ── Required accessors — missing key ─────────────────────────────────────────

TEST(GGUFKVBag, GetUint32MissingThrowsWithKeyName) {
    GGUFKVBag bag;
    try {
        bag.get_uint32("missing_u32");
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("missing_u32"), std::string::npos) << msg;
        EXPECT_NE(msg.find("missing"), std::string::npos) << msg;
    }
}

TEST(GGUFKVBag, GetFloatMissingThrows) {
    GGUFKVBag bag;
    EXPECT_THROW(bag.get_float("no_float"), std::runtime_error);
}

TEST(GGUFKVBag, GetStringMissingThrows) {
    GGUFKVBag bag;
    EXPECT_THROW(bag.get_string("no_string"), std::runtime_error);
}

// ── Required accessors — wrong type ──────────────────────────────────────────

TEST(GGUFKVBag, GetUint32WrongTypeThrowsNamingBoth) {
    GGUFKVBag bag;
    bag.set("typed", std::string{"hello"});
    try {
        bag.get_uint32("typed");
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("typed"), std::string::npos) << msg;
        EXPECT_NE(msg.find("uint32"), std::string::npos) << msg;
        EXPECT_NE(msg.find("string"), std::string::npos) << msg;
    }
}

TEST(GGUFKVBag, GetFloatWrongTypeThrows) {
    GGUFKVBag bag;
    bag.set("k", uint32_t{5});
    try {
        bag.get_float("k");
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("float"), std::string::npos) << msg;
        EXPECT_NE(msg.find("uint32"), std::string::npos) << msg;
    }
}

TEST(GGUFKVBag, GetBoolWrongTypeThrows) {
    GGUFKVBag bag;
    bag.set("k", int32_t{1});
    EXPECT_THROW(bag.get_bool("k"), std::runtime_error);
}

TEST(GGUFKVBag, GetStringWrongTypeThrows) {
    GGUFKVBag bag;
    bag.set("k", float{3.14f});
    EXPECT_THROW(bag.get_string("k"), std::runtime_error);
}

TEST(GGUFKVBag, GetInt32WrongTypeThrows) {
    GGUFKVBag bag;
    bag.set("k", uint32_t{0});
    // uint32_t and int32_t are distinct variant alternatives — wrong-type throw expected.
    EXPECT_THROW(bag.get_int32("k"), std::runtime_error);
}

// ── Optional accessors — nullopt for missing ──────────────────────────────────

TEST(GGUFKVBag, OptionalUint32NulloptWhenAbsent) {
    GGUFKVBag bag;
    EXPECT_EQ(bag.get_uint32_opt("no_key"), std::nullopt);
}

TEST(GGUFKVBag, OptionalInt32NulloptWhenAbsent) {
    GGUFKVBag bag;
    EXPECT_EQ(bag.get_int32_opt("no_key"), std::nullopt);
}

TEST(GGUFKVBag, OptionalFloatNulloptWhenAbsent) {
    GGUFKVBag bag;
    EXPECT_EQ(bag.get_float_opt("no_key"), std::nullopt);
}

TEST(GGUFKVBag, OptionalBoolNulloptWhenAbsent) {
    GGUFKVBag bag;
    EXPECT_EQ(bag.get_bool_opt("no_key"), std::nullopt);
}

TEST(GGUFKVBag, OptionalStringNulloptWhenAbsent) {
    GGUFKVBag bag;
    EXPECT_EQ(bag.get_string_opt("no_key"), std::nullopt);
}

// ── Optional accessors — value when present ───────────────────────────────────

TEST(GGUFKVBag, OptionalUint32ReturnsValueWhenPresent) {
    GGUFKVBag bag;
    bag.set("k", uint32_t{7});
    auto v = bag.get_uint32_opt("k");
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 7u);
}

TEST(GGUFKVBag, OptionalFloatReturnsValueWhenPresent) {
    GGUFKVBag bag;
    bag.set("f", float{2.5f});
    auto v = bag.get_float_opt("f");
    ASSERT_TRUE(v.has_value());
    EXPECT_FLOAT_EQ(*v, 2.5f);
}

TEST(GGUFKVBag, OptionalStringReturnsValueWhenPresent) {
    GGUFKVBag bag;
    bag.set("s", std::string{"qwen35"});
    auto v = bag.get_string_opt("s");
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, "qwen35");
}

TEST(GGUFKVBag, OptionalBoolReturnsValueWhenPresent) {
    GGUFKVBag bag;
    bag.set("b", bool{true});
    auto v = bag.get_bool_opt("b");
    ASSERT_TRUE(v.has_value());
    EXPECT_TRUE(*v);
}

// ── Optional accessors — wrong type still throws ──────────────────────────────

TEST(GGUFKVBag, OptionalUint32ThrowsOnWrongType) {
    GGUFKVBag bag;
    bag.set("k", std::string{"oops"});
    EXPECT_THROW(bag.get_uint32_opt("k"), std::runtime_error);
}

TEST(GGUFKVBag, OptionalStringThrowsOnWrongType) {
    GGUFKVBag bag;
    bag.set("k", uint32_t{99});
    EXPECT_THROW(bag.get_string_opt("k"), std::runtime_error);
}

// ── Multiple keys — independent entries ──────────────────────────────────────

TEST(GGUFKVBag, MultipleKeysIndependent) {
    GGUFKVBag bag;
    bag.set("a", uint32_t{1});
    bag.set("b", float{2.0f});
    bag.set("c", std::string{"three"});
    bag.set("d", bool{false});
    bag.set("e", int32_t{-5});

    EXPECT_EQ(bag.get_uint32("a"), 1u);
    EXPECT_FLOAT_EQ(bag.get_float("b"), 2.0f);
    EXPECT_EQ(bag.get_string("c"), "three");
    EXPECT_FALSE(bag.get_bool("d"));
    EXPECT_EQ(bag.get_int32("e"), -5);

    EXPECT_TRUE(bag.contains("a"));
    EXPECT_TRUE(bag.contains("b"));
    EXPECT_FALSE(bag.contains("z"));
}
