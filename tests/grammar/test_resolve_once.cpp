// test_resolve_once.cpp
// TDD tests for the resolve-once optimization in grammar_vocab.cpp.
//
// The optimization pre-resolves grammar states after a LITERAL once per group,
// then matches remaining token characters against the pre-resolved terminals.
// These tests verify correctness by exercising the exact scenarios described
// in docs/plan-resolve-once.md.

#include "grammar_vocab.h"
#include "token-trie.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <set>
#include <string>
#include <stdexcept>
#include <algorithm>

// ============================================================================
// Minimal Test Framework
// ============================================================================

static int g_passed = 0;
static int g_failed = 0;

#define ASSERT_NE(a, b) \
    if ((a) == (b))     \
    throw std::runtime_error("ASSERT_NE failed: " #a " == " #b)
#define ASSERT_EQ(a, b) \
    if ((a) != (b))     \
    throw std::runtime_error("ASSERT_EQ failed: " #a " != " #b)
#define EXPECT_TRUE(x) \
    if (!(x))          \
    throw std::runtime_error("EXPECT_TRUE failed: " #x)
#define EXPECT_FALSE(x) \
    if ((x))            \
    throw std::runtime_error("EXPECT_FALSE failed: " #x)

void run_test(const char *name, void (*func)())
{
    std::cout << "  " << name << "... " << std::flush;
    try
    {
        func();
        std::cout << "\033[32mPASS\033[0m\n";
        g_passed++;
    }
    catch (const std::exception &e)
    {
        std::cout << "\033[31mFAIL\033[0m: " << e.what() << "\n";
        g_failed++;
    }
}

// Helper: compare valid token sets
void assert_same_valid_tokens(const std::vector<int32_t>& a, const std::vector<int32_t>& b,
                               const char* context)
{
    std::set<int32_t> sa(a.begin(), a.end());
    std::set<int32_t> sb(b.begin(), b.end());
    if (sa != sb) {
        std::string msg = std::string(context) + "\nExpected {";
        for (auto x : sa) msg += std::to_string(x) + " ";
        msg += "} got {";
        for (auto x : sb) msg += std::to_string(x) + " ";
        msg += "}";

        // Show diff
        std::vector<int32_t> only_in_a, only_in_b;
        std::set_difference(sa.begin(), sa.end(), sb.begin(), sb.end(), std::back_inserter(only_in_a));
        std::set_difference(sb.begin(), sb.end(), sa.begin(), sa.end(), std::back_inserter(only_in_b));
        if (!only_in_a.empty()) {
            msg += "\nMissing: {";
            for (auto x : only_in_a) msg += std::to_string(x) + " ";
            msg += "}";
        }
        if (!only_in_b.empty()) {
            msg += "\nExtra: {";
            for (auto x : only_in_b) msg += std::to_string(x) + " ";
            msg += "}";
        }
        throw std::runtime_error(msg);
    }
}

// ============================================================================
// Grammar: models the real DSL with many space-prefixed candidates
// This is the exact scenario from the plan: " " literal with many candidates
// ============================================================================

const char* RESOLVE_ONCE_GRAMMAR = R"DELIM(
root ::= statement

statement ::= let_decl | await_call

let_decl ::= "const" " " identifier " " "=" " " expression ";"

await_call ::= "await" " " function_call ";"

expression ::= await_expr | variable | string

await_expr ::= "await" " " function_call

function_call ::= function_name "(" args ")"

args ::= "{" " "? param_list " "? "}" | ""

param_list ::= param ("," " "? param)*

param ::= param_key ":" " "? param_value

param_key ::= "accountId" | "tier" | "note"

param_value ::= variable | string

variable ::= identifier ("." identifier)*

identifier ::= "accounts" | "account" | "orders" | "tickets" | "id" | "length"

function_name ::= "getAccounts" | "getOpenOrders" | "addAccountNote"

string ::= "\"" string_content "\""

string_content ::= [^"]*
)DELIM";

// Build a vocab with many space-prefixed tokens (simulating real BPE)
// This creates the "424 candidates" scenario from the plan
std::vector<std::string> build_resolve_once_vocab()
{
    std::vector<std::string> vocab(300);
    // Keywords
    vocab[0]  = "const";    vocab[1]  = "for";      vocab[2]  = "of";
    vocab[3]  = "if";       vocab[4]  = "else";      vocab[5]  = "await";
    // Space-prefixed keywords (BPE merges)
    vocab[6]  = " const";   vocab[7]  = " for";      vocab[8]  = " of";
    vocab[9]  = " if";      vocab[10] = " else";     vocab[11] = " await";
    // Punctuation
    vocab[12] = "(";        vocab[13] = ")";         vocab[14] = "{";
    vocab[15] = "}";        vocab[16] = ";";         vocab[17] = ".";
    vocab[18] = ",";        vocab[19] = ":";         vocab[20] = "=";
    vocab[21] = ">";        vocab[22] = "<";         vocab[23] = "\"";
    // Whitespace
    vocab[24] = " ";        vocab[25] = "\n";
    // Function names
    vocab[30] = "getAccounts";     vocab[31] = "getOpenOrders";
    vocab[32] = "addAccountNote";
    // Space-prefixed function names
    vocab[33] = " getAccounts";    vocab[34] = " getOpenOrders";
    vocab[35] = " addAccountNote";
    // Identifiers
    vocab[40] = "accounts"; vocab[41] = "account";   vocab[42] = "orders";
    vocab[43] = "tickets";  vocab[44] = "id";        vocab[45] = "length";
    // Space-prefixed identifiers
    vocab[46] = " accounts"; vocab[47] = " account";
    vocab[48] = " orders";   vocab[49] = " tickets";
    // Param keys
    vocab[50] = "accountId"; vocab[51] = "tier";     vocab[52] = "note";
    // Split tokens
    vocab[60] = "get";       vocab[61] = "Accounts"; vocab[62] = "Open";
    vocab[63] = "Orders";
    // String values (dynamic content)
    vocab[70] = "gold";      vocab[71] = "platinum";
    vocab[72] = "hello";     vocab[73] = "world";
    // Digits (for char class testing)
    vocab[80] = "0"; vocab[81] = "1"; vocab[82] = "2"; vocab[83] = "3";
    vocab[84] = "4"; vocab[85] = "5"; vocab[86] = "6"; vocab[87] = "7";
    vocab[88] = "8"; vocab[89] = "9";

    // Bulk space-prefixed tokens (simulating BPE vocab)
    // These create many candidates for " " literal prefix matching
    for (int i = 100; i < 200; i++) {
        vocab[i] = " tok" + std::to_string(i);
    }

    return vocab;
}

// ============================================================================
// Test 1: Equivalence — brute-force vs trie path at initial position
// Both paths must produce identical valid token sets.
// ============================================================================

void test_equivalence_initial_position()
{
    auto vocab = build_resolve_once_vocab();

    // Get valid tokens WITHOUT trie (brute-force)
    auto grammar_bf = qwenium::GrammarVocab::parse_impl(RESOLVE_ONCE_GRAMMAR);
    ASSERT_NE(grammar_bf, nullptr);
    auto valid_bf = grammar_bf->get_valid_tokens(vocab);

    // Get valid tokens WITH trie
    auto grammar_trie = qwenium::GrammarVocab::parse_impl(RESOLVE_ONCE_GRAMMAR);
    ASSERT_NE(grammar_trie, nullptr);
    qwenium::TokenTrie trie;
    trie.build(vocab);
    grammar_trie->set_token_trie(&trie);
    auto valid_trie = grammar_trie->get_valid_tokens(vocab);

    assert_same_valid_tokens(valid_bf, valid_trie, "initial position");
}

// ============================================================================
// Test 2: Equivalence after accepting "const" " " — at identifier position
// This tests the " " literal with many space-prefixed candidates
// ============================================================================

void test_equivalence_after_space_literal()
{
    auto vocab = build_resolve_once_vocab();
    qwenium::TokenTrie trie;
    trie.build(vocab);

    auto grammar_bf = qwenium::GrammarVocab::parse_impl(RESOLVE_ONCE_GRAMMAR);
    auto grammar_trie = qwenium::GrammarVocab::parse_impl(RESOLVE_ONCE_GRAMMAR);
    grammar_trie->set_token_trie(&trie);

    // Accept "const" then " " → grammar now at identifier position
    grammar_bf->accept_token(0, vocab);    // "const"
    grammar_trie->accept_token(0, vocab);
    grammar_bf->accept_token(24, vocab);   // " "
    grammar_trie->accept_token(24, vocab);

    auto valid_bf = grammar_bf->get_valid_tokens(vocab);
    auto valid_trie = grammar_trie->get_valid_tokens(vocab);

    assert_same_valid_tokens(valid_bf, valid_trie, "after 'const' ' '");
}

// ============================================================================
// Test 3: Many candidates for single-char literal " "
// Verifies that space-prefixed tokens are correctly handled
// when the grammar expects " " followed by a complex rule.
// ============================================================================

void test_space_literal_many_candidates()
{
    auto vocab = build_resolve_once_vocab();
    qwenium::TokenTrie trie;
    trie.build(vocab);

    auto grammar = qwenium::GrammarVocab::parse_impl(RESOLVE_ONCE_GRAMMAR);
    grammar->set_token_trie(&trie);

    // "const" → now at " " literal
    grammar->accept_token(0, vocab);

    auto valid = grammar->get_valid_tokens(vocab);
    std::set<int32_t> valid_set(valid.begin(), valid.end());

    // " " should be valid (bare space)
    EXPECT_TRUE(valid_set.count(24));
    // " accounts" should be valid (space + valid identifier)
    EXPECT_TRUE(valid_set.count(46));
    // " account" should be valid
    EXPECT_TRUE(valid_set.count(47));
    // " orders" should be valid
    EXPECT_TRUE(valid_set.count(48));
    // " tok100" etc should NOT be valid (not a valid identifier)
    EXPECT_FALSE(valid_set.count(100));
    // "const" should NOT be valid (doesn't start with space)
    EXPECT_FALSE(valid_set.count(0));
}

// ============================================================================
// Test 4: Token that exactly matches remaining literal (no overshoot)
// ============================================================================

void test_exact_literal_match()
{
    auto vocab = build_resolve_once_vocab();
    qwenium::TokenTrie trie;
    trie.build(vocab);

    auto grammar = qwenium::GrammarVocab::parse_impl(RESOLVE_ONCE_GRAMMAR);
    grammar->set_token_trie(&trie);

    // Feed "const" → grammar expects " "
    grammar->accept_token(0, vocab);

    auto valid = grammar->get_valid_tokens(vocab);
    std::set<int32_t> valid_set(valid.begin(), valid.end());

    // Bare " " (token 24) exactly matches the literal — no overshoot
    EXPECT_TRUE(valid_set.count(24));

    // Accept " " → grammar at identifier
    grammar->accept_token(24, vocab);
    auto valid2 = grammar->get_valid_tokens(vocab);
    std::set<int32_t> valid_set2(valid2.begin(), valid2.end());

    // Identifiers should be valid
    EXPECT_TRUE(valid_set2.count(40));  // "accounts"
    EXPECT_TRUE(valid_set2.count(41));  // "account"
}

// ============================================================================
// Test 5: Multi-element token spanning 3+ grammar elements
// e.g. " getAccounts(" spans " " → "getAccounts" → "("
// ============================================================================

void test_multi_element_token()
{
    const char* grammar_str = R"DELIM(
root ::= "await" " " "getAccounts" "(" ")" ";"
)DELIM";

    std::vector<std::string> vocab(50);
    vocab[0] = "await";
    vocab[1] = " ";
    vocab[2] = "getAccounts";
    vocab[3] = "(";
    vocab[4] = ")";
    vocab[5] = ";";
    vocab[6] = " getAccounts(";  // spans 3 elements: " " + "getAccounts" + "("
    vocab[7] = " getAccounts";   // spans 2 elements: " " + "getAccounts"

    qwenium::TokenTrie trie;
    trie.build(vocab);

    // Without trie
    auto grammar_bf = qwenium::GrammarVocab::parse_impl(grammar_str);
    grammar_bf->accept_token(0, vocab); // "await"
    auto valid_bf = grammar_bf->get_valid_tokens(vocab);

    // With trie
    auto grammar_trie = qwenium::GrammarVocab::parse_impl(grammar_str);
    grammar_trie->set_token_trie(&trie);
    grammar_trie->accept_token(0, vocab); // "await"
    auto valid_trie = grammar_trie->get_valid_tokens(vocab);

    assert_same_valid_tokens(valid_bf, valid_trie, "multi-element token");

    // Verify the multi-element tokens are in the valid set
    std::set<int32_t> valid_set(valid_trie.begin(), valid_trie.end());
    EXPECT_TRUE(valid_set.count(1));   // " "
    EXPECT_TRUE(valid_set.count(6));   // " getAccounts("
    EXPECT_TRUE(valid_set.count(7));   // " getAccounts"
}

// ============================================================================
// Test 6: CHAR_CLASS immediately following a LITERAL
// After consuming a literal, the next element is a CHAR_CLASS
// ============================================================================

void test_char_class_after_literal()
{
    const char* grammar_str = R"DELIM(
root ::= "prefix" [a-z]+
)DELIM";

    std::vector<std::string> vocab(30);
    vocab[0] = "prefix";
    vocab[1] = "prefixa";    // "prefix" + "a" (literal + char class)
    vocab[2] = "prefixabc";  // "prefix" + "abc"
    vocab[3] = "a";
    vocab[4] = "abc";
    vocab[5] = "xyz";
    vocab[6] = "PREFIX";     // wrong case

    qwenium::TokenTrie trie;
    trie.build(vocab);

    // Without trie
    auto grammar_bf = qwenium::GrammarVocab::parse_impl(grammar_str);
    auto valid_bf = grammar_bf->get_valid_tokens(vocab);

    // With trie
    auto grammar_trie = qwenium::GrammarVocab::parse_impl(grammar_str);
    grammar_trie->set_token_trie(&trie);
    auto valid_trie = grammar_trie->get_valid_tokens(vocab);

    assert_same_valid_tokens(valid_bf, valid_trie, "char_class after literal");

    std::set<int32_t> valid_set(valid_trie.begin(), valid_trie.end());
    EXPECT_TRUE(valid_set.count(0));   // "prefix" (partial — still needs [a-z]+)
    // Wait, "prefix" alone doesn't satisfy [a-z]+ — it needs at least one char.
    // Actually "prefix" IS a partial match of "prefix" literal — it fully consumes
    // the literal but still needs the CHAR_CLASS.
    // Let's check: after accepting "prefix", [a-z]+ requires at least one char.
    // So "prefix" by itself should NOT complete the grammar, but it SHOULD be
    // a valid token because the grammar can continue with the next token matching [a-z]+.
    // Actually no — after fully consuming the literal "prefix", the grammar needs
    // [a-z]+ which needs at least one more char. If the token IS exactly "prefix",
    // the grammar state after is at [a-z]+ position which is valid (grammar continues).
    EXPECT_TRUE(valid_set.count(1));   // "prefixa" — spans literal + one char class match
    EXPECT_TRUE(valid_set.count(2));   // "prefixabc" — spans literal + three matches
    EXPECT_FALSE(valid_set.count(6));  // "PREFIX" — wrong case, doesn't match
}

// ============================================================================
// Test 7: Full equivalence through a multi-step decode
// Walk through an entire statement, checking equivalence at each step
// ============================================================================

void test_equivalence_full_decode()
{
    auto vocab = build_resolve_once_vocab();
    qwenium::TokenTrie trie;
    trie.build(vocab);

    auto grammar_bf = qwenium::GrammarVocab::parse_impl(RESOLVE_ONCE_GRAMMAR);
    auto grammar_trie = qwenium::GrammarVocab::parse_impl(RESOLVE_ONCE_GRAMMAR);
    grammar_trie->set_token_trie(&trie);

    // Decode: const accounts = await getAccounts();
    int32_t tokens[] = {
        0,   // "const"
        24,  // " "
        40,  // "accounts"
        24,  // " "
        20,  // "="
        24,  // " "
        5,   // "await"
        24,  // " "
        30,  // "getAccounts"
        12,  // "("
        13,  // ")"
        16,  // ";"
    };

    for (int i = 0; i < 12; i++) {
        auto valid_bf = grammar_bf->get_valid_tokens(vocab);
        auto valid_trie = grammar_trie->get_valid_tokens(vocab);

        std::string ctx = "step " + std::to_string(i) + " (before '" + vocab[tokens[i]] + "')";
        assert_same_valid_tokens(valid_bf, valid_trie, ctx.c_str());

        grammar_bf->accept_token(tokens[i], vocab);
        grammar_trie->accept_token(tokens[i], vocab);
    }

    EXPECT_TRUE(grammar_bf->is_accepting_state());
    EXPECT_TRUE(grammar_trie->is_accepting_state());
}

// ============================================================================
// Test 8: Nested groups with OP_STAR after literal
// ============================================================================

void test_star_after_literal()
{
    const char* grammar_str = R"DELIM(
root ::= "a" ("b" "c")* "d"
)DELIM";

    std::vector<std::string> vocab(20);
    vocab[0] = "a";
    vocab[1] = "b";
    vocab[2] = "c";
    vocab[3] = "d";
    vocab[4] = "ab";   // "a" + first char of group
    vocab[5] = "abc";  // "a" + full first iteration
    vocab[6] = "abcd"; // "a" + one iteration + "d"
    vocab[7] = "ad";   // "a" + "d" (zero iterations)
    vocab[8] = "bc";   // full iteration
    vocab[9] = "bcd";  // full iteration + "d"

    qwenium::TokenTrie trie;
    trie.build(vocab);

    // Without trie
    auto grammar_bf = qwenium::GrammarVocab::parse_impl(grammar_str);
    auto valid_bf = grammar_bf->get_valid_tokens(vocab);

    // With trie
    auto grammar_trie = qwenium::GrammarVocab::parse_impl(grammar_str);
    grammar_trie->set_token_trie(&trie);
    auto valid_trie = grammar_trie->get_valid_tokens(vocab);

    assert_same_valid_tokens(valid_bf, valid_trie, "star after literal");
}

// ============================================================================
// Test 9: Equivalence with string content (char class + literal combo)
// ============================================================================

void test_equivalence_string_content()
{
    const char* grammar_str = R"DELIM(
root ::= "\"" [^"]* "\""
)DELIM";

    std::vector<std::string> vocab(30);
    vocab[0] = "\"";
    vocab[1] = "hello";
    vocab[2] = "world";
    vocab[3] = " ";
    vocab[4] = "hello world";
    vocab[5] = "123";
    vocab[6] = "a";

    qwenium::TokenTrie trie;
    trie.build(vocab);

    // After opening quote, both paths should agree
    auto grammar_bf = qwenium::GrammarVocab::parse_impl(grammar_str);
    grammar_bf->accept_token(0, vocab); // "\""
    auto valid_bf = grammar_bf->get_valid_tokens(vocab);

    auto grammar_trie = qwenium::GrammarVocab::parse_impl(grammar_str);
    grammar_trie->set_token_trie(&trie);
    grammar_trie->accept_token(0, vocab); // "\""
    auto valid_trie = grammar_trie->get_valid_tokens(vocab);

    assert_same_valid_tokens(valid_bf, valid_trie, "string content after quote");
}

// ============================================================================
// Test 10: Partial literal consumption (token shorter than literal)
// This path should be trivially correct — just char comparison.
// ============================================================================

void test_partial_literal_consumption()
{
    const char* grammar_str = R"DELIM(
root ::= "getAccounts" "(" ")"
)DELIM";

    std::vector<std::string> vocab(20);
    vocab[0] = "get";          // partial match of "getAccounts"
    vocab[1] = "getAccounts";  // full match
    vocab[2] = "getOpen";      // wrong partial
    vocab[3] = "Accounts";     // suffix only
    vocab[4] = "(";
    vocab[5] = ")";
    vocab[6] = "getAccounts("; // full match + overshoot into next element
    vocab[7] = "getAccounts()"; // full match + 2 elements

    qwenium::TokenTrie trie;
    trie.build(vocab);

    auto grammar_bf = qwenium::GrammarVocab::parse_impl(grammar_str);
    auto valid_bf = grammar_bf->get_valid_tokens(vocab);

    auto grammar_trie = qwenium::GrammarVocab::parse_impl(grammar_str);
    grammar_trie->set_token_trie(&trie);
    auto valid_trie = grammar_trie->get_valid_tokens(vocab);

    assert_same_valid_tokens(valid_bf, valid_trie, "partial literal consumption");

    std::set<int32_t> valid_set(valid_trie.begin(), valid_trie.end());
    EXPECT_TRUE(valid_set.count(0));   // "get" — partial match, valid
    EXPECT_TRUE(valid_set.count(1));   // "getAccounts" — full match
    EXPECT_FALSE(valid_set.count(2));  // "getOpen" — wrong chars
    EXPECT_FALSE(valid_set.count(3));  // "Accounts" — doesn't start with 'g'
    EXPECT_TRUE(valid_set.count(6));   // "getAccounts(" — overshoot, matches next element
    EXPECT_TRUE(valid_set.count(7));   // "getAccounts()" — overshoot, matches 2 elements
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    std::cout << "\n";
    std::cout << "+--------------------------------------------------------+\n";
    std::cout << "|  Resolve-Once Optimization Tests                       |\n";
    std::cout << "+--------------------------------------------------------+\n\n";

    std::cout << "Phase 1: Brute-force vs Trie Equivalence\n";
    run_test("equivalence_initial_position", test_equivalence_initial_position);
    run_test("equivalence_after_space_literal", test_equivalence_after_space_literal);
    run_test("equivalence_full_decode", test_equivalence_full_decode);
    run_test("equivalence_string_content", test_equivalence_string_content);

    std::cout << "\nPhase 2: Specific Scenarios\n";
    run_test("space_literal_many_candidates", test_space_literal_many_candidates);
    run_test("exact_literal_match", test_exact_literal_match);
    run_test("multi_element_token", test_multi_element_token);
    run_test("char_class_after_literal", test_char_class_after_literal);
    run_test("partial_literal_consumption", test_partial_literal_consumption);

    std::cout << "\nPhase 3: Edge Cases\n";
    run_test("star_after_literal", test_star_after_literal);

    std::cout << "\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << " Results: " << g_passed << " passed, " << g_failed << " failed\n";
    std::cout << "--------------------------------------------------------\n";

    if (g_failed == 0)
    {
        std::cout << "\033[32m[PASS] All resolve-once tests passed!\033[0m\n\n";
        return 0;
    }
    else
    {
        std::cout << "\033[31m[FAIL] " << g_failed << " test(s) failed.\033[0m\n\n";
        return 1;
    }
}
