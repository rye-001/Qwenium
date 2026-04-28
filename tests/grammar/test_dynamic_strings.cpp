// test_dynamic_strings.cpp
// Verify that dynamic string content (like "3327") works with GrammarVocab.
//
// Use case:
//   User: get an account with Id 3327
//   Assistant: await getAccount({ accountId: "3327" });
//
// The string "3327" is NOT a grammar literal - it comes from user input.
// The grammar uses [^"]* to match any non-quote characters.

#include "grammar_vocab.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <set>
#include <string>
#include <stdexcept>

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

// ============================================================================
// Grammar with character class for dynamic strings
// ============================================================================

const char *DYNAMIC_STRING_GRAMMAR = R"DELIM(
root ::= await_call

await_call ::= "await" " " function_call ";"

function_call ::= "getAccount" "(" object ")"

object ::= "{" " "? param " "? "}"

param ::= "accountId" ":" " "? string

string ::= "\"" string_content "\""

string_content ::= [^"]*
)DELIM";

std::vector<std::string> build_dynamic_string_vocab()
{
    std::vector<std::string> vocab(200);
    vocab[1] = "await";
    vocab[2] = " ";
    vocab[3] = "getAccount";
    vocab[4] = "(";
    vocab[5] = ")";
    vocab[6] = "{";
    vocab[7] = "}";
    vocab[8] = "accountId";
    vocab[9] = ":";
    vocab[10] = "\"";
    vocab[11] = ";";
    vocab[100] = "3327";
    vocab[101] = "12345";
    vocab[102] = "hello";
    vocab[103] = "Urgent follow-up";
    return vocab;
}

// Helper for accepting tokens with good error messages
class TokenAcceptor
{
public:
    TokenAcceptor(qwenium::GrammarVocab *g, const std::vector<std::string> &v)
        : grammar(g), vocab(v) {}

    void accept(int32_t token, const char *desc)
    {
        auto valid = grammar->get_valid_tokens(vocab);
        std::set<int32_t> valid_set(valid.begin(), valid.end());
        if (valid_set.count(token) == 0)
        {
            std::string msg = "Token not valid at: ";
            msg += desc;
            msg += " (token " + std::to_string(token) + " = '" + vocab[token] + "')";
            msg += "\nValid tokens: {";
            for (auto t : valid)
            {
                msg += std::to_string(t) + "='" + vocab[t] + "', ";
            }
            msg += "}";
            throw std::runtime_error(msg);
        }
        grammar->accept_token(token, vocab);
    }

    bool is_valid(int32_t token)
    {
        auto valid = grammar->get_valid_tokens(vocab);
        std::set<int32_t> valid_set(valid.begin(), valid.end());
        return valid_set.count(token) > 0;
    }

    std::vector<int32_t> get_valid()
    {
        return grammar->get_valid_tokens(vocab);
    }

private:
    qwenium::GrammarVocab *grammar;
    const std::vector<std::string> &vocab;
};

// ============================================================================
// Test 1: Grammar parses successfully
// ============================================================================

void test_dynamic_grammar_parses()
{
    auto grammar = qwenium::GrammarVocab::parse_impl(DYNAMIC_STRING_GRAMMAR);
    ASSERT_NE(grammar, nullptr);
}

// ============================================================================
// Test 2: Can we reach the string content position?
// await getAccount({ accountId: "
// ============================================================================

void test_reach_string_content()
{
    auto grammar = qwenium::GrammarVocab::parse_impl(DYNAMIC_STRING_GRAMMAR);
    ASSERT_NE(grammar, nullptr);

    auto vocab = build_dynamic_string_vocab();
    TokenAcceptor accept(grammar.get(), vocab);

    // await getAccount({ accountId: "
    accept.accept(1, "await");
    accept.accept(2, " ");
    accept.accept(3, "getAccount");
    accept.accept(4, "(");
    accept.accept(6, "{");
    accept.accept(2, " ");
    accept.accept(8, "accountId");
    accept.accept(9, ":");
    accept.accept(2, " ");
    accept.accept(10, "\"");  // Opening quote

    // We've reached the string content position
    // The grammar is now at: string_content ::= [^"]*
    auto valid = accept.get_valid();
    // Should have at least the closing quote and some dynamic content tokens
    EXPECT_TRUE(valid.size() > 0);
}

// ============================================================================
// Test 3: Is "3327" valid as string content?
// This is the KEY test - can the LLM generate dynamic content?
// ============================================================================

void test_dynamic_string_content_valid()
{
    auto grammar = qwenium::GrammarVocab::parse_impl(DYNAMIC_STRING_GRAMMAR);
    ASSERT_NE(grammar, nullptr);

    auto vocab = build_dynamic_string_vocab();
    TokenAcceptor accept(grammar.get(), vocab);

    // await getAccount({ accountId: "
    accept.accept(1, "await");
    accept.accept(2, " ");
    accept.accept(3, "getAccount");
    accept.accept(4, "(");
    accept.accept(6, "{");
    accept.accept(2, " ");
    accept.accept(8, "accountId");
    accept.accept(9, ":");
    accept.accept(2, " ");
    accept.accept(10, "\"");

    // NOW: Is "3327" (token 100) valid here?
    // The grammar says [^"]* - any non-quote character
    // "3327" starts with '3' which is not '"', so it SHOULD be valid
    EXPECT_TRUE(accept.is_valid(100));  // "3327" should be valid
}

// ============================================================================
// Test 4: Full statement with dynamic content
// await getAccount({ accountId: "3327" });
// ============================================================================

void test_full_dynamic_string_statement()
{
    auto grammar = qwenium::GrammarVocab::parse_impl(DYNAMIC_STRING_GRAMMAR);
    ASSERT_NE(grammar, nullptr);

    auto vocab = build_dynamic_string_vocab();
    TokenAcceptor accept(grammar.get(), vocab);

    // await getAccount({ accountId: "3327" });
    accept.accept(1, "await");
    accept.accept(2, " ");
    accept.accept(3, "getAccount");
    accept.accept(4, "(");
    accept.accept(6, "{");
    accept.accept(2, " ");
    accept.accept(8, "accountId");
    accept.accept(9, ":");
    accept.accept(2, " ");
    accept.accept(10, "\"");
    accept.accept(100, "3327");    // Dynamic content!
    accept.accept(10, "\"");       // Closing quote
    accept.accept(2, " ");
    accept.accept(7, "}");
    accept.accept(5, ")");
    accept.accept(11, ";");

    EXPECT_TRUE(grammar->is_accepting_state());
}

// ============================================================================
// Test 5: After accepting dynamic content, closing quote should be valid
// ============================================================================

void test_closing_quote_valid_after_content()
{
    auto grammar = qwenium::GrammarVocab::parse_impl(DYNAMIC_STRING_GRAMMAR);
    ASSERT_NE(grammar, nullptr);

    auto vocab = build_dynamic_string_vocab();
    TokenAcceptor accept(grammar.get(), vocab);

    // await getAccount({ accountId: "3327
    accept.accept(1, "await");
    accept.accept(2, " ");
    accept.accept(3, "getAccount");
    accept.accept(4, "(");
    accept.accept(6, "{");
    accept.accept(2, " ");
    accept.accept(8, "accountId");
    accept.accept(9, ":");
    accept.accept(2, " ");
    accept.accept(10, "\"");
    accept.accept(100, "3327");

    // After "3327", the closing quote should be valid
    // Because [^"]* can match zero or more, and we need to continue to "\""
    EXPECT_TRUE(accept.is_valid(10));  // Closing quote should be valid
}

// ============================================================================
// Test 6: Multiple tokens of dynamic content
// await getAccount({ accountId: "hello world" });
// where "hello" and " world" are separate tokens
// ============================================================================

void test_multi_token_dynamic_content()
{
    std::vector<std::string> vocab(200);
    vocab[1] = "await";
    vocab[2] = " ";
    vocab[3] = "getAccount";
    vocab[4] = "(";
    vocab[5] = ")";
    vocab[6] = "{";
    vocab[7] = "}";
    vocab[8] = "accountId";
    vocab[9] = ":";
    vocab[10] = "\"";
    vocab[11] = ";";
    vocab[102] = "hello";
    vocab[104] = " world";

    auto grammar = qwenium::GrammarVocab::parse_impl(DYNAMIC_STRING_GRAMMAR);
    ASSERT_NE(grammar, nullptr);

    TokenAcceptor accept(grammar.get(), vocab);

    // await getAccount({ accountId: "hello world" });
    accept.accept(1, "await");
    accept.accept(2, " ");
    accept.accept(3, "getAccount");
    accept.accept(4, "(");
    accept.accept(6, "{");
    accept.accept(2, " ");
    accept.accept(8, "accountId");
    accept.accept(9, ":");
    accept.accept(2, " ");
    accept.accept(10, "\"");
    accept.accept(102, "hello");      // First part of string
    accept.accept(104, " world");     // Second part of string (with [^"]* this should work)
    accept.accept(10, "\"");
    accept.accept(2, " ");
    accept.accept(7, "}");
    accept.accept(5, ")");
    accept.accept(11, ";");

    EXPECT_TRUE(grammar->is_accepting_state());
}

// ============================================================================
// Test 7: Verify star operator allows MULTIPLE tokens
// After accepting "hello", can we accept ANOTHER token before closing quote?
// ============================================================================

void test_star_allows_continuation()
{
    std::vector<std::string> vocab(200);
    vocab[1] = "await";
    vocab[2] = " ";
    vocab[3] = "getAccount";
    vocab[4] = "(";
    vocab[5] = ")";
    vocab[6] = "{";
    vocab[7] = "}";
    vocab[8] = "accountId";
    vocab[9] = ":";
    vocab[10] = "\"";
    vocab[11] = ";";
    vocab[102] = "hello";
    vocab[105] = "world";

    auto grammar = qwenium::GrammarVocab::parse_impl(DYNAMIC_STRING_GRAMMAR);

    TokenAcceptor accept(grammar.get(), vocab);

    accept.accept(1, "await");
    accept.accept(2, " ");
    accept.accept(3, "getAccount");
    accept.accept(4, "(");
    accept.accept(6, "{");
    accept.accept(2, " ");
    accept.accept(8, "accountId");
    accept.accept(9, ":");
    accept.accept(2, " ");
    accept.accept(10, "\"");
    accept.accept(102, "hello");  // First token of string content

    // KEY CHECK: After "hello", is "world" STILL valid?
    // If [^"]* works correctly, we should be able to continue adding content
    auto valid = accept.get_valid();
    std::set<int32_t> valid_set(valid.begin(), valid.end());

    // Both "world" (more content) and "\"" (close string) should be valid
    EXPECT_TRUE(valid_set.count(105) > 0);  // "world" - continue string
    EXPECT_TRUE(valid_set.count(10) > 0);   // "\"" - close string
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    std::cout << "\n";
    std::cout << "+--------------------------------------------------------+\n";
    std::cout << "|  Dynamic String Content Tests                          |\n";
    std::cout << "+--------------------------------------------------------+\n\n";

    std::cout << "Phase 1: Grammar Setup\n";
    run_test("grammar_parses", test_dynamic_grammar_parses);
    run_test("reach_string_content", test_reach_string_content);

    std::cout << "\nPhase 2: Dynamic Content Validation\n";
    run_test("dynamic_string_content_valid", test_dynamic_string_content_valid);
    run_test("closing_quote_valid_after_content", test_closing_quote_valid_after_content);
    run_test("full_dynamic_string_statement", test_full_dynamic_string_statement);

    std::cout << "\nPhase 3: Multi-Token Dynamic Content\n";
    run_test("multi_token_dynamic_content", test_multi_token_dynamic_content);
    run_test("star_allows_continuation", test_star_allows_continuation);

    std::cout << "\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << " Results: " << g_passed << " passed, " << g_failed << " failed\n";
    std::cout << "--------------------------------------------------------\n";

    if (g_failed == 0)
    {
        std::cout << "\033[32m[PASS] All dynamic string tests passed!\033[0m\n\n";
        return 0;
    }
    else
    {
        std::cout << "\033[31m[FAIL] " << g_failed << " test(s) failed.\033[0m\n\n";
        return 1;
    }
}
