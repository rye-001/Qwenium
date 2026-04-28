// test_multitoken.cpp
// Multi-token literal tests adapted for qwenium::GrammarVocab.
//
// In grammar_vocab.cpp, multi-token literal handling is inherent:
//   "Promise.all" is one grammar literal; if the tokenizer splits it as
//   vocab[10]="Promise", vocab[11]=".", vocab[12]="all", then:
//     - accept("Promise")  → char_idx advances from 0 to 7 (partial match)
//     - accept(".")        → char_idx advances to 8
//     - accept("all")      → literal fully consumed, grammar advances
// No pretokenized map is needed – only the vocab is required.

#include "grammar_vocab.h"
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <stdexcept>

static int g_passed = 0;
static int g_failed = 0;

#define ASSERT_NE(a, b) if ((a) == (b)) throw std::runtime_error("ASSERT_NE failed: " #a " == " #b)
#define ASSERT_EQ(a, b) if ((a) != (b)) throw std::runtime_error("ASSERT_EQ failed: " #a " != " #b)
#define EXPECT_TRUE(x)  if (!(x))       throw std::runtime_error("EXPECT_TRUE failed: " #x)
#define EXPECT_FALSE(x) if  (x)         throw std::runtime_error("EXPECT_FALSE failed: " #x)

static void run_test(const char* name, void(*func)())
{
    std::cout << "  " << name << "... " << std::flush;
    try   { func(); std::cout << "\033[32mPASS\033[0m\n"; g_passed++; }
    catch (const std::exception& e)
          { std::cout << "\033[31mFAIL\033[0m: " << e.what() << "\n"; g_failed++; }
}

// ---------------------------------------------------------------------------
// TokenAcceptor helper – owns no grammar, just calls vocab-aware methods
// ---------------------------------------------------------------------------
class TokenAcceptor {
public:
    TokenAcceptor(qwenium::GrammarVocab* g, const std::vector<std::string>& v)
        : grammar(g), vocab(v) {}

    void accept(int32_t token, const char* desc)
    {
        auto valid = grammar->get_valid_tokens(vocab);
        std::set<int32_t> vs(valid.begin(), valid.end());
        if (!vs.count(token)) {
            std::string msg = "Token not valid at: ";
            msg += desc;
            msg += " (token " + std::to_string(token) + " = '" + vocab[token] + "')";
            msg += "\nValid: {";
            for (auto t : valid) msg += std::to_string(t) + "='" + vocab[t] + "', ";
            msg += "}";
            throw std::runtime_error(msg);
        }
        grammar->accept_token(token, vocab);
    }

    bool is_valid(int32_t token)
    {
        auto v = grammar->get_valid_tokens(vocab);
        return std::set<int32_t>(v.begin(), v.end()).count(token) > 0;
    }

    std::vector<int32_t> get_valid() { return grammar->get_valid_tokens(vocab); }

private:
    qwenium::GrammarVocab* grammar;
    const std::vector<std::string>& vocab;
};

// ============================================================================
// Grammar & vocab builders
// ============================================================================

static const char* MULTITOKEN_GRAMMAR = R"DELIM(
root ::= statement
statement ::= promise_call | simple_call
promise_call ::= "await" ws "Promise.all" ws? "(" ws? ")" ws? ";"
simple_call  ::= "await" ws func_name ws? "(" ws? ")" ws? ";"
func_name    ::= "getAccounts" | "getSupportTickets"
ws ::= " "+
)DELIM";

static std::vector<std::string> build_multitoken_vocab()
{
    std::vector<std::string> vocab(20);
    vocab[1]  = "await";
    vocab[2]  = "(";
    vocab[3]  = ")";
    vocab[4]  = ";";
    vocab[5]  = " ";
    vocab[6]  = "getAccounts";
    vocab[7]  = "getSupportTickets";
    vocab[10] = "Promise";  // first part of "Promise.all"
    vocab[11] = ".";        // second part
    vocab[12] = "all";      // third part
    return vocab;
}

// ============================================================================
// Phase 1: Basic setup
// ============================================================================
static void test_grammar_parses()
{
    auto g = qwenium::GrammarVocab::parse_impl(MULTITOKEN_GRAMMAR);
    ASSERT_NE(g, nullptr);
}

static void test_single_token_still_works()
{
    auto g = qwenium::GrammarVocab::parse_impl(MULTITOKEN_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_multitoken_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(1, "await");
    ta.accept(5, "space");
    ta.accept(6, "getAccounts");
    ta.accept(2, "(");
    ta.accept(3, ")");
    ta.accept(4, ";");
    EXPECT_TRUE(g->is_accepting_state());
}

// ============================================================================
// Phase 2: Multi-token literal validation
// ============================================================================
static void test_first_token_valid()
{
    auto g = qwenium::GrammarVocab::parse_impl(MULTITOKEN_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_multitoken_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(1, "await");
    ta.accept(5, "space");

    auto valid = ta.get_valid();
    std::set<int32_t> vs(valid.begin(), valid.end());
    EXPECT_TRUE(vs.count(10) > 0);  // "Promise" – first token of "Promise.all"
    EXPECT_TRUE(vs.count(6) > 0);   // "getAccounts"
    EXPECT_TRUE(vs.count(7) > 0);   // "getSupportTickets"
}

static void test_middle_token_valid()
{
    auto g = qwenium::GrammarVocab::parse_impl(MULTITOKEN_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_multitoken_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(1,  "await");
    ta.accept(5,  "space");
    ta.accept(10, "Promise");  // char_idx = 7 in "Promise.all"

    auto valid = ta.get_valid();
    std::set<int32_t> vs(valid.begin(), valid.end());
    EXPECT_TRUE(vs.count(11) > 0);   // "." expected next
    EXPECT_FALSE(vs.count(6) > 0);   // getAccounts not valid mid-literal
    EXPECT_FALSE(vs.count(10) > 0);  // "Promise" not valid at char_idx=7
}

static void test_last_token_valid()
{
    auto g = qwenium::GrammarVocab::parse_impl(MULTITOKEN_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_multitoken_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(1,  "await");
    ta.accept(5,  "space");
    ta.accept(10, "Promise");
    ta.accept(11, ".");

    auto valid = ta.get_valid();
    std::set<int32_t> vs(valid.begin(), valid.end());
    EXPECT_TRUE(vs.count(12) > 0);   // "all"
    EXPECT_FALSE(vs.count(11) > 0);  // "." not valid
}

static void test_full_statement()
{
    auto g = qwenium::GrammarVocab::parse_impl(MULTITOKEN_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_multitoken_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(1,  "await");
    ta.accept(5,  "space");
    ta.accept(10, "Promise");
    ta.accept(11, ".");
    ta.accept(12, "all");
    ta.accept(2,  "(");
    ta.accept(3,  ")");
    ta.accept(4,  ";");
    EXPECT_TRUE(g->is_accepting_state());
}

static void test_grammar_advances_after_completion()
{
    auto g = qwenium::GrammarVocab::parse_impl(MULTITOKEN_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_multitoken_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(1,  "await");
    ta.accept(5,  "space");
    ta.accept(10, "Promise");
    ta.accept(11, ".");
    ta.accept(12, "all");

    auto valid = ta.get_valid();
    std::set<int32_t> vs(valid.begin(), valid.end());
    EXPECT_TRUE(vs.count(2) > 0 || vs.count(5) > 0);  // "(" or optional space
}

// ============================================================================
// Phase 3: Two-token literal
// ============================================================================
static const char* TWO_TOKEN_GRAMMAR = R"DELIM(root ::= "getAccounts" "(" ")")DELIM";

static std::vector<std::string> build_two_token_vocab()
{
    std::vector<std::string> vocab(110);
    vocab[100] = "get";       // first part of "getAccounts"
    vocab[101] = "Accounts";  // second part
    vocab[102] = "(";
    vocab[103] = ")";
    return vocab;
}

static void test_two_token_literal()
{
    auto g = qwenium::GrammarVocab::parse_impl(TWO_TOKEN_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_two_token_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(100, "get");       // partial: char_idx=3 in "getAccounts"
    ta.accept(101, "Accounts");  // completes literal
    ta.accept(102, "(");
    ta.accept(103, ")");
    EXPECT_TRUE(g->is_accepting_state());
}

static void test_two_token_intermediate_state()
{
    auto g = qwenium::GrammarVocab::parse_impl(TWO_TOKEN_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_two_token_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(100, "get");
    EXPECT_FALSE(g->is_accepting_state());

    auto valid = ta.get_valid();
    ASSERT_EQ(valid.size(), 1u);
    ASSERT_EQ(valid[0], 101);   // only "Accounts" can continue
}

// ============================================================================
// Phase 4: Multi-token with alternatives (shared prefix)
// ============================================================================
static const char* ALTERNATIVES_GRAMMAR = R"DELIM(
root ::= "Promise.all" | "Promise.race" | "getAccounts"
)DELIM";

static std::vector<std::string> build_alternatives_vocab()
{
    std::vector<std::string> vocab(210);
    vocab[200] = "Promise";
    vocab[201] = ".";
    vocab[202] = "all";
    vocab[203] = "race";
    vocab[204] = "getAccounts";
    return vocab;
}

static void test_alternatives_initial_tokens()
{
    auto g = qwenium::GrammarVocab::parse_impl(ALTERNATIVES_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_alternatives_vocab();
    TokenAcceptor ta(g.get(), vocab);

    auto valid = ta.get_valid();
    std::set<int32_t> vs(valid.begin(), valid.end());
    EXPECT_TRUE(vs.count(200) > 0);  // "Promise"
    EXPECT_TRUE(vs.count(204) > 0);  // "getAccounts"
}

static void test_alternatives_shared_prefix()
{
    auto g = qwenium::GrammarVocab::parse_impl(ALTERNATIVES_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_alternatives_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(200, "Promise");
    ta.accept(201, ".");

    auto valid = ta.get_valid();
    std::set<int32_t> vs(valid.begin(), valid.end());
    EXPECT_TRUE(vs.count(202) > 0);   // "all"
    EXPECT_TRUE(vs.count(203) > 0);   // "race"
    EXPECT_FALSE(vs.count(200) > 0);  // "Promise" not valid here
}

static void test_alternatives_complete_promise_all()
{
    auto g = qwenium::GrammarVocab::parse_impl(ALTERNATIVES_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_alternatives_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(200, "Promise");
    ta.accept(201, ".");
    ta.accept(202, "all");
    EXPECT_TRUE(g->is_accepting_state());
}

static void test_alternatives_complete_promise_race()
{
    auto g = qwenium::GrammarVocab::parse_impl(ALTERNATIVES_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_alternatives_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(200, "Promise");
    ta.accept(201, ".");
    ta.accept(203, "race");
    EXPECT_TRUE(g->is_accepting_state());
}

// ============================================================================
// Phase 5: Multi-token in sequence
// ============================================================================
static const char* SEQUENCE_GRAMMAR = R"DELIM(
root ::= "await" " " "Promise.all" "(" ")"
)DELIM";

static std::vector<std::string> build_sequence_vocab()
{
    std::vector<std::string> vocab(310);
    vocab[300] = "await";
    vocab[301] = " ";
    vocab[302] = "Promise";
    vocab[303] = ".";
    vocab[304] = "all";
    vocab[305] = "(";
    vocab[306] = ")";
    return vocab;
}

static void test_multitoken_in_sequence()
{
    auto g = qwenium::GrammarVocab::parse_impl(SEQUENCE_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_sequence_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(300, "await");
    ta.accept(301, " ");
    ta.accept(302, "Promise");
    ta.accept(303, ".");
    ta.accept(304, "all");
    ta.accept(305, "(");
    ta.accept(306, ")");
    EXPECT_TRUE(g->is_accepting_state());
}

static void test_sequence_after_multitoken_first_token()
{
    auto g = qwenium::GrammarVocab::parse_impl(SEQUENCE_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_sequence_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(300, "await");
    ta.accept(301, " ");
    ta.accept(302, "Promise");
    ta.accept(303, ".");
    ta.accept(304, "all");

    auto valid = ta.get_valid();
    std::set<int32_t> vs(valid.begin(), valid.end());
    EXPECT_TRUE(vs.count(305) > 0);  // "(" must follow
}

// ============================================================================
// Phase 6: Multi-token with star repetition
// ============================================================================
static const char* REPEAT_GRAMMAR = R"DELIM(root ::= "start" (" " "Promise.all")*)DELIM";

static std::vector<std::string> build_repeat_vocab()
{
    std::vector<std::string> vocab(410);
    vocab[400] = "start";
    vocab[401] = " ";
    vocab[402] = "Promise";
    vocab[403] = ".";
    vocab[404] = "all";
    return vocab;
}

static void test_multitoken_with_star_zero()
{
    auto g = qwenium::GrammarVocab::parse_impl(REPEAT_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_repeat_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(400, "start");
    EXPECT_TRUE(g->is_accepting_state());
}

static void test_multitoken_with_star_one()
{
    auto g = qwenium::GrammarVocab::parse_impl(REPEAT_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_repeat_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(400, "start");
    ta.accept(401, " ");
    ta.accept(402, "Promise");
    ta.accept(403, ".");
    ta.accept(404, "all");
    EXPECT_TRUE(g->is_accepting_state());
}

static void test_multitoken_with_star_two()
{
    auto g = qwenium::GrammarVocab::parse_impl(REPEAT_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_repeat_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(400, "start");
    ta.accept(401, " ");
    ta.accept(402, "Promise");
    ta.accept(403, ".");
    ta.accept(404, "all");
    ta.accept(401, " ");   // second repetition
    ta.accept(402, "Promise");
    ta.accept(403, ".");
    ta.accept(404, "all");
    EXPECT_TRUE(g->is_accepting_state());
}

// ============================================================================
// Phase 7: Realistic tokenization (actual GPT/LLaMA token IDs)
// ============================================================================
static const char* REALISTIC_GRAMMAR = R"DELIM(
root ::= let_decl
let_decl    ::= "const" ws identifier ws? "=" ws? expression ws? ";"
expression  ::= function_call | variable
function_call ::= func_name ws? "(" ws? ")"
func_name   ::= "getAccounts" | "getSupportTickets" | "getOpenOrders"
variable    ::= identifier
identifier  ::= "accounts" | "account" | "tickets" | "ticket" | "orders"
ws          ::= " "+
)DELIM";

static std::vector<std::string> build_realistic_vocab()
{
    std::vector<std::string> vocab(100000);
    vocab[1024]  = "const";
    vocab[220]   = " ";
    vocab[28]    = "=";
    vocab[7]     = "(";
    vocab[8]     = ")";
    vocab[26]    = ";";
    // identifiers
    vocab[26206] = "accounts";
    vocab[4608]  = "account";
    vocab[68727] = "tickets";
    vocab[26534] = "ticket";
    vocab[7917]  = "orders";
    // "getAccounts" split as get(455) + Accounts(41369)
    vocab[455]   = "get";
    vocab[41369] = "Accounts";
    // "getSupportTickets" split as get(455) + Support(7916) + Tickets(55321)
    vocab[7916]  = "Support";
    vocab[55321] = "Tickets";
    // "getOpenOrders" split as get(455) + Open(5002) + Orders(24898)
    vocab[5002]  = "Open";
    vocab[24898] = "Orders";
    return vocab;
}

static void test_realistic_const_accounts_getAccounts()
{
    auto g = qwenium::GrammarVocab::parse_impl(REALISTIC_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_realistic_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(1024,  "const");
    ta.accept(220,   " ");
    ta.accept(26206, "accounts");
    ta.accept(220,   " ");
    ta.accept(28,    "=");
    ta.accept(220,   " ");
    ta.accept(455,   "get");
    ta.accept(41369, "Accounts");
    ta.accept(7,     "(");
    ta.accept(8,     ")");
    ta.accept(26,    ";");
    EXPECT_TRUE(g->is_accepting_state());
}

static void test_realistic_identifier_choice()
{
    auto g = qwenium::GrammarVocab::parse_impl(REALISTIC_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_realistic_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(1024, "const");
    ta.accept(220,  " ");

    auto valid = ta.get_valid();
    std::set<int32_t> vs(valid.begin(), valid.end());
    EXPECT_TRUE(vs.count(26206) > 0);  // accounts
    EXPECT_TRUE(vs.count(4608)  > 0);  // account
    EXPECT_TRUE(vs.count(68727) > 0);  // tickets
    EXPECT_TRUE(vs.count(26534) > 0);  // ticket
    EXPECT_TRUE(vs.count(7917)  > 0);  // orders
}

static void test_realistic_three_token_function()
{
    auto g = qwenium::GrammarVocab::parse_impl(REALISTIC_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_realistic_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(1024,  "const");
    ta.accept(220,   " ");
    ta.accept(68727, "tickets");
    ta.accept(220,   " ");
    ta.accept(28,    "=");
    ta.accept(220,   " ");
    ta.accept(455,   "get");
    ta.accept(7916,  "Support");
    ta.accept(55321, "Tickets");
    ta.accept(7,     "(");
    ta.accept(8,     ")");
    ta.accept(26,    ";");
    EXPECT_TRUE(g->is_accepting_state());
}

static void test_realistic_getOpenOrders()
{
    auto g = qwenium::GrammarVocab::parse_impl(REALISTIC_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_realistic_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(1024,  "const");
    ta.accept(220,   " ");
    ta.accept(7917,  "orders");
    ta.accept(220,   " ");
    ta.accept(28,    "=");
    ta.accept(220,   " ");
    ta.accept(455,   "get");
    ta.accept(5002,  "Open");
    ta.accept(24898, "Orders");
    ta.accept(7,     "(");
    ta.accept(8,     ")");
    ta.accept(26,    ";");
    EXPECT_TRUE(g->is_accepting_state());
}

static void test_realistic_function_shared_prefix()
{
    auto g = qwenium::GrammarVocab::parse_impl(REALISTIC_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_realistic_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(1024,  "const");
    ta.accept(220,   " ");
    ta.accept(26206, "accounts");
    ta.accept(220,   " ");
    ta.accept(28,    "=");
    ta.accept(220,   " ");
    ta.accept(455,   "get");  // shared first token for all three functions

    auto valid = ta.get_valid();
    std::set<int32_t> vs(valid.begin(), valid.end());
    EXPECT_TRUE(vs.count(41369) > 0);  // "Accounts" -> getAccounts
    EXPECT_TRUE(vs.count(7916)  > 0);  // "Support"  -> getSupportTickets
    EXPECT_TRUE(vs.count(5002)  > 0);  // "Open"     -> getOpenOrders
}

static void test_realistic_mid_multitoken_constraint()
{
    auto g = qwenium::GrammarVocab::parse_impl(REALISTIC_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_realistic_vocab();
    TokenAcceptor ta(g.get(), vocab);

    ta.accept(1024,  "const");
    ta.accept(220,   " ");
    ta.accept(26206, "accounts");
    ta.accept(220,   " ");
    ta.accept(28,    "=");
    ta.accept(220,   " ");
    ta.accept(455,   "get");
    ta.accept(7916,  "Support");  // committed to getSupportTickets

    auto valid = ta.get_valid();
    ASSERT_EQ(valid.size(), 1u);
    ASSERT_EQ(valid[0], 55321);  // only "Tickets"
}

// ============================================================================
// main
// ============================================================================
int main()
{
    std::cout << "\n";
    std::cout << "+--------------------------------------------------------+\n";
    std::cout << "|  grammar_vocab.cpp – Multi-Token Literal Tests         |\n";
    std::cout << "+--------------------------------------------------------+\n\n";

    std::cout << "Phase 1: Basic Setup\n";
    run_test("grammar_parses",          test_grammar_parses);
    run_test("single_token_still_works",test_single_token_still_works);

    std::cout << "\nPhase 2: Multi-Token Literal Validation\n";
    run_test("first_token_valid",                 test_first_token_valid);
    run_test("middle_token_valid",                test_middle_token_valid);
    run_test("last_token_valid",                  test_last_token_valid);
    run_test("full_statement",                    test_full_statement);
    run_test("grammar_advances_after_completion", test_grammar_advances_after_completion);

    std::cout << "\nPhase 3: Two-Token Literal\n";
    run_test("two_token_literal",           test_two_token_literal);
    run_test("two_token_intermediate_state",test_two_token_intermediate_state);

    std::cout << "\nPhase 4: Multi-Token with Alternatives (Shared Prefix)\n";
    run_test("alternatives_initial_tokens",       test_alternatives_initial_tokens);
    run_test("alternatives_shared_prefix",        test_alternatives_shared_prefix);
    run_test("alternatives_complete_promise_all", test_alternatives_complete_promise_all);
    run_test("alternatives_complete_promise_race",test_alternatives_complete_promise_race);

    std::cout << "\nPhase 5: Multi-Token in Sequence\n";
    run_test("multitoken_in_sequence",                 test_multitoken_in_sequence);
    run_test("sequence_after_multitoken_first_token",  test_sequence_after_multitoken_first_token);

    std::cout << "\nPhase 6: Multi-Token with Star Repetition\n";
    run_test("multitoken_with_star_zero", test_multitoken_with_star_zero);
    run_test("multitoken_with_star_one",  test_multitoken_with_star_one);
    run_test("multitoken_with_star_two",  test_multitoken_with_star_two);

    std::cout << "\nPhase 7: Realistic Tokenization (actual GPT-style token IDs)\n";
    run_test("realistic_const_accounts_getAccounts", test_realistic_const_accounts_getAccounts);
    run_test("realistic_identifier_choice",          test_realistic_identifier_choice);
    run_test("realistic_three_token_function",       test_realistic_three_token_function);
    run_test("realistic_getOpenOrders",              test_realistic_getOpenOrders);
    run_test("realistic_function_shared_prefix",     test_realistic_function_shared_prefix);
    run_test("realistic_mid_multitoken_constraint",  test_realistic_mid_multitoken_constraint);

    std::cout << "\n--------------------------------------------------------\n";
    std::cout << " Results: " << g_passed << " passed, " << g_failed << " failed\n";
    std::cout << "--------------------------------------------------------\n";
    if (g_failed == 0) { std::cout << "\033[32m[PASS] All tests passed!\033[0m\n\n"; return 0; }
    else               { std::cout << "\033[31m[FAIL] " << g_failed << " failed.\033[0m\n\n"; return 1; }
}
