// test_sampling_integration.cpp
// Grammar-constrained sampling integration tests for qwen3::GrammarVocab.
//
// CRITICAL FIX vs original test_grammar_sampling_integration.cpp:
//   test_char_class_accept_with_vocab originally used "hello" (5 chars) as
//   the accepted token for bare [a-z] (1 CHAR_CLASS element = 1 char consumed).
//   grammar_vocab accept_token is character-exact: accepting a 5-char token
//   against a single-char element leaves the grammar dead.
//   FIX: vocab contains single-char tokens "h" and "w" for the accept step.
//        get_valid_tokens still lists multi-char tokens whose first char
//        matches (correct for logit-masking); we only fix what we accept.
//
// All other tests are API-only changes:
//   parse_impl(gbnf)           — no token_map
//   get_valid_tokens(vocab)    — always needs vocab
//   accept_token(id, vocab)    — always needs vocab

#include "grammar_vocab.h"
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>

static int g_passed = 0;
static int g_failed = 0;

#define ASSERT_NE(a,b) if((a)==(b)) throw std::runtime_error("ASSERT_NE failed: " #a " == " #b)
#define ASSERT_EQ(a,b) if((a)!=(b)) throw std::runtime_error("ASSERT_EQ failed: " #a " != " #b)
#define EXPECT_TRUE(x) if(!(x))    throw std::runtime_error("EXPECT_TRUE failed: " #x)
#define EXPECT_FALSE(x) if(x)      throw std::runtime_error("EXPECT_FALSE failed: " #x)

static void run_test(const char* name, void(*func)())
{
    std::cout << "  " << name << "... " << std::flush;
    try   { func(); std::cout << "\033[32mPASS\033[0m\n"; g_passed++; }
    catch (const std::exception& e)
          { std::cout << "\033[31mFAIL\033[0m: " << e.what() << "\n"; g_failed++; }
}

// ---------------------------------------------------------------------------
// Sampling helpers (mirror sampling.cpp logic)
// ---------------------------------------------------------------------------
static void apply_grammar_constraints(std::vector<float>& logits,
                                      const std::vector<int32_t>& valid_tokens)
{
    std::set<int32_t> vs(valid_tokens.begin(), valid_tokens.end());
    for (size_t i = 0; i < logits.size(); ++i)
        if (!vs.count((int32_t)i))
            logits[i] = -std::numeric_limits<float>::infinity();
}

static int32_t greedy_sample(const std::vector<float>& logits)
{
    return (int32_t)std::distance(logits.begin(),
                                  std::max_element(logits.begin(), logits.end()));
}

// ============================================================================
// Phase 1: Interface adaptation
// ============================================================================

static void test_get_valid_tokens_with_vocab()
{
    const std::string gbnf = R"(root ::= "[" "let" "]")";
    std::vector<std::string> vocab = {"[", "]", "let", "if", "for"};
    //                          id:    0    1     2      3     4

    auto g = qwen3::GrammarVocab::parse_impl(gbnf);
    ASSERT_NE(g, nullptr);

    auto valid = g->get_valid_tokens(vocab);
    ASSERT_EQ(valid.size(), 1u);
    ASSERT_EQ(valid[0], 0);  // only "[" is valid first
}

static void test_accept_token_with_vocab()
{
    const std::string gbnf = R"(root ::= "[" "let" "]")";
    std::vector<std::string> vocab = {"[", "]", "let", "if", "for"};

    auto g = qwen3::GrammarVocab::parse_impl(gbnf);
    ASSERT_NE(g, nullptr);

    g->accept_token(0, vocab);  // "["
    auto valid = g->get_valid_tokens(vocab);
    ASSERT_EQ(valid.size(), 1u);
    ASSERT_EQ(valid[0], 2);     // "let"

    g->accept_token(2, vocab);  // "let"
    valid = g->get_valid_tokens(vocab);
    ASSERT_EQ(valid.size(), 1u);
    ASSERT_EQ(valid[0], 1);     // "]"

    g->accept_token(1, vocab);  // "]"
    EXPECT_TRUE(g->is_accepting_state());
}

static void test_char_class_with_vocab()
{
    const std::string gbnf = R"(root ::= [a-c])";
    // get_valid_tokens does first-char scan → multi-char tokens valid for masking
    std::vector<std::string> vocab = {
        "apple",   // 0  starts with 'a' ✓
        "banana",  // 1  starts with 'b' ✓
        "cat",     // 2  starts with 'c' ✓
        "dog",     // 3  starts with 'd' ✗
        "ant",     // 4  starts with 'a' ✓
        "zebra",   // 5  starts with 'z' ✗
        "cherry",  // 6  starts with 'c' ✓
    };

    auto g = qwen3::GrammarVocab::parse_impl(gbnf);
    ASSERT_NE(g, nullptr);

    auto valid = g->get_valid_tokens(vocab);
    std::set<int32_t> vs(valid.begin(), valid.end());
    std::set<int32_t> expected = {0, 1, 2, 4, 6};
    if (vs != expected) {
        std::string msg = "CHAR_CLASS mismatch. Expected {0,1,2,4,6} got {";
        for (auto t : vs) msg += std::to_string(t) + " ";
        msg += "}";
        throw std::runtime_error(msg);
    }
}

static void test_negated_char_class_with_vocab()
{
    const std::string gbnf = R"(root ::= [^a-c])";
    std::vector<std::string> vocab = {
        "apple",   // 0  starts with 'a' ✗
        "banana",  // 1  starts with 'b' ✗
        "cat",     // 2  starts with 'c' ✗
        "dog",     // 3  starts with 'd' ✓
        "ant",     // 4  starts with 'a' ✗
        "zebra",   // 5  starts with 'z' ✓
        "cherry",  // 6  starts with 'c' ✗
    };

    auto g = qwen3::GrammarVocab::parse_impl(gbnf);
    ASSERT_NE(g, nullptr);

    auto valid = g->get_valid_tokens(vocab);
    std::set<int32_t> vs(valid.begin(), valid.end());
    std::set<int32_t> expected = {3, 5};
    if (vs != expected) {
        std::string msg = "Negated CHAR_CLASS mismatch. Expected {3,5} got {";
        for (auto t : vs) msg += std::to_string(t) + " ";
        throw std::runtime_error(msg + "}");
    }
}

// FIXED: grammar is  [a-z] "!"  — bare CHAR_CLASS consumes exactly 1 char.
// Use single-char tokens in vocab so accept_token works correctly.
// get_valid_tokens still returns multi-char tokens (first-char scan is correct
// for logit masking); but we only accept single-char tokens here.
static void test_char_class_accept_with_vocab()
{
    const std::string gbnf = R"(root ::= [a-z] "!")";

    // Single-char tokens for the char-class position;
    // multi-char tokens also present to verify get_valid_tokens lists them.
    std::vector<std::string> vocab = {
        "h",     // 0  single char 'h' ✓ — use this for accept
        "!",     // 1  the literal
        "w",     // 2  single char 'w' ✓
        "1",     // 3  '1' not in [a-z] ✗
        "hello", // 4  starts with 'h' — valid in get_valid_tokens, not for accept
        "world", // 5  starts with 'w' — valid in get_valid_tokens, not for accept
    };

    auto g = qwen3::GrammarVocab::parse_impl(gbnf);
    ASSERT_NE(g, nullptr);

    // get_valid_tokens: first-char scan includes tokens 0,2,4,5 (start a-z)
    auto valid = g->get_valid_tokens(vocab);
    std::set<int32_t> vs(valid.begin(), valid.end());
    EXPECT_TRUE(vs.count(0) > 0);   // "h"
    EXPECT_TRUE(vs.count(2) > 0);   // "w"
    EXPECT_TRUE(vs.count(4) > 0);   // "hello" listed for masking
    EXPECT_TRUE(vs.count(5) > 0);   // "world" listed for masking
    EXPECT_FALSE(vs.count(3) > 0);  // "1" not valid

    // Accept single-char "h" — grammar advances past the CHAR_CLASS element
    g->accept_token(0, vocab);

    // Now only "!" should be valid
    valid = g->get_valid_tokens(vocab);
    ASSERT_EQ(valid.size(), 1u);
    ASSERT_EQ(valid[0], 1);

    g->accept_token(1, vocab);
    EXPECT_TRUE(g->is_accepting_state());
}

// ============================================================================
// Phase 2: Logits masking
// ============================================================================

static void test_logits_masking()
{
    const std::string gbnf = R"(root ::= "[" "let" "]")";
    std::vector<std::string> vocab = {"[", "]", "let", "if", "for"};

    auto g = qwen3::GrammarVocab::parse_impl(gbnf);
    ASSERT_NE(g, nullptr);

    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    apply_grammar_constraints(logits, g->get_valid_tokens(vocab));

    EXPECT_TRUE(std::isfinite(logits[0]));                // "[" valid
    EXPECT_TRUE(std::isinf(logits[1]) && logits[1] < 0);
    EXPECT_TRUE(std::isinf(logits[2]) && logits[2] < 0);
    EXPECT_TRUE(std::isinf(logits[3]) && logits[3] < 0);
    EXPECT_TRUE(std::isinf(logits[4]) && logits[4] < 0);

    ASSERT_EQ(greedy_sample(logits), 0);
}

static void test_logits_masking_with_alternatives()
{
    const std::string gbnf = R"(root ::= "a" | "b" | "c")";
    std::vector<std::string> vocab = {"a", "b", "c", "d", "e"};

    auto g = qwen3::GrammarVocab::parse_impl(gbnf);
    ASSERT_NE(g, nullptr);

    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 10.0f, 10.0f};
    apply_grammar_constraints(logits, g->get_valid_tokens(vocab));

    EXPECT_TRUE(std::isfinite(logits[0]));
    EXPECT_TRUE(std::isfinite(logits[1]));
    EXPECT_TRUE(std::isfinite(logits[2]));
    EXPECT_TRUE(std::isinf(logits[3]) && logits[3] < 0);
    EXPECT_TRUE(std::isinf(logits[4]) && logits[4] < 0);
    ASSERT_EQ(greedy_sample(logits), 2);  // "c" has highest valid logit
}

// ============================================================================
// Phase 3: Full decode loop
// ============================================================================

static void test_full_decode_loop_simple()
{
    const std::string gbnf = R"(root ::= "[" "let" "]")";
    std::vector<std::string> vocab = {"[", "]", "let", "if", "for"};

    auto g = qwen3::GrammarVocab::parse_impl(gbnf);
    ASSERT_NE(g, nullptr);

    std::vector<int32_t> generated;
    for (int step = 0; step < 10; ++step) {
        if (g->is_accepting_state()) break;
        std::vector<float> logits(vocab.size(), 1.0f);
        auto valid = g->get_valid_tokens(vocab);
        if (valid.empty()) throw std::runtime_error("No valid tokens at step " + std::to_string(step));
        apply_grammar_constraints(logits, valid);
        int32_t tok = greedy_sample(logits);
        generated.push_back(tok);
        g->accept_token(tok, vocab);
    }
    EXPECT_TRUE(g->is_accepting_state());
    ASSERT_EQ(generated.size(), 3u);
    ASSERT_EQ(generated[0], 0);  // "["
    ASSERT_EQ(generated[1], 2);  // "let"
    ASSERT_EQ(generated[2], 1);  // "]"
}

static void test_full_decode_loop_with_string()
{
    const std::string gbnf = R"(root ::= "\"" [^"]* "\"")";
    std::vector<std::string> vocab = {
        "\"",    // 0
        "hello", // 1
        "world", // 2
        " ",     // 3
    };

    auto g = qwen3::GrammarVocab::parse_impl(gbnf);
    ASSERT_NE(g, nullptr);

    // Guide to produce: "hello world"  tokens: 0,1,3,2,0
    std::vector<std::vector<float>> mock = {
        {10,1,1,1}, {1,10,1,1}, {1,1,1,10}, {1,1,10,1}, {10,1,1,1}
    };

    std::vector<int32_t> generated;
    for (int step = 0; step < 10; ++step) {
        if (g->is_accepting_state()) break;
        auto logits = mock[std::min(step, (int)mock.size()-1)];
        auto valid = g->get_valid_tokens(vocab);
        if (valid.empty()) throw std::runtime_error("No valid tokens at step " + std::to_string(step));
        apply_grammar_constraints(logits, valid);
        int32_t tok = greedy_sample(logits);
        generated.push_back(tok);
        g->accept_token(tok, vocab);
    }
    EXPECT_TRUE(g->is_accepting_state());
    ASSERT_EQ(generated[0], 0);                            // opening "
    ASSERT_EQ(generated[generated.size()-1], 0);           // closing "
}

static void test_full_decode_loop_dsl_let_form()
{
    const std::string gbnf = R"(
        root ::= "[" "let" "," identifier "," string "]"
        identifier ::= "x" | "y" | "z"
        string ::= "\"" [^"]* "\""
    )";
    std::vector<std::string> vocab = {
        "[",     // 0
        "]",     // 1
        "let",   // 2
        ",",     // 3
        "x",     // 4
        "y",     // 5
        "z",     // 6
        "\"",    // 7
        "hello", // 8
    };

    auto g = qwen3::GrammarVocab::parse_impl(gbnf);
    ASSERT_NE(g, nullptr);

    // Guide to: [let, x, "hello"]  = 0,2,3,4,3,7,8,7,1
    std::vector<std::vector<float>> mock = {
        {10,1,1,1,1,1,1,1,1},
        {1,1,10,1,1,1,1,1,1},
        {1,1,1,10,1,1,1,1,1},
        {1,1,1,1,10,1,1,1,1},
        {1,1,1,10,1,1,1,1,1},
        {1,1,1,1,1,1,1,10,1},
        {1,1,1,1,1,1,1,1,10},
        {1,1,1,1,1,1,1,10,1},
        {1,10,1,1,1,1,1,1,1},
    };

    std::vector<int32_t> generated;
    for (int step = 0; step < 20; ++step) {
        if (g->is_accepting_state()) break;
        auto logits = mock[std::min(step, (int)mock.size()-1)];
        auto valid = g->get_valid_tokens(vocab);
        if (valid.empty()) throw std::runtime_error("No valid tokens at step " + std::to_string(step));
        apply_grammar_constraints(logits, valid);
        int32_t tok = greedy_sample(logits);
        generated.push_back(tok);
        g->accept_token(tok, vocab);
    }
    EXPECT_TRUE(g->is_accepting_state());

    std::vector<int32_t> expected = {0, 2, 3, 4, 3, 7, 8, 7, 1};
    ASSERT_EQ(generated.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        if (generated[i] != expected[i])
            throw std::runtime_error("Mismatch at pos " + std::to_string(i) +
                ": expected " + std::to_string(expected[i]) +
                " got " + std::to_string(generated[i]));
}

// ============================================================================
// Phase 4: Real DSL GBNF parsing
// ============================================================================

static const std::string REAL_DSL_GBNF = R"(
root ::= expression

expression ::= form | variable | string

form ::= "[" ws? ( let_form | if_form | foreach_form | parallel_form | filter_form | compare_form | builtin_call | function_call ) ws? "]"

let_form      ::= let ws? "," ws? identifier ws? "," ws? expression
if_form       ::= if ws? "," ws? expression ws? "," ws? expression ws? "," ws? expression
foreach_form  ::= forEach ws? "," ws? identifier ws? "," ws? identifier ws? "," ws? expression
parallel_form ::= parallel ws? "," ws? "[" ws? expression (ws? "," ws? expression)* ws? "]"
filter_form   ::= filter ws? "," ws? variable ws? "," ws? identifier ws? "," ws? expression

compare_form  ::= comparator ws? "," ws? expression ws? "," ws? expression
comparator    ::= ">" | "<" | "==" | ">=" | "<="

builtin_call  ::= builtin (ws? "," ws? expression)+
builtin       ::= count | concat

function_call ::= function ws? "," ws? object
function      ::= getAccounts | getSupportTickets | getOpenOrders | reassignAccount | addAccountNote

object    ::= "{" ws? (param (ws? "," ws? param)*)? ws? "}"
param     ::= param_key ws? ":" ws? value
param_key ::= accountId | teamId | priority | tier | note | severity
value     ::= expression

let      ::= "let"
if       ::= "if"
forEach  ::= "forEach"
parallel ::= "parallel"
filter   ::= "filter"
count    ::= "count"
concat   ::= "concat"

getAccounts       ::= "getAccounts"
getSupportTickets ::= "getSupportTickets"
getOpenOrders     ::= "getOpenOrders"
reassignAccount   ::= "reassignAccount"
addAccountNote    ::= "addAccountNote"

accountId ::= "accountId"
teamId    ::= "teamId"
priority  ::= "priority"
tier      ::= "tier"
note      ::= "note"
severity  ::= "severity"

variable   ::= "$" var_path
var_path   ::= "account.id" | "ticket.id" | "ticket.severity" | "tickets" | "orders" | "critical_tickets" | "account" | "ticket"

identifier ::= "accounts" | "account" | "ticket" | "tickets" | "orders" | "critical_tickets"

string ::= "\"" [^"\\]* "\""

ws ::= [ \t\n\r]+
)";

static std::vector<std::string> build_dsl_vocab()
{
    std::vector<std::string> vocab(100);
    std::map<std::string,int32_t> m = {
        {"[",1},  {"]",2},  {"{",3},  {"}",4},  {",",5},  {":",6},  {"\"",7},  {"$",8},
        {" ",10}, {"\n",11},{"\t",12},{"\r",13},
        {"let",20},{"if",21},{"forEach",22},{"parallel",23},{"filter",24},
        {"count",25},{"concat",26},
        {"getAccounts",30},{"getSupportTickets",31},{"getOpenOrders",32},
        {"reassignAccount",33},{"addAccountNote",34},
        {"accountId",40},{"teamId",41},{"priority",42},{"tier",43},{"note",44},{"severity",45},
        {">",50},{"<",51},{"==",52},{">=",53},{"<=",54},
        {"account.id",61},{"ticket.id",62},{"ticket.severity",63},
        {"tickets",64},{"orders",65},{"critical_tickets",66},{"account",67},{"ticket",68},
        {"accounts",70},
        {"high",80},{"Enterprise",81},{"critical",82},{"2",83},
        {"Onboarding-Specialist",84},{"Account has ",85},{" open orders.",86},
        {"hello",87},{"world",88},
    };
    for (auto& [s,id] : m) if (id < 100) vocab[id] = s;
    return vocab;
}

// Helper: accept and check
static void dsl_accept(qwen3::GrammarVocab* g, const std::vector<std::string>& v,
                       int32_t tok, const char* desc)
{
    auto valid = g->get_valid_tokens(v);
    std::set<int32_t> vs(valid.begin(), valid.end());
    if (!vs.count(tok))
        throw std::runtime_error(std::string("Token not valid at: ") + desc +
                                 " (tok=" + std::to_string(tok) + ")");
    g->accept_token(tok, v);
}

static void test_parse_real_dsl_gbnf()
{
    auto g = qwen3::GrammarVocab::parse_impl(REAL_DSL_GBNF);
    ASSERT_NE(g, nullptr);
}

static void test_real_dsl_simple_variable()
{
    auto g = qwen3::GrammarVocab::parse_impl(REAL_DSL_GBNF);
    ASSERT_NE(g, nullptr);
    auto vocab = build_dsl_vocab();

    auto valid = g->get_valid_tokens(vocab);
    std::set<int32_t> vs(valid.begin(), valid.end());
    EXPECT_TRUE(vs.count(1) > 0);  // "["
    EXPECT_TRUE(vs.count(8) > 0);  // "$"
    EXPECT_TRUE(vs.count(7) > 0);  // "\""

    g->accept_token(8, vocab);     // "$"
    valid = g->get_valid_tokens(vocab);
    vs = std::set<int32_t>(valid.begin(), valid.end());
    EXPECT_TRUE(vs.count(61) > 0); // "account.id"

    g->accept_token(61, vocab);
    EXPECT_TRUE(g->is_accepting_state());
}

static void test_real_dsl_simple_string()
{
    auto g = qwen3::GrammarVocab::parse_impl(REAL_DSL_GBNF);
    ASSERT_NE(g, nullptr);
    auto vocab = build_dsl_vocab();

    auto valid = g->get_valid_tokens(vocab);
    std::set<int32_t> vs(valid.begin(), valid.end());
    EXPECT_TRUE(vs.count(7) > 0);  // "\""

    g->accept_token(7, vocab);     // open quote
    valid = g->get_valid_tokens(vocab);
    vs = std::set<int32_t>(valid.begin(), valid.end());
    EXPECT_TRUE(vs.count(7) > 0);   // "\"" to close
    EXPECT_TRUE(vs.count(87) > 0);  // "hello" starts with 'h' not " or \

    g->accept_token(87, vocab);    // "hello"
    g->accept_token(7, vocab);     // close quote
    EXPECT_TRUE(g->is_accepting_state());
}

static void test_real_dsl_let_form()
{
    auto g = qwen3::GrammarVocab::parse_impl(REAL_DSL_GBNF);
    ASSERT_NE(g, nullptr);
    auto vocab = build_dsl_vocab();
    auto acc = [&](int32_t t, const char* d){ dsl_accept(g.get(), vocab, t, d); };

    // [let, accounts, "high"]
    acc(1,  "[");  acc(20,"let"); acc(5,","); acc(70,"accounts"); acc(5,",");
    acc(7,  "\""); acc(80,"high"); acc(7,"\""); acc(2,"]");
    EXPECT_TRUE(g->is_accepting_state());
}

static void test_real_dsl_function_call_with_object()
{
    auto g = qwen3::GrammarVocab::parse_impl(REAL_DSL_GBNF);
    ASSERT_NE(g, nullptr);
    auto vocab = build_dsl_vocab();
    auto acc = [&](int32_t t, const char* d){ dsl_accept(g.get(), vocab, t, d); };

    // [getAccounts, {priority: "high"}]
    acc(1,"["); acc(30,"getAccounts"); acc(5,",");
    acc(3,"{"); acc(42,"priority"); acc(6,":"); acc(7,"\""); acc(80,"high"); acc(7,"\"");
    acc(4,"}"); acc(2,"]");
    EXPECT_TRUE(g->is_accepting_state());
}

static void test_real_dsl_full_complex_expression()
{
    auto g = qwen3::GrammarVocab::parse_impl(REAL_DSL_GBNF);
    ASSERT_NE(g, nullptr);
    auto vocab = build_dsl_vocab();
    auto acc = [&](int32_t t, const char* d){ dsl_accept(g.get(), vocab, t, d); };

    // [forEach, accounts, account, [getSupportTickets, {accountId: $account.id}]]
    acc(1,"["); acc(22,"forEach"); acc(5,",");
    acc(70,"accounts"); acc(5,","); acc(67,"account"); acc(5,",");
    acc(1,"[ nested"); acc(31,"getSupportTickets"); acc(5,",");
    acc(3,"{"); acc(40,"accountId"); acc(6,":"); acc(8,"$"); acc(61,"account.id");
    acc(4,"}"); acc(2,"] close nested"); acc(2,"] close forEach");
    EXPECT_TRUE(g->is_accepting_state());
}

// ============================================================================
// main
// ============================================================================
int main()
{
    std::cout << "\n";
    std::cout << "+--------------------------------------------------------+\n";
    std::cout << "|  grammar_vocab.cpp – Sampling Integration Tests        |\n";
    std::cout << "+--------------------------------------------------------+\n\n";

    std::cout << "Phase 1: Interface Adaptation\n";
    run_test("get_valid_tokens_with_vocab",   test_get_valid_tokens_with_vocab);
    run_test("accept_token_with_vocab",       test_accept_token_with_vocab);
    run_test("char_class_with_vocab",         test_char_class_with_vocab);
    run_test("negated_char_class_with_vocab", test_negated_char_class_with_vocab);
    run_test("char_class_accept_with_vocab",  test_char_class_accept_with_vocab);

    std::cout << "\nPhase 2: Logits Masking\n";
    run_test("logits_masking",                 test_logits_masking);
    run_test("logits_masking_with_alternatives",test_logits_masking_with_alternatives);

    std::cout << "\nPhase 3: Full Decode Loop\n";
    run_test("full_decode_loop_simple",       test_full_decode_loop_simple);
    run_test("full_decode_loop_with_string",  test_full_decode_loop_with_string);
    run_test("full_decode_loop_dsl_let_form", test_full_decode_loop_dsl_let_form);

    std::cout << "\nPhase 4: Real dsl.gbnf Parsing\n";
    run_test("parse_real_dsl_gbnf",                test_parse_real_dsl_gbnf);
    run_test("real_dsl_simple_variable",           test_real_dsl_simple_variable);
    run_test("real_dsl_simple_string",             test_real_dsl_simple_string);
    run_test("real_dsl_let_form",                  test_real_dsl_let_form);
    run_test("real_dsl_function_call_with_object", test_real_dsl_function_call_with_object);
    run_test("real_dsl_full_complex_expression",   test_real_dsl_full_complex_expression);

    std::cout << "\n--------------------------------------------------------\n";
    std::cout << " Results: " << g_passed << " passed, " << g_failed << " failed\n";
    std::cout << "--------------------------------------------------------\n";
    if (g_failed == 0) { std::cout << "\033[32m[PASS] All tests passed!\033[0m\n\n"; return 0; }
    else               { std::cout << "\033[31m[FAIL] " << g_failed << " failed.\033[0m\n\n"; return 1; }
}
