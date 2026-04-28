// test_js_subset.cpp
// JS-subset grammar integration tests adapted for qwenium::GrammarVocab.
//
// API changes from test_js_subset_grammar.cpp:
//   parse_impl(gbnf_str)        — no token_map argument
//   get_valid_tokens(vocab)     — always needs vocab
//   accept_token(id, vocab)     — always needs vocab
// Everything else is identical to the original tests.

#include "grammar_vocab.h"
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>

static int g_passed = 0;
static int g_failed = 0;

#define ASSERT_NE(a,b) if((a)==(b)) throw std::runtime_error("ASSERT_NE failed: " #a " == " #b)
#define EXPECT_TRUE(x) if(!(x))    throw std::runtime_error("EXPECT_TRUE failed: " #x)

static void run_test(const char* name, void(*func)())
{
    std::cout << "  " << name << "... " << std::flush;
    try   { func(); std::cout << "\033[32mPASS\033[0m\n"; g_passed++; }
    catch (const std::exception& e)
          { std::cout << "\033[31mFAIL\033[0m: " << e.what() << "\n"; g_failed++; }
}

// ============================================================================
// Grammar
// ============================================================================

static const char* JS_SUBSET_GRAMMAR = R"DELIM(
root ::= statement

statement ::= for_of | let_decl | if_else | parallel_block | await_call

for_of ::= "for" " " "(" "const" " " identifier " " "of" " " identifier ")" " "? block

let_decl ::= "const" " " identifier " "? "=" " "? expression " "? ";"

if_else ::= "if" " "? "(" " "? condition " "? ")" " "? block " "? "else" " "? block

parallel_block ::= "await" " " "Promise.all" " "? "(" " "? "[" " "? await_call_list " "? "]" " "? ")" " "? ";"

await_call ::= "await" " " function_call " "? ";"

await_call_list ::= await_call_item (" "? "," " "? await_call_item)*
await_call_item ::= "await" " " function_call

block ::= "{" " "? statement* " "? "}"

expression ::= await_expr | count_expr | variable | string

await_expr ::= "await" " " function_call

count_expr ::= identifier "." "length"

condition ::= expression " "? comparator " "? expression
comparator ::= ">" | "<" | "===" | ">=" | "<="

function_call ::= function_name " "? "(" " "? object? " "? ")"
function_name ::= "getAccounts" | "getSupportTickets" | "getOpenOrders" | "reassignAccount" | "addAccountNote"

object ::= "{" " "? (param (" "? "," " "? param)*)? " "? "}"
param ::= param_key " "? ":" " "? param_value
param_key ::= "accountId" | "teamId" | "priority" | "tier" | "note" | "severity"
param_value ::= expression

variable ::= identifier ("." identifier)*
identifier ::= "accounts" | "account" | "ticket" | "tickets" | "orders" | "critical_tickets" | "id" | "severity" | "length"

string ::= "\"" string_content "\""
string_content ::= "high" | "Enterprise" | "Specialist" | "critical" | "2" | ""
)DELIM";

// ============================================================================
// Vocab builder
// ============================================================================

static std::vector<std::string> build_js_vocab()
{
    std::map<std::string,int32_t> m = {
        {"for",1},{"const",2},{"of",3},{"if",4},{"else",5},{"await",6},{"Promise.all",7},
        {"(",10},{")",11},{"{",12},{"}",13},{"[",14},{"]",15},{";",16},{"=",17},{",",18},
        {":",19},{".",20},{"\"",21},
        {">",30},{"<",31},{"===",32},{">=",33},{"<=",34},
        {"getAccounts",40},{"getSupportTickets",41},{"getOpenOrders",42},
        {"reassignAccount",43},{"addAccountNote",44},
        {"accountId",50},{"teamId",51},{"priority",52},{"tier",53},{"note",54},{"severity",55},
        {"accounts",60},{"account",61},{"ticket",62},{"tickets",63},
        {"orders",64},{"critical_tickets",65},{"id",66},{"length",67},
        {" ",70},
        {"high",80},{"Enterprise",81},{"Specialist",82},{"critical",83},{"2",84},{"",85},
    };
    size_t max_id = 0;
    for (auto& [s,id] : m) if ((size_t)id > max_id) max_id = id;
    std::vector<std::string> vocab(max_id + 1);
    for (auto& [s,id] : m) vocab[id] = s;
    return vocab;
}

// Helper: accept and verify token is in valid set
class Acc {
public:
    Acc(qwenium::GrammarVocab* g, const std::vector<std::string>& v) : g(g), v(v) {}
    void operator()(int32_t tok, const char* desc) {
        auto valid = g->get_valid_tokens(v);
        std::set<int32_t> vs(valid.begin(), valid.end());
        if (!vs.count(tok)) {
            std::string msg = "Token not valid at: "; msg += desc;
            msg += " (tok=" + std::to_string(tok);
            if (tok < (int32_t)v.size()) msg += " '" + v[tok] + "'";
            msg += ") valid={";
            for (auto t : valid) {
                msg += std::to_string(t) + "='";
                if (t < (int32_t)v.size()) msg += v[t];
                msg += "' ";
            }
            msg += "}";
            throw std::runtime_error(msg);
        }
        g->accept_token(tok, v);
    }
private:
    qwenium::GrammarVocab* g;
    const std::vector<std::string>& v;
};

// ============================================================================
// Tests
// ============================================================================

static void test_js_grammar_parses()
{
    auto g = qwenium::GrammarVocab::parse_impl(JS_SUBSET_GRAMMAR);
    ASSERT_NE(g, nullptr);
}

// await getAccounts({ priority: "high" });
static void test_js_await_call()
{
    auto g = qwenium::GrammarVocab::parse_impl(JS_SUBSET_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_js_vocab();
    Acc acc(g.get(), vocab);

    acc(6,"await"); acc(70,"space"); acc(40,"getAccounts");
    acc(10,"("); acc(12,"{"); acc(70,"space");
    acc(52,"priority"); acc(19,":"); acc(70,"space");
    acc(21,"open-quote"); acc(80,"high"); acc(21,"close-quote");
    acc(70,"space"); acc(13,"}"); acc(11,")"); acc(16,";");
    EXPECT_TRUE(g->is_accepting_state());
}

// await getAccounts();  (no args)
static void test_js_await_call_2()
{
    auto g = qwenium::GrammarVocab::parse_impl(JS_SUBSET_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_js_vocab();
    Acc acc(g.get(), vocab);

    acc(2,"const"); acc(70,"space"); acc(60,"accounts");
    acc(17,"="); acc(70,"space");
    acc(6,"await"); acc(70,"space"); acc(40,"getAccounts");
    acc(10,"("); acc(11,")"); acc(16,";");
    EXPECT_TRUE(g->is_accepting_state());
}

// const tickets = await getSupportTickets({ accountId: account.id });
static void test_js_let_decl()
{
    auto g = qwenium::GrammarVocab::parse_impl(JS_SUBSET_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_js_vocab();
    Acc acc(g.get(), vocab);

    acc(2,"const"); acc(70,"space"); acc(63,"tickets"); acc(70,"space");
    acc(17,"="); acc(70,"space");
    acc(6,"await"); acc(70,"space"); acc(41,"getSupportTickets");
    acc(10,"("); acc(12,"{"); acc(70,"space");
    acc(50,"accountId"); acc(19,":"); acc(70,"space");
    acc(61,"account"); acc(20,"."); acc(66,"id");
    acc(70,"space"); acc(13,"}"); acc(11,")"); acc(16,";");
    EXPECT_TRUE(g->is_accepting_state());
}

// for (const account of accounts) { await getAccounts({}); }
static void test_js_for_of()
{
    auto g = qwenium::GrammarVocab::parse_impl(JS_SUBSET_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_js_vocab();
    Acc acc(g.get(), vocab);

    acc(1,"for"); acc(70,"space"); acc(10,"(");
    acc(2,"const"); acc(70,"space"); acc(61,"account"); acc(70,"space");
    acc(3,"of"); acc(70,"space"); acc(60,"accounts"); acc(11,")");
    acc(70,"space"); acc(12,"{"); acc(70,"space");
    acc(6,"await"); acc(70,"space"); acc(40,"getAccounts");
    acc(10,"("); acc(12,"{"); acc(13,"}"); acc(11,")");
    acc(16,";"); acc(70,"space"); acc(13,"}");
    EXPECT_TRUE(g->is_accepting_state());
}

// if (tickets.length > "2") { await reassignAccount({}); } else { await addAccountNote({}); }
static void test_js_if_else()
{
    auto g = qwenium::GrammarVocab::parse_impl(JS_SUBSET_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_js_vocab();
    Acc acc(g.get(), vocab);

    acc(4,"if"); acc(70,"space"); acc(10,"(");
    acc(63,"tickets"); acc(20,"."); acc(67,"length");
    acc(70,"space"); acc(30,">"); acc(70,"space");
    acc(21,"open-quote"); acc(84,"2"); acc(21,"close-quote");
    acc(11,")"); acc(70,"space"); acc(12,"{"); acc(70,"space");
    acc(6,"await"); acc(70,"space"); acc(43,"reassignAccount");
    acc(10,"("); acc(12,"{"); acc(13,"}"); acc(11,")"); acc(16,";");
    acc(70,"space"); acc(13,"}"); acc(70,"space");
    acc(5,"else"); acc(70,"space"); acc(12,"{"); acc(70,"space");
    acc(6,"await"); acc(70,"space"); acc(44,"addAccountNote");
    acc(10,"("); acc(12,"{"); acc(13,"}"); acc(11,")"); acc(16,";");
    acc(70,"space"); acc(13,"}");
    EXPECT_TRUE(g->is_accepting_state());
}

// await Promise.all([ await getAccounts({}), await getSupportTickets({}) ]);
static void test_js_promise_all()
{
    auto g = qwenium::GrammarVocab::parse_impl(JS_SUBSET_GRAMMAR);
    ASSERT_NE(g, nullptr);
    auto vocab = build_js_vocab();
    Acc acc(g.get(), vocab);

    acc(6,"await"); acc(70,"space"); acc(7,"Promise.all");
    acc(10,"("); acc(14,"["); acc(70,"space");
    acc(6,"await"); acc(70,"space"); acc(40,"getAccounts");
    acc(10,"("); acc(12,"{"); acc(13,"}"); acc(11,")");
    acc(18,","); acc(70,"space");
    acc(6,"await"); acc(70,"space"); acc(41,"getSupportTickets");
    acc(10,"("); acc(12,"{"); acc(13,"}"); acc(11,")");
    acc(70,"space"); acc(15,"]"); acc(11,")"); acc(16,";");
    EXPECT_TRUE(g->is_accepting_state());
}

// ============================================================================
// main
// ============================================================================
int main()
{
    std::cout << "\n";
    std::cout << "+--------------------------------------------------------+\n";
    std::cout << "|  grammar_vocab.cpp – JS Subset Grammar Tests           |\n";
    std::cout << "+--------------------------------------------------------+\n\n";

    std::cout << "Phase 1: Grammar Parsing\n";
    run_test("js_grammar_parses", test_js_grammar_parses);

    std::cout << "\nPhase 2: Statement Types\n";
    run_test("js_await_call",   test_js_await_call);
    run_test("js_await_call_2", test_js_await_call_2);
    run_test("js_let_decl",     test_js_let_decl);
    run_test("js_for_of",       test_js_for_of);
    run_test("js_if_else",      test_js_if_else);
    run_test("js_promise_all",  test_js_promise_all);

    std::cout << "\n--------------------------------------------------------\n";
    std::cout << " Results: " << g_passed << " passed, " << g_failed << " failed\n";
    std::cout << "--------------------------------------------------------\n";
    if (g_failed == 0) { std::cout << "\033[32m[PASS] All tests passed!\033[0m\n\n"; return 0; }
    else               { std::cout << "\033[31m[FAIL] " << g_failed << " failed.\033[0m\n\n"; return 1; }
}
