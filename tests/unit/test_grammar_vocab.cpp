// test_grammar_vocab.cpp
// TDD for GrammarVocab (Option B — vocab-scan, no pretokenized literals map).
// Compile: g++ -std=c++17 -o test_grammar_vocab test_grammar_vocab.cpp grammar_vocab.cpp -I.
//
// RED baseline: grammar_vocab.cpp does not exist yet — tests will fail to link.
// GREEN target: all 25 tests pass after grammar_vocab.cpp is implemented.

#include "../../src/sampling/grammar_vocab.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// Minimal test framework
// ============================================================================

static int g_pass = 0, g_fail = 0;

#define TEST(name, body) \
    do { \
        std::cout << "  [ RUN  ] " << (name) << std::endl; \
        try { body g_pass++; std::cout << "  [ PASS ] " << (name) << std::endl; } \
        catch (const std::exception& _e) { \
            g_fail++; std::cout << "  [ FAIL ] " << (name) << "\n           => " << _e.what() << std::endl; } \
    } while(0)

#define ASSERT_TRUE(c)  if (!(c)) throw std::runtime_error("ASSERT_TRUE: " #c)
#define ASSERT_FALSE(c) if ((c))  throw std::runtime_error("ASSERT_FALSE: " #c " (expected false)")
#define ASSERT_IN(v, val) \
    if (std::find((v).begin(),(v).end(),(val))==(v).end()) \
        throw std::runtime_error(std::string("ASSERT_IN: id=") + std::to_string(val) + " not in valid set")
#define ASSERT_NOT_IN(v, val) \
    if (std::find((v).begin(),(v).end(),(val))!=(v).end()) \
        throw std::runtime_error(std::string("ASSERT_NOT_IN: id=") + std::to_string(val) + " unexpectedly in valid set")

// ============================================================================
// Vocab builder
// We assign indices deliberately so tests can reason about exact IDs.
//
// Design rule: vocab[i] is the *decoded* string of token i.
// BPE reality is modelled by having both bare and space-prefixed variants
// as distinct token IDs — exactly as a real tokenizer would.
// ============================================================================

enum TokId : int32_t {
    // keywords — bare
    T_CONST = 0, T_FOR, T_OF, T_IF, T_ELSE, T_AWAIT,
    // keywords — space-prefixed (single BPE token in real tokenizers)
    T_SP_CONST, T_SP_FOR, T_SP_OF, T_SP_IF, T_SP_ELSE, T_SP_AWAIT,
    // punctuation
    T_LPAREN, T_RPAREN, T_LBRACE, T_RBRACE, T_SEMI, T_DOT, T_COMMA,
    T_COLON, T_EQ, T_GT, T_LT, T_GTE, T_LTE, T_TRIPLE_EQ,
    // whitespace
    T_SPACE, T_NEWLINE,
    // quote
    T_QUOTE,
    // function names — single token
    T_GETACCOUNTS, T_GETSUPPORTTICKETS, T_GETOPENORDERS,
    T_REASSIGNACCOUNT, T_ADDACCOUNTNOTE,
    // function names — space-prefixed
    T_SP_GETACCOUNTS,
    // function names — deliberately split across two tokens (key Option B test)
    T_GET,        // "get"    — partial prefix of "getAccounts"
    T_ACCOUNTS_S, // "Accounts" — completion suffix
    // param keys
    T_ACCOUNTID, T_TEAMID, T_PRIORITY, T_TIER, T_NOTE, T_SEVERITY,
    // identifiers
    T_ACCOUNTS, T_ACCOUNT, T_TICKET, T_TICKETS, T_ORDERS,
    T_CRITICAL_TICKETS, T_ID, T_LENGTH,
    // string values
    T_GOLD, T_PLATINUM, T_STANDARD, T_HIGH, T_MEDIUM, T_LOW,
    T_CRITICAL, T_ACTIVE, T_SUSPENDED, T_OPEN, T_IN_PROGRESS,
    T_RESOLVED, T_CLOSED, T_PROCESSING, T_COMPLETED, T_CANCELLED,
    // digits
    T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9,
    // EOS
    T_EOS,
    VOCAB_SIZE
};

std::vector<std::string> build_vocab() {
    std::vector<std::string> v(VOCAB_SIZE);
    v[T_CONST]   = "const";   v[T_FOR]    = "for";     v[T_OF]    = "of";
    v[T_IF]      = "if";      v[T_ELSE]   = "else";    v[T_AWAIT] = "await";
    v[T_SP_CONST]  = " const";  v[T_SP_FOR]   = " for";
    v[T_SP_OF]     = " of";     v[T_SP_IF]    = " if";
    v[T_SP_ELSE]   = " else";   v[T_SP_AWAIT] = " await";
    v[T_LPAREN]  = "(";   v[T_RPAREN]  = ")";  v[T_LBRACE] = "{";
    v[T_RBRACE]  = "}";   v[T_SEMI]    = ";";  v[T_DOT]    = ".";
    v[T_COMMA]   = ",";   v[T_COLON]   = ":";  v[T_EQ]     = "=";
    v[T_GT]      = ">";   v[T_LT]      = "<";  v[T_GTE]    = ">=";
    v[T_LTE]     = "<=";  v[T_TRIPLE_EQ] = "===";
    v[T_SPACE]   = " ";   v[T_NEWLINE] = "\n"; v[T_QUOTE]  = "\"";
    v[T_GETACCOUNTS]      = "getAccounts";
    v[T_GETSUPPORTTICKETS]= "getSupportTickets";
    v[T_GETOPENORDERS]    = "getOpenOrders";
    v[T_REASSIGNACCOUNT]  = "reassignAccount";
    v[T_ADDACCOUNTNOTE]   = "addAccountNote";
    v[T_SP_GETACCOUNTS]   = " getAccounts";
    // Split tokens — key for Option B multi-token literal test
    v[T_GET]        = "get";       // partial prefix of "getAccounts"
    v[T_ACCOUNTS_S] = "Accounts";  // suffix completing "getAccounts"
    v[T_ACCOUNTID]  = "accountId"; v[T_TEAMID]   = "teamId";
    v[T_PRIORITY]   = "priority";  v[T_TIER]     = "tier";
    v[T_NOTE]       = "note";      v[T_SEVERITY] = "severity";
    v[T_ACCOUNTS]   = "accounts";  v[T_ACCOUNT]  = "account";
    v[T_TICKET]     = "ticket";    v[T_TICKETS]  = "tickets";
    v[T_ORDERS]     = "orders";    v[T_CRITICAL_TICKETS] = "critical_tickets";
    v[T_ID]         = "id";        v[T_LENGTH]   = "length";
    v[T_GOLD]       = "gold";      v[T_PLATINUM] = "platinum";
    v[T_STANDARD]   = "standard";  v[T_HIGH]     = "high";
    v[T_MEDIUM]     = "medium";    v[T_LOW]      = "low";
    v[T_CRITICAL]   = "critical";  v[T_ACTIVE]   = "active";
    v[T_SUSPENDED]  = "suspended"; v[T_OPEN]     = "open";
    v[T_IN_PROGRESS]= "in_progress"; v[T_RESOLVED] = "resolved";
    v[T_CLOSED]     = "closed";    v[T_PROCESSING] = "processing";
    v[T_COMPLETED]  = "completed"; v[T_CANCELLED]  = "cancelled";
    v[T_0]="0"; v[T_1]="1"; v[T_2]="2"; v[T_3]="3"; v[T_4]="4";
    v[T_5]="5"; v[T_6]="6"; v[T_7]="7"; v[T_8]="8"; v[T_9]="9";
    v[T_EOS] = "<|endoftext|>";
    return v;
}

// ============================================================================
// feed_sequence: drives grammar through a token-ID sequence.
// Throws with detail on first rejection.
// ============================================================================

bool feed_sequence(qwen3::GrammarVocab& g,
                   const std::vector<int32_t>& tokens,
                   const std::vector<std::string>& vocab,
                   bool verbose = false)
{
    g.reset();
    for (size_t i = 0; i < tokens.size(); ++i) {
        int32_t tid = tokens[i];
        auto valid = g.get_valid_tokens(vocab);
        if (std::find(valid.begin(), valid.end(), tid) == valid.end()) {
            // Build error with valid set
            std::ostringstream oss;
            oss << "Token '" << vocab[tid] << "' (id=" << tid
                << ") rejected at step " << i
                << ". Valid (" << valid.size() << "):";
            for (int32_t v : valid) {
                if (static_cast<size_t>(v) < vocab.size())
                    oss << " '" << vocab[v] << "'";
            }
            throw std::runtime_error(oss.str());
        }
        if (verbose)
            std::cout << "      step " << i << ": '" << vocab[tid] << "'" << std::endl;
        g.accept_token(tid, vocab);
    }
    return g.is_accepting_state();
}

// Feed prefix only (no validation) for state-inspection tests
void feed_prefix(qwen3::GrammarVocab& g,
                 const std::vector<int32_t>& tokens,
                 const std::vector<std::string>& vocab)
{
    g.reset();
    for (int32_t t : tokens) g.accept_token(t, vocab);
}

std::string load_gbnf() {
    return R"GBNF(
root ::= statement ("\n" statement)*

statement ::= for_of | let_decl | if_else | parallel_block | await_call

for_of ::= "for" " " "(" "const" " " identifier " " "of" " " identifier ")" " "? block

let_decl ::= "const" " " identifier " "? "=" " "? expression " "? ";"

if_else ::= "if" " "? "(" " "? condition " "? ")" " "? block (" "? "else" " "? block)?

parallel_block ::= "await" " " "Promise.all" " "? "(" " "? "[" " "? await_call_list " "? "]" " "? ")" " "? ";"

await_call ::= "await" " " function_call " "? ";"

await_call_list ::= await_call_item (" "? "," " "? await_call_item)*
await_call_item ::= "await" " " function_call

block ::= "{" "\n" (statement "\n")* "}"

expression ::= await_expr | count_expr | number | variable | string

await_expr ::= "await" " " function_call

count_expr ::= identifier "." "length"

number ::= [0-9] ([0-9])*

condition ::= expression " "? comparator " "? expression
comparator ::= ">" | "<" | "===" | ">=" | "<="

function_call ::= function_name " "? "(" object? ")"
function_name ::= "getAccounts" | "getSupportTickets" | "getOpenOrders" | "reassignAccount" | "addAccountNote"

object ::= "{" " "? (param (" "? "," " "? param)*)? " "? "}"
param ::= param_key " "? ":" " "? param_value
param_key ::= "accountId" | "teamId" | "priority" | "tier" | "note" | "severity"
param_value ::= expression

variable ::= identifier ("." identifier)*
identifier ::= "accounts" | "account" | "ticket" | "tickets" | "orders" | "critical_tickets" | "id" | "severity" | "length"

string ::= "\"" string_value "\""
string_value ::= "gold" | "platinum" | "standard" | "high" | "medium" | "low" | "critical" | "active" | "suspended" | "open" | "in_progress" | "resolved" | "closed" | "processing" | "completed" | "cancelled"
)GBNF";
}

// ============================================================================
// SUITE 1 — Parse & basic infrastructure
// ============================================================================
void suite1_parse(const std::string& gbnf, const std::vector<std::string>& vocab) {
    std::cout << "\n=== Suite 1: Parse & infrastructure ===" << std::endl;

    TEST("parse_impl succeeds on valid GBNF", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf);
        ASSERT_TRUE(g != nullptr);
    });

    TEST("parse_impl returns nullptr on empty string", {
        auto g = qwen3::GrammarVocab::parse_impl("");
        ASSERT_TRUE(g == nullptr);
    });

    TEST("parse_impl returns nullptr on missing root", {
        auto g = qwen3::GrammarVocab::parse_impl("foo ::= \"bar\"");
        ASSERT_TRUE(g == nullptr);
    });

    TEST("initial state is not accepting", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf);
        ASSERT_TRUE(g != nullptr);
        ASSERT_FALSE(g->is_accepting_state());
    });

    TEST("initial valid tokens include statement starters", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf);
        ASSERT_TRUE(g != nullptr);
        auto valid = g->get_valid_tokens(vocab);
        ASSERT_IN(valid, T_CONST);
        ASSERT_IN(valid, T_FOR);
        ASSERT_IN(valid, T_IF);
        ASSERT_IN(valid, T_AWAIT);
        // EOS is not a valid start
        ASSERT_NOT_IN(valid, T_EOS);
    });
}

// ============================================================================
// SUITE 2 — Regression: fixed GBNF still works (mirrors test_grammar.cpp)
// ============================================================================
void suite2_regression(const std::string& gbnf, const std::vector<std::string>& vocab) {
    std::cout << "\n=== Suite 2: Regression — fixed GBNF productions ===" << std::endl;

    TEST("let_decl: getAccounts() no args", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        ASSERT_TRUE(feed_sequence(*g, {
            T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
            T_AWAIT, T_SPACE, T_GETACCOUNTS, T_LPAREN, T_RPAREN, T_SEMI
        }, vocab));
    });

    TEST("let_decl: getAccounts({ tier: \"gold\" })", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        ASSERT_TRUE(feed_sequence(*g, {
            T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
            T_AWAIT, T_SPACE, T_GETACCOUNTS, T_LPAREN, T_LBRACE, T_SPACE,
            T_TIER, T_COLON, T_SPACE, T_QUOTE, T_GOLD, T_QUOTE,
            T_SPACE, T_RBRACE, T_RPAREN, T_SEMI
        }, vocab));
    });

    TEST("for_of: simple loop with await body", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        ASSERT_TRUE(feed_sequence(*g, {
            T_FOR, T_SPACE, T_LPAREN, T_CONST, T_SPACE, T_ACCOUNT,
            T_SPACE, T_OF, T_SPACE, T_ACCOUNTS, T_RPAREN, T_SPACE,
            T_LBRACE, T_NEWLINE,
            T_AWAIT, T_SPACE, T_ADDACCOUNTNOTE, T_LPAREN, T_LBRACE, T_SPACE,
            T_ACCOUNTID, T_COLON, T_SPACE, T_ACCOUNT,
            T_SPACE, T_RBRACE, T_RPAREN, T_SEMI,
            T_NEWLINE, T_RBRACE
        }, vocab));
    });

    TEST("no double space: after '(' in empty call, space NOT valid", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        feed_prefix(*g, {
            T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
            T_AWAIT, T_SPACE, T_GETACCOUNTS, T_LPAREN
        }, vocab);
        auto valid = g->get_valid_tokens(vocab);
        ASSERT_IN(valid, T_RPAREN);
        ASSERT_NOT_IN(valid, T_SPACE);
    });

    TEST("multi-statement: two let_decls", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        ASSERT_TRUE(feed_sequence(*g, {
            T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
            T_AWAIT, T_SPACE, T_GETACCOUNTS, T_LPAREN, T_RPAREN, T_SEMI,
            T_NEWLINE,
            T_CONST, T_SPACE, T_TICKETS, T_SPACE, T_EQ, T_SPACE,
            T_AWAIT, T_SPACE, T_GETSUPPORTTICKETS, T_LPAREN, T_LBRACE, T_SPACE,
            T_ACCOUNTID, T_COLON, T_SPACE, T_ACCOUNT,
            T_SPACE, T_RBRACE, T_RPAREN, T_SEMI
        }, vocab));
    });

    TEST("numeric condition: orders.length > 5", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        ASSERT_TRUE(feed_sequence(*g, {
            T_IF, T_SPACE, T_LPAREN,
            T_ORDERS, T_DOT, T_LENGTH, T_GT, T_5,
            T_RPAREN, T_SPACE,
            T_LBRACE, T_NEWLINE,
            T_AWAIT, T_SPACE, T_ADDACCOUNTNOTE, T_LPAREN, T_LBRACE, T_SPACE,
            T_ACCOUNTID, T_COLON, T_SPACE, T_ACCOUNT,
            T_SPACE, T_RBRACE, T_RPAREN, T_SEMI,
            T_NEWLINE, T_RBRACE
        }, vocab));
    });

    TEST("full platinum example", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        ASSERT_TRUE(feed_sequence(*g, {
            T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
            T_AWAIT, T_SPACE, T_GETACCOUNTS, T_LPAREN, T_LBRACE, T_SPACE,
            T_TIER, T_COLON, T_SPACE, T_QUOTE, T_PLATINUM, T_QUOTE,
            T_SPACE, T_RBRACE, T_RPAREN, T_SEMI,
            T_NEWLINE,
            T_FOR, T_SPACE, T_LPAREN, T_CONST, T_SPACE, T_ACCOUNT,
            T_SPACE, T_OF, T_SPACE, T_ACCOUNTS, T_RPAREN, T_SPACE,
            T_LBRACE, T_NEWLINE,
            T_CONST, T_SPACE, T_ORDERS, T_SPACE, T_EQ, T_SPACE,
            T_AWAIT, T_SPACE, T_GETOPENORDERS, T_LPAREN, T_LBRACE, T_SPACE,
            T_ACCOUNTID, T_COLON, T_SPACE, T_ACCOUNT, T_DOT, T_ID,
            T_SPACE, T_RBRACE, T_RPAREN, T_SEMI,
            T_NEWLINE,
            T_IF, T_SPACE, T_LPAREN,
            T_ORDERS, T_DOT, T_LENGTH, T_GT, T_5,
            T_RPAREN, T_SPACE,
            T_LBRACE, T_NEWLINE,
            T_AWAIT, T_SPACE, T_ADDACCOUNTNOTE, T_LPAREN, T_LBRACE, T_SPACE,
            T_ACCOUNTID, T_COLON, T_SPACE, T_ACCOUNT, T_DOT, T_ID, T_COMMA, T_SPACE,
            T_NOTE, T_COLON, T_SPACE, T_QUOTE, T_HIGH, T_QUOTE,
            T_SPACE, T_RBRACE, T_RPAREN, T_SEMI,
            T_NEWLINE, T_RBRACE,
            T_NEWLINE, T_RBRACE
        }, vocab, /*verbose=*/true));
    });
}

// ============================================================================
// SUITE 3 — Option B specific: char_idx tracks partial literal consumption
// This is impossible with the old token_idx approach because the map would
// need "get" → partial match of "getAccounts" pre-registered.
// ============================================================================
void suite3_option_b_char_tracking(const std::string& gbnf, const std::vector<std::string>& vocab) {
    std::cout << "\n=== Suite 3: Option B — char_idx partial literal consumption ===" << std::endl;

    TEST("'get' is valid at start of getAccounts literal (partial prefix)", {
        // Grammar expects "getAccounts". Vocab has T_GET="get" and T_ACCOUNTS_S="Accounts".
        // char_idx=0: "get" == "getAccounts".substr(0,3) → valid
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        feed_prefix(*g, {
            T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
            T_AWAIT, T_SPACE  // now at function_name position
        }, vocab);
        auto valid = g->get_valid_tokens(vocab);
        ASSERT_IN(valid, T_GET);           // "get" is a valid prefix of "getAccounts"
        ASSERT_IN(valid, T_GETACCOUNTS);   // full token also valid
    });

    TEST("after accepting 'get', 'Accounts' is the only valid completion", {
        // char_idx advances to 3. Now only tokens matching "getAccounts".substr(3,...) = "Accounts..."
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        feed_prefix(*g, {
            T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
            T_AWAIT, T_SPACE, T_GET  // partial: consumed "get", char_idx=3
        }, vocab);
        auto valid = g->get_valid_tokens(vocab);
        ASSERT_IN(valid, T_ACCOUNTS_S);    // "Accounts" == "getAccounts".substr(3) ✓
        ASSERT_NOT_IN(valid, T_GETACCOUNTS); // full "getAccounts" ≠ substr(3) ✗
        ASSERT_NOT_IN(valid, T_RPAREN);    // ")" is not a completion of the literal ✗
    });

    TEST("full split-token sequence: 'get'+'Accounts'+'('+')'  accepts", {
        // Proves end-to-end: multi-token literal via char_idx works
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        ASSERT_TRUE(feed_sequence(*g, {
            T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
            T_AWAIT, T_SPACE,
            T_GET, T_ACCOUNTS_S,   // ← "getAccounts" split across two tokens
            T_LPAREN, T_RPAREN, T_SEMI
        }, vocab));
    });

    TEST("mid-literal state is NOT accepting", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        feed_prefix(*g, {
            T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
            T_AWAIT, T_SPACE, T_GET  // mid-literal
        }, vocab);
        ASSERT_FALSE(g->is_accepting_state());
    });
}

// ============================================================================
// SUITE 4 — Space-prefixed tokens work via separate " " literal in GBNF
// This tests that the vocab-scan naturally allows space-prefixed tokens
// at whitespace positions without any pre-enumeration.
// ============================================================================
void suite4_space_tokens(const std::string& gbnf, const std::vector<std::string>& vocab) {
    std::cout << "\n=== Suite 4: Space-prefixed token handling ===" << std::endl;

    TEST("at ' ' literal position, both ' ' and ' getAccounts' are valid", {
        // After "await", grammar expects " " then function_name.
        // In BPE, " getAccounts" as one token = the space consumed by the " " literal.
        // char_idx tracks: " getAccounts"[0] == ' ' → matches " " literal.
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        feed_prefix(*g, {
            T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
            T_AWAIT  // grammar now at the " " literal before function_name
        }, vocab);
        auto valid = g->get_valid_tokens(vocab);
        ASSERT_IN(valid, T_SPACE);           // bare " " ✓
        ASSERT_IN(valid, T_SP_GETACCOUNTS);  // " getAccounts" starts with " " ✓
        ASSERT_NOT_IN(valid, T_GETACCOUNTS); // "getAccounts" doesn't start with " " ✗
    });

    TEST("accepting ' getAccounts' at space position advances char into literal", {
        // After accepting " getAccounts" at the " " position:
        // The " " literal is fully consumed (1 char). We move to function_name.
        // But " getAccounts" has 12 chars and only 1 was used — the remaining
        // "getAccounts" (11 chars) must continue matching function_name literal.
        // This tests cross-element token boundary handling.
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        feed_prefix(*g, {
            T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
            T_AWAIT, T_SP_GETACCOUNTS  // " getAccounts" spans " " + "getAccounts"
        }, vocab);
        auto valid = g->get_valid_tokens(vocab);
        // Should now be at "(" position
        ASSERT_IN(valid, T_LPAREN);
        ASSERT_NOT_IN(valid, T_GETACCOUNTS); // already past function_name
    });

    TEST("every valid token at ' ' position must start with ' '", {
        // Option B key property: at a LITERAL(" ") state, the vocab-scan enforces
        // first-char matching without any pre-enumeration. Every accepted token
        // must begin with ' ' — no special-casing required.
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        feed_prefix(*g, {
            T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE, T_AWAIT
            // grammar now at " " literal before function_name
        }, vocab);
        auto valid = g->get_valid_tokens(vocab);
        // Every token in the valid set must start with ' '
        for (int32_t id : valid) {
            if (id < 0 || static_cast<size_t>(id) >= vocab.size()) continue;
            ASSERT_TRUE(!vocab[id].empty() && vocab[id][0] == ' ');
        }
        // Bare keywords must NOT be valid (they don't start with ' ')
        ASSERT_NOT_IN(valid, T_AWAIT);
        ASSERT_NOT_IN(valid, T_GETACCOUNTS);
    });
}

// ============================================================================
// SUITE 5 — CHAR_CLASS (number rule) still works
// ============================================================================
void suite5_char_class(const std::string& gbnf, const std::vector<std::string>& vocab) {
    std::cout << "\n=== Suite 5: CHAR_CLASS — number rule ===" << std::endl;

    TEST("after '>' comparator, digit tokens are valid", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        feed_prefix(*g, {
            T_IF, T_SPACE, T_LPAREN, T_ORDERS, T_DOT, T_LENGTH, T_GT
        }, vocab);
        auto valid = g->get_valid_tokens(vocab);
        for (int32_t d = T_0; d <= T_9; ++d)
            ASSERT_IN(valid, d);
        ASSERT_NOT_IN(valid, T_RBRACE);
    });

    TEST("multi-digit number: after '5', another digit or ')' is valid", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        feed_prefix(*g, {
            T_IF, T_SPACE, T_LPAREN, T_ORDERS, T_DOT, T_LENGTH, T_GT, T_5
        }, vocab);
        auto valid = g->get_valid_tokens(vocab);
        // Another digit continues the number
        ASSERT_IN(valid, T_0);
        // ) closes the condition
        ASSERT_IN(valid, T_RPAREN);
    });
}

// ============================================================================
// SUITE 6 — reset() restores initial state
// ============================================================================
void suite6_reset(const std::string& gbnf, const std::vector<std::string>& vocab) {
    std::cout << "\n=== Suite 6: reset() ===" << std::endl;

    TEST("reset after partial feed restores initial valid tokens", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        // Feed half a statement
        g->accept_token(T_CONST, vocab);
        g->accept_token(T_SPACE, vocab);
        // Reset
        g->reset();
        // Should be back to initial state
        auto valid = g->get_valid_tokens(vocab);
        ASSERT_IN(valid, T_CONST);
        ASSERT_IN(valid, T_FOR);
        ASSERT_FALSE(g->is_accepting_state());
    });

    TEST("reset after complete statement allows re-parsing", {
        auto g = qwen3::GrammarVocab::parse_impl(gbnf); ASSERT_TRUE(g);
        feed_sequence(*g, {
            T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
            T_AWAIT, T_SPACE, T_GETACCOUNTS, T_LPAREN, T_RPAREN, T_SEMI
        }, vocab);
        ASSERT_TRUE(g->is_accepting_state());
        g->reset();
        ASSERT_FALSE(g->is_accepting_state());
        // Can parse again
        ASSERT_TRUE(feed_sequence(*g, {
            T_AWAIT, T_SPACE, T_ADDACCOUNTNOTE, T_LPAREN, T_LBRACE, T_SPACE,
            T_ACCOUNTID, T_COLON, T_SPACE, T_ACCOUNT,
            T_SPACE, T_RBRACE, T_RPAREN, T_SEMI
        }, vocab));
    });
}

// ============================================================================
// main
// ============================================================================
int main() {
    std::string gbnf = load_gbnf();
    auto vocab = build_vocab();

    std::cout << "GrammarVocab TDD (Option B — vocab-scan)\n";
    std::cout << "Vocab size: " << vocab.size() << " tokens\n";

    suite1_parse(gbnf, vocab);
    suite2_regression(gbnf, vocab);
    suite3_option_b_char_tracking(gbnf, vocab);
    suite4_space_tokens(gbnf, vocab);
    suite5_char_class(gbnf, vocab);
    suite6_reset(gbnf, vocab);

    std::cout << "\n==============================\n"
              << "  PASSED: " << g_pass << "\n"
              << "  FAILED: " << g_fail << "\n"
              << "==============================\n";
    return g_fail > 0 ? 1 : 0;
}