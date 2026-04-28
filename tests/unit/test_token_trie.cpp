// test_token_trie.cpp — Unit tests for TokenTrie (Phase 1) and
// grammar integration equivalence tests (Phase 2).

#include <gtest/gtest.h>
#include "../../src/sampling/token-trie.h"
#include "../../src/sampling/grammar_vocab.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

using namespace qwenium;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Small vocab for focused testing
static std::vector<std::string> build_small_vocab() {
    return {
        /* 0 */ "p",
        /* 1 */ "pr",
        /* 2 */ "pri",
        /* 3 */ "price",
        /* 4 */ "print",
        /* 5 */ "hello",
        /* 6 */ "he",
        /* 7 */ "h",
        /* 8 */ "123",
        /* 9 */ "1",
        /* 10 */ "",         // empty token
        /* 11 */ "price_tag",
        /* 12 */ "a",
        /* 13 */ "ab",
        /* 14 */ "abc",
        /* 15 */ "abd",
        /* 16 */ "z",
        /* 17 */ "9lives",
        /* 18 */ "\n",
        /* 19 */ " ",        // space token
    };
}

static TokenTrie build_small_trie() {
    TokenTrie trie;
    trie.build(build_small_vocab());
    return trie;
}

// Convenience: collect into a sorted vector for comparison
static std::vector<int32_t> sorted(std::vector<int32_t> v) {
    std::sort(v.begin(), v.end());
    return v;
}

// ---------------------------------------------------------------------------
// Suite 1: Build correctness
// ---------------------------------------------------------------------------

TEST(TokenTrie, BuildDoesNotCrash) {
    TokenTrie trie;
    auto vocab = build_small_vocab();
    EXPECT_NO_THROW(trie.build(vocab));
}

TEST(TokenTrie, BuildFromEmptyVocab) {
    TokenTrie trie;
    std::vector<std::string> empty;
    EXPECT_NO_THROW(trie.build(empty));
}

TEST(TokenTrie, BuildSkipsEmptyTokens) {
    // Empty string at index 10 should not appear in any collection
    auto trie = build_small_trie();
    std::vector<int32_t> out;
    trie.collect_by_first_char('p', out);
    // token 10 (empty) should never appear
    EXPECT_EQ(std::find(out.begin(), out.end(), 10), out.end());
}

// ---------------------------------------------------------------------------
// Suite 2: collect_by_first_char
// ---------------------------------------------------------------------------

TEST(TokenTrie, CollectByFirstChar_P) {
    auto trie = build_small_trie();
    std::vector<int32_t> out;
    trie.collect_by_first_char('p', out);
    auto result = sorted(out);
    // tokens starting with 'p': 0(p), 1(pr), 2(pri), 3(price), 4(print), 11(price_tag)
    EXPECT_EQ(result, (std::vector<int32_t>{0, 1, 2, 3, 4, 11}));
}

TEST(TokenTrie, CollectByFirstChar_H) {
    auto trie = build_small_trie();
    std::vector<int32_t> out;
    trie.collect_by_first_char('h', out);
    auto result = sorted(out);
    // tokens starting with 'h': 5(hello), 6(he), 7(h)
    EXPECT_EQ(result, (std::vector<int32_t>{5, 6, 7}));
}

TEST(TokenTrie, CollectByFirstChar_NoMatch) {
    auto trie = build_small_trie();
    std::vector<int32_t> out;
    trie.collect_by_first_char('x', out);
    EXPECT_TRUE(out.empty());
}

// ---------------------------------------------------------------------------
// Suite 3: collect_by_prefix (the critical operation for LITERAL matching)
//
// collect_by_prefix("ice", ...) should return:
//   - tokens on the path from root: tokens whose string is a prefix of "ice"
//     (but only those that start from offset 0 in "ice")
//   - tokens in the subtree at "ice": tokens that start with "ice"
//
// BUT the real use case is: given remaining literal "ice" (from "price" at
// char_idx=2), find all token IDs whose string:
//   (a) matches a prefix of "ice" → partial consumption (token "i" consumes 1 char)
//   (b) exactly matches "ice" → full consumption
//   (c) starts with "ice" and extends → overshoot (token "icebreaker")
// ---------------------------------------------------------------------------

TEST(TokenTrie, CollectByPrefix_ExactAndExtensions) {
    // Vocab: "price" (3), "price_tag" (11)
    // Prefix "price" → should get: "p"(0), "pr"(1), "pri"(2), "price"(3), "price_tag"(11), "print"(4)
    // Wait — "print" starts with "pri" which is a prefix of "price", but at 4th char
    // "print"[3]='n' vs "price"[3]='c' → "print" does NOT match prefix "price" past "pri"
    // So path tokens: "p"(0), "pr"(1), "pri"(2), "price"(3)
    // Subtree tokens at "price": "price"(3), "price_tag"(11)
    // Union: {0, 1, 2, 3, 11}
    auto trie = build_small_trie();
    std::vector<int32_t> out;
    trie.collect_by_prefix("price", 0, out);
    auto result = sorted(out);
    EXPECT_EQ(result, (std::vector<int32_t>{0, 1, 2, 3, 11}));
}

TEST(TokenTrie, CollectByPrefix_PartialLiteral) {
    // remaining = "ice" (as if grammar is in LITERAL "price" at char_idx=2)
    // Path tokens: tokens that are prefixes of "ice" → need a token "i" — not in vocab
    // Subtree tokens at "ice": none in this vocab
    // So result = {} (no tokens match "ice" as prefix or extension)
    auto trie = build_small_trie();
    std::vector<int32_t> out;
    trie.collect_by_prefix("ice", 0, out);
    EXPECT_TRUE(out.empty());
}

TEST(TokenTrie, CollectByPrefix_SingleChar) {
    // remaining = "a"
    // Path tokens at "a": "a"(12)
    // Subtree: "ab"(13), "abc"(14), "abd"(15)
    // Union: {12, 13, 14, 15}
    auto trie = build_small_trie();
    std::vector<int32_t> out;
    trie.collect_by_prefix("a", 0, out);
    auto result = sorted(out);
    EXPECT_EQ(result, (std::vector<int32_t>{12, 13, 14, 15}));
}

TEST(TokenTrie, CollectByPrefix_WithOffset) {
    // prefix = "price", offset = 2 → effectively querying "ice"
    auto trie = build_small_trie();
    std::vector<int32_t> out;
    trie.collect_by_prefix("price", 2, out);
    EXPECT_TRUE(out.empty());  // no tokens start with "i","ic","ice"
}

TEST(TokenTrie, CollectByPrefix_FullVocabToken) {
    // remaining = "hello"
    // Path: "h"(7), "he"(6), "hello"(5)
    // Subtree at "hello": just "hello"(5)
    // Union: {5, 6, 7}
    auto trie = build_small_trie();
    std::vector<int32_t> out;
    trie.collect_by_prefix("hello", 0, out);
    auto result = sorted(out);
    EXPECT_EQ(result, (std::vector<int32_t>{5, 6, 7}));
}

TEST(TokenTrie, CollectByPrefix_LongerThanAnyToken) {
    // remaining = "price_tag_extra" — longer than any vocab token
    // Path: "p"(0), "pr"(1), "pri"(2), "price"(3), "price_tag"(11)
    // Subtree at "price_tag_extra": nothing (no token that long)
    // Union: {0, 1, 2, 3, 11}
    auto trie = build_small_trie();
    std::vector<int32_t> out;
    trie.collect_by_prefix("price_tag_extra", 0, out);
    auto result = sorted(out);
    EXPECT_EQ(result, (std::vector<int32_t>{0, 1, 2, 3, 11}));
}

TEST(TokenTrie, CollectByPrefix_EmptyPrefix) {
    // Empty prefix → should match ALL non-empty tokens (every token is "extension" of "")
    auto trie = build_small_trie();
    auto vocab = build_small_vocab();
    std::vector<int32_t> out;
    trie.collect_by_prefix("", 0, out);
    // Should be all tokens except the empty one (index 10)
    size_t non_empty = 0;
    for (size_t i = 0; i < vocab.size(); ++i)
        if (!vocab[i].empty()) non_empty++;
    EXPECT_EQ(out.size(), non_empty);
}

// ---------------------------------------------------------------------------
// Suite 4: collect_by_char_class
// ---------------------------------------------------------------------------

TEST(TokenTrie, CollectByCharClass_Digits) {
    // [0-9] → tokens starting with a digit: "123"(8), "1"(9), "9lives"(17)
    auto trie = build_small_trie();
    std::set<char> digits;
    for (char c = '0'; c <= '9'; c++) digits.insert(c);

    std::vector<int32_t> out;
    trie.collect_by_char_class(digits, /*negated=*/false, out);
    auto result = sorted(out);
    EXPECT_EQ(result, (std::vector<int32_t>{8, 9, 17}));
}

TEST(TokenTrie, CollectByCharClass_NegatedDigits) {
    // [^0-9] → all tokens NOT starting with a digit
    auto trie = build_small_trie();
    auto vocab = build_small_vocab();
    std::set<char> digits;
    for (char c = '0'; c <= '9'; c++) digits.insert(c);

    std::vector<int32_t> out;
    trie.collect_by_char_class(digits, /*negated=*/true, out);
    auto result = sorted(out);

    // Manually compute: all non-empty tokens not starting with digit
    std::vector<int32_t> expected;
    for (size_t i = 0; i < vocab.size(); ++i) {
        if (vocab[i].empty()) continue;
        if (digits.count(vocab[i][0]) == 0) expected.push_back(i);
    }
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(result, expected);
}

TEST(TokenTrie, CollectByCharClass_SingleChar) {
    // {z} → only token "z"(16)
    auto trie = build_small_trie();
    std::set<char> chars = {'z'};
    std::vector<int32_t> out;
    trie.collect_by_char_class(chars, false, out);
    EXPECT_EQ(sorted(out), (std::vector<int32_t>{16}));
}

TEST(TokenTrie, CollectByCharClass_EmptySet) {
    // Empty char set, not negated → nothing matches
    auto trie = build_small_trie();
    std::set<char> empty;
    std::vector<int32_t> out;
    trie.collect_by_char_class(empty, false, out);
    EXPECT_TRUE(out.empty());
}

TEST(TokenTrie, CollectByCharClass_EmptySetNegated) {
    // Empty char set, negated → everything matches (every char is NOT in empty set)
    auto trie = build_small_trie();
    auto vocab = build_small_vocab();
    std::set<char> empty;
    std::vector<int32_t> out;
    trie.collect_by_char_class(empty, true, out);
    size_t non_empty = 0;
    for (const auto& t : vocab) if (!t.empty()) non_empty++;
    EXPECT_EQ(out.size(), non_empty);
}

// ---------------------------------------------------------------------------
// Suite 5: Special characters and edge cases
// ---------------------------------------------------------------------------

TEST(TokenTrie, SpecialChars_Newline) {
    auto trie = build_small_trie();
    std::vector<int32_t> out;
    trie.collect_by_first_char('\n', out);
    EXPECT_EQ(sorted(out), (std::vector<int32_t>{18}));
}

TEST(TokenTrie, SpecialChars_Space) {
    auto trie = build_small_trie();
    std::vector<int32_t> out;
    trie.collect_by_first_char(' ', out);
    EXPECT_EQ(sorted(out), (std::vector<int32_t>{19}));
}

TEST(TokenTrie, CollectByPrefix_SpecialCharPrefix) {
    // remaining = "\n" — should find token 18
    auto trie = build_small_trie();
    std::vector<int32_t> out;
    trie.collect_by_prefix("\n", 0, out);
    EXPECT_EQ(sorted(out), (std::vector<int32_t>{18}));
}

// ---------------------------------------------------------------------------
// Suite 6: Equivalence — trie collect_by_first_char matches brute force
// ---------------------------------------------------------------------------

TEST(TokenTrie, Equivalence_FirstCharMatchesBruteForce) {
    auto vocab = build_small_vocab();
    auto trie = build_small_trie();

    // For every possible byte value, verify trie matches brute-force
    for (int c = 0; c < 256; c++) {
        std::vector<int32_t> trie_result;
        trie.collect_by_first_char(static_cast<char>(c), trie_result);

        std::vector<int32_t> brute;
        for (size_t i = 0; i < vocab.size(); ++i) {
            if (!vocab[i].empty() && vocab[i][0] == static_cast<char>(c))
                brute.push_back(static_cast<int32_t>(i));
        }

        EXPECT_EQ(sorted(trie_result), sorted(brute))
            << "Mismatch for char " << c;
    }
}

// ===========================================================================
// Phase 2: Grammar integration equivalence tests
//
// Verify that get_valid_tokens() with a trie attached produces identical
// results to the brute-force path (no trie).
// Uses the same grammar and vocab from test_grammar_vocab.cpp.
// ===========================================================================

// --- Reuse the grammar vocab from test_grammar_vocab.cpp ---
enum TokId : int32_t {
    T_CONST = 0, T_FOR, T_OF, T_IF, T_ELSE, T_AWAIT,
    T_SP_CONST, T_SP_FOR, T_SP_OF, T_SP_IF, T_SP_ELSE, T_SP_AWAIT,
    T_LPAREN, T_RPAREN, T_LBRACE, T_RBRACE, T_SEMI, T_DOT, T_COMMA,
    T_COLON, T_EQ, T_GT, T_LT, T_GTE, T_LTE, T_TRIPLE_EQ,
    T_SPACE, T_NEWLINE,
    T_QUOTE,
    T_GETACCOUNTS, T_GETSUPPORTTICKETS, T_GETOPENORDERS,
    T_REASSIGNACCOUNT, T_ADDACCOUNTNOTE,
    T_SP_GETACCOUNTS,
    T_GET, T_ACCOUNTS_S,
    T_ACCOUNTID, T_TEAMID, T_PRIORITY, T_TIER, T_NOTE, T_SEVERITY,
    T_ACCOUNTS, T_ACCOUNT, T_TICKET, T_TICKETS, T_ORDERS,
    T_CRITICAL_TICKETS, T_ID, T_LENGTH,
    T_GOLD, T_PLATINUM, T_STANDARD, T_HIGH, T_MEDIUM, T_LOW,
    T_CRITICAL, T_ACTIVE, T_SUSPENDED, T_OPEN, T_IN_PROGRESS,
    T_RESOLVED, T_CLOSED, T_PROCESSING, T_COMPLETED, T_CANCELLED,
    T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9,
    T_EOS,
    VOCAB_SIZE
};

static std::vector<std::string> build_grammar_vocab() {
    std::vector<std::string> v(VOCAB_SIZE);
    v[T_CONST]="const"; v[T_FOR]="for"; v[T_OF]="of"; v[T_IF]="if";
    v[T_ELSE]="else"; v[T_AWAIT]="await";
    v[T_SP_CONST]=" const"; v[T_SP_FOR]=" for"; v[T_SP_OF]=" of";
    v[T_SP_IF]=" if"; v[T_SP_ELSE]=" else"; v[T_SP_AWAIT]=" await";
    v[T_LPAREN]="("; v[T_RPAREN]=")"; v[T_LBRACE]="{"; v[T_RBRACE]="}";
    v[T_SEMI]=";"; v[T_DOT]="."; v[T_COMMA]=","; v[T_COLON]=":";
    v[T_EQ]="="; v[T_GT]=">"; v[T_LT]="<"; v[T_GTE]=">=";
    v[T_LTE]="<="; v[T_TRIPLE_EQ]="===";
    v[T_SPACE]=" "; v[T_NEWLINE]="\n"; v[T_QUOTE]="\"";
    v[T_GETACCOUNTS]="getAccounts"; v[T_GETSUPPORTTICKETS]="getSupportTickets";
    v[T_GETOPENORDERS]="getOpenOrders"; v[T_REASSIGNACCOUNT]="reassignAccount";
    v[T_ADDACCOUNTNOTE]="addAccountNote";
    v[T_SP_GETACCOUNTS]=" getAccounts";
    v[T_GET]="get"; v[T_ACCOUNTS_S]="Accounts";
    v[T_ACCOUNTID]="accountId"; v[T_TEAMID]="teamId";
    v[T_PRIORITY]="priority"; v[T_TIER]="tier";
    v[T_NOTE]="note"; v[T_SEVERITY]="severity";
    v[T_ACCOUNTS]="accounts"; v[T_ACCOUNT]="account";
    v[T_TICKET]="ticket"; v[T_TICKETS]="tickets";
    v[T_ORDERS]="orders"; v[T_CRITICAL_TICKETS]="critical_tickets";
    v[T_ID]="id"; v[T_LENGTH]="length";
    v[T_GOLD]="gold"; v[T_PLATINUM]="platinum"; v[T_STANDARD]="standard";
    v[T_HIGH]="high"; v[T_MEDIUM]="medium"; v[T_LOW]="low";
    v[T_CRITICAL]="critical"; v[T_ACTIVE]="active";
    v[T_SUSPENDED]="suspended"; v[T_OPEN]="open";
    v[T_IN_PROGRESS]="in_progress"; v[T_RESOLVED]="resolved";
    v[T_CLOSED]="closed"; v[T_PROCESSING]="processing";
    v[T_COMPLETED]="completed"; v[T_CANCELLED]="cancelled";
    v[T_0]="0"; v[T_1]="1"; v[T_2]="2"; v[T_3]="3"; v[T_4]="4";
    v[T_5]="5"; v[T_6]="6"; v[T_7]="7"; v[T_8]="8"; v[T_9]="9";
    v[T_EOS]="<|endoftext|>";
    return v;
}

static std::string load_test_gbnf() {
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

// Helper: compare get_valid_tokens with and without trie at a given grammar state.
// Feeds `prefix_tokens` first, then checks equivalence at the current position.
static void assert_equivalence_at(const std::string& gbnf,
                                   const std::vector<std::string>& vocab,
                                   const TokenTrie& trie,
                                   const std::vector<int32_t>& prefix_tokens,
                                   const std::string& description) {
    // Brute-force: no trie
    auto g_brute = qwenium::GrammarVocab::parse_impl(gbnf);
    ASSERT_NE(g_brute, nullptr);
    for (int32_t t : prefix_tokens) g_brute->accept_token(t, vocab);
    auto brute_result = sorted(g_brute->get_valid_tokens(vocab));

    // Trie-accelerated
    auto g_trie = qwenium::GrammarVocab::parse_impl(gbnf);
    ASSERT_NE(g_trie, nullptr);
    g_trie->set_token_trie(&trie);
    for (int32_t t : prefix_tokens) g_trie->accept_token(t, vocab);
    auto trie_result = sorted(g_trie->get_valid_tokens(vocab));

    EXPECT_EQ(trie_result, brute_result) << "Equivalence failure: " << description;
}

// --- Equivalence tests ---

TEST(TokenTrieGrammar, Equivalence_InitialState) {
    auto vocab = build_grammar_vocab();
    auto gbnf = load_test_gbnf();
    TokenTrie trie;
    trie.build(vocab);
    assert_equivalence_at(gbnf, vocab, trie, {}, "initial state");
}

TEST(TokenTrieGrammar, Equivalence_AfterConst) {
    auto vocab = build_grammar_vocab();
    auto gbnf = load_test_gbnf();
    TokenTrie trie;
    trie.build(vocab);
    assert_equivalence_at(gbnf, vocab, trie,
        {T_CONST},
        "after 'const'");
}

TEST(TokenTrieGrammar, Equivalence_MidLiteral_AfterGet) {
    // After "get" — grammar is mid-literal in "getAccounts" etc.
    // This is the critical partial-consumption case.
    auto vocab = build_grammar_vocab();
    auto gbnf = load_test_gbnf();
    TokenTrie trie;
    trie.build(vocab);
    assert_equivalence_at(gbnf, vocab, trie,
        {T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
         T_AWAIT, T_SPACE, T_GET},
        "mid-literal after 'get'");
}

TEST(TokenTrieGrammar, Equivalence_AtSpaceBeforeFuncName) {
    // At the " " literal before function_name — space-prefixed tokens test
    auto vocab = build_grammar_vocab();
    auto gbnf = load_test_gbnf();
    TokenTrie trie;
    trie.build(vocab);
    assert_equivalence_at(gbnf, vocab, trie,
        {T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
         T_AWAIT},
        "at ' ' before function_name");
}

TEST(TokenTrieGrammar, Equivalence_AtComparator) {
    // After "orders.length" — at comparator position (mix of LITERAL states)
    auto vocab = build_grammar_vocab();
    auto gbnf = load_test_gbnf();
    TokenTrie trie;
    trie.build(vocab);
    assert_equivalence_at(gbnf, vocab, trie,
        {T_IF, T_SPACE, T_LPAREN, T_ORDERS, T_DOT, T_LENGTH},
        "at comparator position");
}

TEST(TokenTrieGrammar, Equivalence_AtDigitPosition) {
    // After ">" — at number/CHAR_CLASS position
    auto vocab = build_grammar_vocab();
    auto gbnf = load_test_gbnf();
    TokenTrie trie;
    trie.build(vocab);
    assert_equivalence_at(gbnf, vocab, trie,
        {T_IF, T_SPACE, T_LPAREN, T_ORDERS, T_DOT, T_LENGTH, T_GT},
        "at digit (CHAR_CLASS) position");
}

TEST(TokenTrieGrammar, Equivalence_AfterDigit) {
    // After "5" — digits or ")" valid
    auto vocab = build_grammar_vocab();
    auto gbnf = load_test_gbnf();
    TokenTrie trie;
    trie.build(vocab);
    assert_equivalence_at(gbnf, vocab, trie,
        {T_IF, T_SPACE, T_LPAREN, T_ORDERS, T_DOT, T_LENGTH, T_GT, T_5},
        "after digit '5'");
}

TEST(TokenTrieGrammar, Equivalence_MultiStatementBoundary) {
    // After first full statement + newline — at second statement start
    auto vocab = build_grammar_vocab();
    auto gbnf = load_test_gbnf();
    TokenTrie trie;
    trie.build(vocab);
    assert_equivalence_at(gbnf, vocab, trie,
        {T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
         T_AWAIT, T_SPACE, T_GETACCOUNTS, T_LPAREN, T_RPAREN, T_SEMI,
         T_NEWLINE},
        "at second statement start");
}

TEST(TokenTrieGrammar, Equivalence_InsideObject) {
    // Inside object parameter — after "{ tier:"
    auto vocab = build_grammar_vocab();
    auto gbnf = load_test_gbnf();
    TokenTrie trie;
    trie.build(vocab);
    assert_equivalence_at(gbnf, vocab, trie,
        {T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
         T_AWAIT, T_SPACE, T_GETACCOUNTS, T_LPAREN, T_LBRACE, T_SPACE,
         T_TIER, T_COLON},
        "inside object after 'tier:'");
}

TEST(TokenTrieGrammar, Equivalence_FullSequenceThenCheck) {
    // Drive through a full complex sequence, check at the end
    auto vocab = build_grammar_vocab();
    auto gbnf = load_test_gbnf();
    TokenTrie trie;
    trie.build(vocab);
    // Full platinum example up to the for loop body start
    assert_equivalence_at(gbnf, vocab, trie,
        {T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
         T_AWAIT, T_SPACE, T_GETACCOUNTS, T_LPAREN, T_LBRACE, T_SPACE,
         T_TIER, T_COLON, T_SPACE, T_QUOTE, T_PLATINUM, T_QUOTE,
         T_SPACE, T_RBRACE, T_RPAREN, T_SEMI,
         T_NEWLINE,
         T_FOR, T_SPACE, T_LPAREN, T_CONST, T_SPACE, T_ACCOUNT,
         T_SPACE, T_OF, T_SPACE, T_ACCOUNTS, T_RPAREN, T_SPACE,
         T_LBRACE, T_NEWLINE},
        "after for loop header, inside block body");
}

// --- Phase 3: CHAR_CLASS-specific equivalence tests ---

TEST(TokenTrieGrammar, Equivalence_NegatedCharClass) {
    // Grammar with negated char class: string content is [^"]*
    // This tests the trie CHAR_CLASS path with negation.
    auto vocab = build_grammar_vocab();
    std::string gbnf = R"GBNF(
root ::= "\"" [^"]* "\""
)GBNF";
    TokenTrie trie;
    trie.build(vocab);
    // After opening quote, [^"]* is active — everything except quote is valid
    assert_equivalence_at(gbnf, vocab, trie,
        {T_QUOTE},
        "negated char class [^\"] after opening quote");
}

TEST(TokenTrieGrammar, Equivalence_CharClassDigitSequence) {
    // Exercise multi-digit: at [0-9] with prior digits consumed
    auto vocab = build_grammar_vocab();
    auto gbnf = load_test_gbnf();
    TokenTrie trie;
    trie.build(vocab);
    // After ">53" — still in number, CHAR_CLASS active
    assert_equivalence_at(gbnf, vocab, trie,
        {T_IF, T_SPACE, T_LPAREN, T_ORDERS, T_DOT, T_LENGTH, T_GT, T_5, T_3},
        "CHAR_CLASS after multi-digit '53'");
}

// --- Phase 4: Multi-state optimization tests ---
// These exercise grammars where multiple active states share the same
// remaining prefix, verifying grouped trie queries produce identical results.

TEST(TokenTrieGrammar, MultiState_OverlappingAwaitPrefixes) {
    // Initial state has both await_call and parallel_block starting with "await".
    // The grouped optimization should deduplicate the "await" trie walk.
    auto vocab = build_grammar_vocab();
    auto gbnf = load_test_gbnf();
    TokenTrie trie;
    trie.build(vocab);
    // Initial state — multiple alternatives active, two start with "await"
    assert_equivalence_at(gbnf, vocab, trie, {}, "overlapping await prefixes at root");
}

TEST(TokenTrieGrammar, MultiState_ManyAlternativesAtFunctionName) {
    // At function_name position, 5 LITERAL alternatives are active simultaneously:
    // "getAccounts", "getSupportTickets", "getOpenOrders", "reassignAccount", "addAccountNote"
    // Three share "get" prefix. Grouped query should walk "get" subtree once.
    auto vocab = build_grammar_vocab();
    auto gbnf = load_test_gbnf();
    TokenTrie trie;
    trie.build(vocab);
    assert_equivalence_at(gbnf, vocab, trie,
        {T_CONST, T_SPACE, T_ACCOUNTS, T_SPACE, T_EQ, T_SPACE,
         T_AWAIT, T_SPACE},
        "5 function_name alternatives active");
}

TEST(TokenTrieGrammar, MultiState_DuplicateLiteralsInAlternatives) {
    // Grammar where the same literal appears in multiple alternatives
    auto vocab = build_grammar_vocab();
    std::string gbnf = R"GBNF(
root ::= a | b | c
a ::= "const" " " "accounts"
b ::= "const" " " "orders"
c ::= "for" " " "accounts"
)GBNF";
    TokenTrie trie;
    trie.build(vocab);
    // At initial state: two states expect "const", one expects "for"
    // Grouped optimization deduplicates the "const" trie walk
    assert_equivalence_at(gbnf, vocab, trie, {},
        "duplicate 'const' literals across alternatives");
    // After "const " — two states active, both at different next literals
    assert_equivalence_at(gbnf, vocab, trie,
        {T_CONST, T_SPACE},
        "after 'const ' with two active alternatives");
}

TEST(TokenTrieGrammar, MultiState_MixedLiteralAndCharClass) {
    // Grammar where LITERAL and CHAR_CLASS states are active simultaneously
    auto vocab = build_grammar_vocab();
    std::string gbnf = R"GBNF(
root ::= num_or_word
num_or_word ::= [0-9] [0-9]* | "const"
)GBNF";
    TokenTrie trie;
    trie.build(vocab);
    // Initial state: CHAR_CLASS [0-9] and LITERAL "const" both active
    assert_equivalence_at(gbnf, vocab, trie, {},
        "mixed LITERAL + CHAR_CLASS at root");
}
