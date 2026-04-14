// grammar_vocab.cpp
// GrammarState::char_idx tracks characters consumed within the current LITERAL,
// enabling tokens to partially match, fully match, or cross element boundaries.

// BUG: Mutual-recursion state leak in get_valid_tokens.
// When rule A references rule B which references A (e.g. root→expr, expr→root|"x"),
// completing the A→B→A branch does not prune the sibling alternatives of the outer B
// invocation. Result: get_valid_tokens returns extra token IDs that are grammatically
// invalid at that position (too permissive, never too restrictive).
// Affects only grammars with two-rule mutual recursion where both alternatives
// complete at the same recursion depth. Direct self-recursion is unaffected.

#include "grammar_vocab.h"
#include "token-trie.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace qwen3 {

// ============================================================================
// Internal element types — identical to grammar.cpp
// ============================================================================
struct Elem {
    enum Type {
        END = 0, ALT = 1, RULE_REF = 2, LITERAL = 3,
        OPEN = 4, CLOSE = 5, OP_STAR = 6, OP_PLUS = 7, OP_OPT = 8, CHAR_CLASS = 9,
    };
    Type type;
    uint32_t value;
};
using Rule = std::vector<Elem>;

// ============================================================================
// Impl
// ============================================================================
struct GrammarVocab::Impl {
    std::vector<Rule> rules;
    std::map<std::string, uint32_t> symbol_ids;
    std::vector<std::string> literal_values;
    std::vector<std::pair<std::set<char>, bool>> char_classes; // {allowed_chars, is_negated}

    // KEY CHANGE vs grammar.cpp: char_idx instead of token_idx.
    // char_idx = number of characters of the current LITERAL already consumed.
    // Always 0 for CHAR_CLASS and END states.
    struct GrammarState {
        const Elem* pos;
        size_t char_idx;
        std::vector<const Elem*> continuations;
    };
    std::vector<GrammarState> stacks;

    // ---- Navigation: identical logic to grammar.cpp ----
    static const Elem* find_rule_end(const Elem* pos, const Impl* impl);
    static void resolve_states(const Impl* impl, const Elem* pos,
                               std::vector<const Elem*> conts,
                               std::vector<GrammarState>& out);
    static void explore_alternatives(const Impl* impl, const Elem* pos,
                                     std::vector<const Elem*> conts,
                                     std::vector<GrammarState>& out);

    // consume_token: try to consume token_str[offset:] starting from (pos, char_idx).
    //   On success, appends resulting GrammarState(s) to `result`.
    //   On mismatch, appends nothing.
    static void consume_token(const Impl* impl,
                              const std::string& token_str, size_t offset,
                              const Elem* pos, size_t char_idx,
                              const std::vector<const Elem*>& conts,
                              std::vector<GrammarState>& result);

    // advance_after_terminal: called when a terminal (LITERAL or CHAR_CLASS) has been
    //   fully consumed up to new_offset. Handles ALT/END/sequence navigation then
    //   recursively continues consuming the token.
    static void advance_after_terminal(const Impl* impl,
                                       const std::string& token_str, size_t new_offset,
                                       const Elem* terminal_pos,
                                       const std::vector<const Elem*>& conts,
                                       std::vector<GrammarState>& result);

    // resolve_and_consume: resolve pos to terminal states via explore_alternatives,
    //   then call consume_token on each.
    static void resolve_and_consume(const Impl* impl,
                                    const std::string& token_str, size_t offset,
                                    const Elem* pos,
                                    const std::vector<const Elem*>& conts,
                                    std::vector<GrammarState>& result);
};

// ============================================================================
// Navigation — same logic as grammar.cpp, adapted for Elem
// ============================================================================

const Elem* GrammarVocab::Impl::find_rule_end(const Elem* pos, const Impl* impl) {
    for (const auto& rule : impl->rules)
        if (pos >= rule.data() && pos < rule.data() + rule.size())
            return rule.data() + rule.size() - 1;
    return nullptr;
}

void GrammarVocab::Impl::resolve_states(
    const Impl* impl, const Elem* pos,
    std::vector<const Elem*> conts,
    std::vector<GrammarState>& out)
{
    if (pos->type == Elem::LITERAL)  { out.push_back({pos, 0, conts}); return; }
    if (pos->type == Elem::CHAR_CLASS){ out.push_back({pos, 0, conts}); return; }

    if (pos->type == Elem::END || pos->type == Elem::ALT) {
        if (!conts.empty()) {
            auto ret = conts.back();
            std::vector<const Elem*> rest(conts.begin(), conts.end()-1);
            resolve_states(impl, ret, rest, out);
        } else if (pos->type == Elem::END) {
            out.push_back({pos, 0, conts}); // accepting state
        }
        return;
    }

    if (pos->type == Elem::RULE_REF) {
        auto new_conts = conts;
        new_conts.push_back(pos + 1);
        explore_alternatives(impl, impl->rules[pos->value].data(), new_conts, out);
        return;
    }

    if (pos->type == Elem::OPEN) {
        // Find matching CLOSE and the operator after it
        int depth = 1;
        const Elem* scan = pos + 1;
        while (depth > 0) {
            if      (scan->type == Elem::OPEN)  depth++;
            else if (scan->type == Elem::CLOSE) depth--;
            scan++;
        }
        const Elem* after_close = scan; // element after CLOSE (the operator, or next sequence element)

        if (after_close->type == Elem::OP_OPT) {
            auto take_conts = conts; take_conts.push_back(after_close + 1);
            explore_alternatives(impl, pos + 1, take_conts, out); // take optional
            explore_alternatives(impl, after_close + 1, conts, out); // skip optional
        } else if (after_close->type == Elem::OP_STAR) {
            auto take_conts = conts; take_conts.push_back(pos); // loop back to OPEN
            explore_alternatives(impl, pos + 1, take_conts, out); // take one iteration
            explore_alternatives(impl, after_close + 1, conts, out); // skip (zero iterations)
        } else {
            // Bare group
            auto group_conts = conts; group_conts.push_back(after_close);
            explore_alternatives(impl, pos + 1, group_conts, out);
        }
        return;
    }

    if (pos->type == Elem::CLOSE) {
        if (!conts.empty()) {
            auto ret = conts.back();
            std::vector<const Elem*> rest(conts.begin(), conts.end()-1);
            explore_alternatives(impl, ret, rest, out);
        }
        return;
    }
}

void GrammarVocab::Impl::explore_alternatives(
    const Impl* impl, const Elem* pos,
    std::vector<const Elem*> conts,
    std::vector<GrammarState>& out)
{
    const Elem* current = pos;
    while (true) {
        resolve_states(impl, current, conts, out);
        // Scan forward to the next ALT at the current level
        int depth = 0;
        const Elem* next_alt = current;
        while (next_alt->type != Elem::END &&
               (next_alt->type != Elem::ALT || depth > 0)) {
            if      (next_alt->type == Elem::OPEN)  depth++;
            else if (next_alt->type == Elem::CLOSE) depth--;
            next_alt++;
        }
        if (next_alt->type == Elem::ALT) current = next_alt + 1;
        else break;
    }
}

// ============================================================================
//  token consumption
// ============================================================================

void GrammarVocab::Impl::advance_after_terminal(
    const Impl* impl,
    const std::string& token_str, size_t new_offset,
    const Elem* terminal_pos,
    const std::vector<const Elem*>& conts,
    std::vector<GrammarState>& result)
{
    const Elem* next_pos = terminal_pos + 1;

    if (next_pos->type == Elem::ALT || next_pos->type == Elem::END) {
        // This alternative / rule is done
        if (!conts.empty()) {
            auto ret = conts.back();
            std::vector<const Elem*> new_conts(conts.begin(), conts.end()-1);
            resolve_and_consume(impl, token_str, new_offset, ret, new_conts, result);
        } else if (new_offset == token_str.size()) {
            // Root-level completion, token fully consumed → accepting state
            auto rule_end = find_rule_end(terminal_pos, impl);
            if (rule_end) result.push_back({rule_end, 0, {}});
        }
        // If new_offset < size and conts empty: token outlives grammar → invalid, don't push
    } else {
        // Sequence continues: resolve next element and keep consuming
        resolve_and_consume(impl, token_str, new_offset, next_pos, conts, result);
    }
}

void GrammarVocab::Impl::resolve_and_consume(
    const Impl* impl,
    const std::string& token_str, size_t offset,
    const Elem* pos,
    const std::vector<const Elem*>& conts,
    std::vector<GrammarState>& result)
{
    // Resolve pos to all reachable terminals, then try consuming token from each
    std::vector<GrammarState> terminals;
    explore_alternatives(impl, pos, conts, terminals);
    for (const auto& ts : terminals)
        consume_token(impl, token_str, offset, ts.pos, ts.char_idx, ts.continuations, result);
}

void GrammarVocab::Impl::consume_token(
    const Impl* impl,
    const std::string& token_str, size_t offset,
    const Elem* pos, size_t char_idx,
    const std::vector<const Elem*>& conts,
    std::vector<GrammarState>& result)
{
    // Base case: token fully consumed — record current grammar position as a result state
    if (offset == token_str.size()) {
        result.push_back({pos, char_idx, conts});
        return;
    }

    if (!pos || pos->type == Elem::END)
        return; // Grammar done but token not exhausted → invalid path

    if (pos->type == Elem::LITERAL) {
        const std::string& lit = impl->literal_values[pos->value];
        // How much of the literal remains after char_idx chars already consumed?
        const size_t lit_remaining = lit.size() - char_idx;
        const size_t tok_remaining = token_str.size() - offset;
        const size_t to_match     = std::min(tok_remaining, lit_remaining);

        // Character-level match
        if (token_str.compare(offset, to_match, lit, char_idx, to_match) != 0)
            return; // mismatch — this path is invalid

        if (tok_remaining < lit_remaining) {
            // Token exhausted mid-literal: partial consumption
            result.push_back({pos, char_idx + tok_remaining, conts});
        } else {
            // Literal fully consumed by this token (tok_remaining >= lit_remaining)
            advance_after_terminal(impl, token_str, offset + lit_remaining, pos, conts, result);
        }

    } else if (pos->type == Elem::CHAR_CLASS) {
        const auto& cc = impl->char_classes[pos->value];
        const char first_char = token_str[offset];
        const bool in_set  = cc.first.count(first_char) > 0;
        const bool allowed = cc.second ? !in_set : in_set;
        if (!allowed) return;

        // CHAR_CLASS matches exactly one character of the token string
        advance_after_terminal(impl, token_str, offset + 1, pos, conts, result);
    }
    // Other types (END, ALT, etc.) are never stored as live states — safe to ignore
}

// ============================================================================
// GBNF Parser — identical to grammar.cpp, uses Elem instead of llama_grammar_element
// ============================================================================
class GBNFParser {
public:
    GBNFParser(const std::string& gbnf,
               std::vector<std::string>& literals,
               std::vector<std::pair<std::set<char>, bool>>& char_classes)
        : pos_(gbnf.data()), end_(gbnf.data() + gbnf.size()),
          literals_(literals), char_classes_(char_classes) {}

    std::map<std::string, uint32_t> parse_rules(std::vector<Rule>& rules) {
        std::map<std::string, uint32_t> sym;

        // First pass: collect rule names
        const char* scan = pos_;
        while (scan < end_) {
            while (scan < end_ && (std::isspace(*scan) || *scan == '#')) {
                if (*scan == '#') while (scan < end_ && *scan != '\n' && *scan != '\r') scan++;
                else scan++;
            }
            if (scan >= end_) break;
            const char* ns = scan;
            while (scan < end_ && (std::isalnum(*scan) || *scan == '-' || *scan == '_')) scan++;
            if (ns == scan) { scan++; continue; }
            std::string name(ns, scan - ns);
            while (scan < end_ && std::isspace(*scan)) scan++;
            if (scan + 3 <= end_ && std::strncmp(scan, "::=", 3) == 0) {
                if (!sym.count(name)) { sym[name] = (uint32_t)rules.size(); rules.emplace_back(); }
                scan += 3;
            }
        }

        // Second pass: parse rule bodies
        while (pos_ < end_) {
            skip_ws();
            if (pos_ >= end_) break;
            std::string name = parse_name();
            skip_ws(); expect("::="); skip_ws();
            rules[sym.at(name)] = parse_alts(sym, rules);
        }
        return sym;
    }

private:
    const char* pos_;
    const char* const end_;
    std::vector<std::string>& literals_;
    std::vector<std::pair<std::set<char>, bool>>& char_classes_;

    void skip_ws() {
        while (pos_ < end_) {
            if (std::isspace(*pos_)) pos_++;
            else if (*pos_ == '#') { while (pos_ < end_ && *pos_ != '\n' && *pos_ != '\r') pos_++; }
            else break;
        }
    }
    void expect(const std::string& s) {
        if ((size_t)(end_ - pos_) < s.size() || std::strncmp(pos_, s.data(), s.size()) != 0)
            throw std::runtime_error("GBNF error: expected '" + s + "'");
        pos_ += s.size();
    }
    std::string parse_name() {
        const char* start = pos_;
        while (pos_ < end_ && (std::isalnum(*pos_) || *pos_ == '-' || *pos_ == '_')) pos_++;
        if (start == pos_) throw std::runtime_error("GBNF error: expected name");
        return {start, pos_};
    }

    Rule parse_alts(const std::map<std::string, uint32_t>& sym, const std::vector<Rule>& rules) {
        Rule r;
        auto seq = parse_seq(sym, rules);
        r.insert(r.end(), seq.begin(), seq.end());
        while (true) {
            skip_ws();
            if (pos_ >= end_ || *pos_ != '|') break;
            pos_++; skip_ws();
            r.push_back({Elem::ALT, 0});
            seq = parse_seq(sym, rules);
            r.insert(r.end(), seq.begin(), seq.end());
        }
        r.push_back({Elem::END, 0});
        return r;
    }

    Rule parse_seq(const std::map<std::string, uint32_t>& sym, const std::vector<Rule>& rules) {
        Rule r;
        while (true) {
            skip_ws();
            if (pos_ >= end_ || *pos_ == '|' || *pos_ == ')' || *pos_ == ']') break;
            if (std::isalnum(*pos_) || *pos_ == '-' || *pos_ == '_') {
                const char* p = pos_;
                while (p < end_ && (std::isalnum(*p) || *p == '-' || *p == '_')) p++;
                while (p < end_ && (*p == ' ' || *p == '\t')) p++;
                if (p + 3 <= end_ && std::strncmp(p, "::=", 3) == 0) break;
            }
            auto term = parse_term(sym, rules);
            r.insert(r.end(), term.begin(), term.end());
        }
        return r;
    }

    Rule parse_term(const std::map<std::string, uint32_t>& sym, const std::vector<Rule>& rules) {
        Rule term;
        if      (*pos_ == '"') term = parse_literal();
        else if (*pos_ == '(') term = parse_group(sym, rules);
        else if (*pos_ == '[') term = parse_char_class();
        else {
            std::string name = parse_name();
            if (!sym.count(name)) throw std::runtime_error("GBNF error: unknown rule '" + name + "'");
            term.push_back({Elem::RULE_REF, sym.at(name)});
        }

        skip_ws();
        if (pos_ < end_) {
            if (*pos_ == '*') { pos_++; term = wrap(term, Elem::OP_STAR); }
            else if (*pos_ == '+') {
                pos_++;
                auto rep = wrap(term, Elem::OP_STAR);
                term.insert(term.end(), rep.begin(), rep.end()); // desugar to term STAR(term)
            }
            else if (*pos_ == '?') { pos_++; term = wrap(term, Elem::OP_OPT); }
        }
        return term;
    }

    Rule parse_literal() {
        pos_++; // skip "
        std::ostringstream ss;
        while (pos_ < end_ && *pos_ != '"') {
            if (*pos_ == '\\') {
                pos_++;
                if (pos_ >= end_) throw std::runtime_error("Unterminated escape");
                switch (*pos_) {
                    case 'n': ss << '\n'; break; case 'r': ss << '\r'; break;
                    case 't': ss << '\t'; break; case '"': ss << '"';  break;
                    case '\\':ss << '\\'; break; default:  ss << *pos_;
                }
            } else { ss << *pos_; }
            pos_++;
        }
        if (pos_ >= end_) throw std::runtime_error("Unterminated literal");
        pos_++; // skip "
        literals_.push_back(ss.str());
        return {{Elem::LITERAL, (uint32_t)(literals_.size()-1)}};
    }

    // Read one char from a char-class body, handling backslash escapes
    // (\n \r \t \\ and pass-through for anything else).
    char read_cc_char() {
        char c = *pos_++;
        if (c != '\\') return c;
        if (pos_ >= end_) throw std::runtime_error("Unterminated escape in char class");
        char e = *pos_++;
        switch (e) {
            case 'n':  return '\n';
            case 'r':  return '\r';
            case 't':  return '\t';
            case '\\': return '\\';
            case ']':  return ']';
            default:   return e;
        }
    }

    Rule parse_char_class() {
        pos_++; // skip [
        bool neg = (pos_ < end_ && *pos_ == '^') ? (pos_++, true) : false;
        std::set<char> chars;
        while (pos_ < end_ && *pos_ != ']') {
            char c = read_cc_char();
            // Peek for a range: c-e (but not c-] which ends the class)
            if (pos_ + 1 < end_ && *pos_ == '-' && *(pos_+1) != ']') {
                pos_++; // skip '-'
                char e = read_cc_char();
                for (char ch = c; ch <= e; ch++) chars.insert(ch);
            } else { chars.insert(c); }
        }
        if (pos_ >= end_) throw std::runtime_error("Unterminated char class");
        pos_++; // skip ]
        char_classes_.push_back({chars, neg});
        return {{Elem::CHAR_CLASS, (uint32_t)(char_classes_.size()-1)}};
    }

    Rule wrap(const Rule& inner, Elem::Type op) {
        Rule r;
        r.push_back({Elem::OPEN, 0});
        r.insert(r.end(), inner.begin(), inner.end());
        r.push_back({Elem::CLOSE, 0});
        r.push_back({op, 0});
        return r;
    }

    Rule parse_group(const std::map<std::string, uint32_t>& sym, const std::vector<Rule>& rules) {
        pos_++; // skip (
        auto inner = parse_alts(sym, rules);
        if (!inner.empty() && inner.back().type == Elem::END) inner.pop_back();
        expect(")");
        Rule r;
        r.push_back({Elem::OPEN, 0});
        r.insert(r.end(), inner.begin(), inner.end());
        r.push_back({Elem::CLOSE, 0});
        return r;
    }
};

// ============================================================================
// GrammarVocab public interface
// ============================================================================

GrammarVocab::GrammarVocab()  : impl_(std::make_unique<Impl>()) {}
GrammarVocab::~GrammarVocab() = default;

void GrammarVocab::set_token_trie(const TokenTrie* trie) { trie_ = trie; }

std::unique_ptr<GrammarVocab> GrammarVocab::parse_impl(const std::string& gbnf_str) {
    if (gbnf_str.empty()) return nullptr;

    std::unique_ptr<GrammarVocab> g(new GrammarVocab());
    try {
        GBNFParser parser(gbnf_str, g->impl_->literal_values, g->impl_->char_classes);
        g->impl_->symbol_ids = parser.parse_rules(g->impl_->rules);
    } catch (const std::exception& e) {
        std::cerr << "GBNF parse error: " << e.what() << std::endl;
        return nullptr;
    }

    if (g->impl_->rules.empty() || !g->impl_->symbol_ids.count("root"))
        return nullptr;

    g->reset();
    return g;
}

void GrammarVocab::reset() {
    if (!impl_ || impl_->rules.empty()) return;
    impl_->stacks.clear();
    const uint32_t root_id = impl_->symbol_ids.at("root");
    Impl::explore_alternatives(impl_.get(), impl_->rules[root_id].data(), {}, impl_->stacks);
}

bool GrammarVocab::is_accepting_state() const {
    if (!impl_) return false;
    for (const auto& s : impl_->stacks)
        if (s.pos && s.pos->type == Elem::END) return true;
    return false;
}

std::vector<int32_t> GrammarVocab::get_valid_tokens(const std::vector<std::string>& vocab) const {
    std::set<int32_t> valid_set;

    for (const auto& state : impl_->stacks) {
        if (!state.pos) continue;

        if (state.pos->type == Elem::LITERAL) {
            const std::string& lit = impl_->literal_values[state.pos->value];
            if (state.char_idx >= lit.size()) continue;

            if (trie_) {
                // Trie fast path: collect only candidate tokens whose string
                // could possibly match the remaining literal suffix.
                std::vector<int32_t> candidates;
                trie_->collect_by_prefix(lit, state.char_idx, candidates);

                for (int32_t tok_id : candidates) {
                    std::vector<Impl::GrammarState> result;
                    Impl::consume_token(impl_.get(), vocab[tok_id], 0,
                                        state.pos, state.char_idx,
                                        state.continuations, result);
                    if (!result.empty())
                        valid_set.insert(tok_id);
                }
            } else {
                // Brute-force fallback: scan all vocab tokens
                const char expected_first = lit[state.char_idx];
                for (size_t i = 0; i < vocab.size(); ++i) {
                    if (vocab[i].empty()) continue;
                    if (vocab[i][0] != expected_first) continue;

                    std::vector<Impl::GrammarState> result;
                    Impl::consume_token(impl_.get(), vocab[i], 0,
                                        state.pos, state.char_idx,
                                        state.continuations, result);
                    if (!result.empty())
                        valid_set.insert(static_cast<int32_t>(i));
                }
            }

        } else if (state.pos->type == Elem::CHAR_CLASS) {
            const auto& cc = impl_->char_classes[state.pos->value];
            for (size_t i = 0; i < vocab.size(); ++i) {
                if (vocab[i].empty()) continue;
                const char fc = vocab[i][0];
                const bool in_set  = cc.first.count(fc) > 0;
                const bool allowed = cc.second ? !in_set : in_set;
                if (allowed) valid_set.insert(static_cast<int32_t>(i));
            }
        }
        // END state: grammar accepting, contributes no further tokens
    }

    return {valid_set.begin(), valid_set.end()};
}

void GrammarVocab::accept_token(int32_t token_id, const std::vector<std::string>& vocab) {
    if (!impl_) return;
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab.size()) return;

    const std::string& token_str = vocab[token_id];
    std::vector<Impl::GrammarState> new_stacks;

    for (const auto& state : impl_->stacks) {
        if (!state.pos) continue;
        Impl::consume_token(impl_.get(), token_str, 0,
                            state.pos, state.char_idx,
                            state.continuations, new_stacks);
    }

    impl_->stacks = std::move(new_stacks);
}

void GrammarVocab::dump_expected() const {
    if (!impl_) return;
    std::cerr << "  Grammar active paths: " << impl_->stacks.size() << "\n";
    for (size_t i = 0; i < impl_->stacks.size(); ++i) {
        const auto& state = impl_->stacks[i];
        if (!state.pos) continue;
        if (state.pos->type == Elem::LITERAL) {
            const std::string& lit = impl_->literal_values[state.pos->value];
            std::cerr << "    [" << i << "] LITERAL: "" << lit << "" (at char_idx " << state.char_idx;
            if (state.char_idx < lit.size()) {
                std::cerr << ", expecting '" << lit[state.char_idx] << "')";
            } else {
                std::cerr << ", complete)";
            }
            std::cerr << "\n";
        } else if (state.pos->type == Elem::CHAR_CLASS) {
            std::cerr << "    [" << i << "] CHAR_CLASS\n";
        } else if (state.pos->type == Elem::END) {
            std::cerr << "    [" << i << "] END (accepting state)\n";
        }
    }
}
} // namespace qwen3
