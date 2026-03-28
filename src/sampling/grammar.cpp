#include "grammar.h"
// #include "token-trie.h" // Assuming this will be used later for tokenization lookups

#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <memory>
#include <iostream>
#include <set>
#include <tuple>

namespace qwen3
{

    // ============================================================================
    // Internal Structures
    // ============================================================================

    struct llama_grammar_element
    {
        enum Type
        {
            END = 0,
            ALT = 1,
            RULE_REF = 2,
            LITERAL = 3,
            OPEN = 4,
            CLOSE = 5,
            OP_STAR = 6,
            OP_PLUS = 7,
            OP_OPT = 8,
            CHAR_CLASS = 9,
        };
        Type type;
        uint32_t value;
    };

    using grammar_rule = std::vector<llama_grammar_element>;

    struct Grammar::Impl
    {
        std::vector<grammar_rule> rules;
        std::map<std::string, uint32_t> symbol_ids;
        std::vector<std::string> literal_values;
        std::map<std::string, std::vector<int32_t>> pretokenized_literals;
        std::vector<std::pair<std::set<char>, bool>> char_classes; // pair<allowed_chars, is_negated>

        struct GrammarState
        {
            const llama_grammar_element *pos;
            size_t token_idx;
            std::vector<const llama_grammar_element *> continuations; // Return points after rule refs complete
        };
        std::vector<GrammarState> stacks;

        static void resolve_states(
            const Grammar::Impl *impl,
            const llama_grammar_element *pos,
            std::vector<const llama_grammar_element *> continuations,
            std::vector<GrammarState> &resolved_stacks);

        static void explore_alternatives(
            const Grammar::Impl *impl,
            const llama_grammar_element *pos,
            std::vector<const llama_grammar_element *> continuations,
            std::vector<GrammarState> &resolved_stacks);

        static const llama_grammar_element *find_rule_end(const llama_grammar_element *pos, const Grammar::Impl *impl);
    };

    // ============================================================================
    // Core Algorithm: State Management
    // ============================================================================

    // Helper to find the END element of the rule that contains the given position.
    const llama_grammar_element *Grammar::Impl::find_rule_end(const llama_grammar_element *pos, const Grammar::Impl *impl)
    {
        for (const auto &rule : impl->rules)
        {
            if (pos >= rule.data() && pos < rule.data() + rule.size())
            {
                return rule.data() + rule.size() - 1; // Pointer to the rule's END element
            }
        }
        return nullptr;
    }

    // Resolves a single path until it hits a terminal (LITERAL or CHAR_CLASS) or the end of the grammar.
    void Grammar::Impl::resolve_states(
        const Grammar::Impl *impl,
        const llama_grammar_element *pos,
        std::vector<const llama_grammar_element *> continuations,
        std::vector<Grammar::Impl::GrammarState> &resolved_stacks)
    {
        if (pos->type == llama_grammar_element::LITERAL)
        {
            resolved_stacks.push_back({pos, 0, continuations});
            return;
        }
        if (pos->type == llama_grammar_element::CHAR_CLASS)
        {
            resolved_stacks.push_back({pos, 0, continuations});
            return;
        }
        if (pos->type == llama_grammar_element::END || pos->type == llama_grammar_element::ALT)
        {
            // Rule/alternative ended. Pop continuation if available, otherwise we're done.
            if (!continuations.empty())
            {
                const llama_grammar_element *return_pos = continuations.back();
                std::vector<const llama_grammar_element *> remaining_continuations(
                    continuations.begin(), continuations.end() - 1);
                // Use resolve_states instead of explore_alternatives to avoid re-exploring alternatives
                resolve_states(impl, return_pos, remaining_continuations, resolved_stacks);
            }
            else if (pos->type == llama_grammar_element::END)
            {
                resolved_stacks.push_back({pos, 0, continuations});
            }
            return;
        }
        if (pos->type == llama_grammar_element::RULE_REF)
        {
            // Push continuation (what comes after this rule ref) and enter the referenced rule
            std::vector<const llama_grammar_element *> new_continuations = continuations;
            new_continuations.push_back(pos + 1);
            const grammar_rule &referenced_rule = impl->rules[pos->value];
            explore_alternatives(impl, referenced_rule.data(), new_continuations, resolved_stacks);
            return;
        }
        if (pos->type == llama_grammar_element::OPEN)
        {
            // Find the matching CLOSE and the operator after it
            int depth = 1;
            const llama_grammar_element *scan = pos + 1;
            while (depth > 0)
            {
                if (scan->type == llama_grammar_element::OPEN)
                    depth++;
                else if (scan->type == llama_grammar_element::CLOSE)
                    depth--;
                scan++;
            }
            // scan now points to the element after CLOSE (the operator)
            const llama_grammar_element *after_close = scan;

            if (after_close->type == llama_grammar_element::OP_OPT)
            {
                // Optional: explore both taking it and skipping it
                // Path 1: Take the optional content
                std::vector<const llama_grammar_element *> take_continuations = continuations;
                take_continuations.push_back(after_close + 1); // continue after OP_OPT when group completes
                explore_alternatives(impl, pos + 1, take_continuations, resolved_stacks);

                // Path 2: Skip the optional content
                explore_alternatives(impl, after_close + 1, continuations, resolved_stacks);
            }
            else if (after_close->type == llama_grammar_element::OP_STAR)
            {
                // Star: zero or more - can skip or take (and loop back)
                // Path 1: Take the content, then loop back to this OPEN
                std::vector<const llama_grammar_element *> take_continuations = continuations;
                take_continuations.push_back(pos); // loop back to OPEN after group completes
                explore_alternatives(impl, pos + 1, take_continuations, resolved_stacks);

                // Path 2: Skip the starred content entirely
                explore_alternatives(impl, after_close + 1, continuations, resolved_stacks);
            }
            else
            {
                // Bare group (no operator) - just enter the group
                std::vector<const llama_grammar_element *> group_continuations = continuations;
                group_continuations.push_back(after_close); // continue after CLOSE when group completes
                explore_alternatives(impl, pos + 1, group_continuations, resolved_stacks);
            }
            return;
        }
        if (pos->type == llama_grammar_element::CLOSE)
        {
            // End of a group. Pop continuation to see where to go next.
            if (!continuations.empty())
            {
                const llama_grammar_element *return_pos = continuations.back();
                std::vector<const llama_grammar_element *> remaining_continuations(
                    continuations.begin(), continuations.end() - 1);
                explore_alternatives(impl, return_pos, remaining_continuations, resolved_stacks);
            }
            return;
        }
        // Future: handle OP_PLUS here
    }

    // Finds all alternative paths starting from a given position and resolves them.
    void Grammar::Impl::explore_alternatives(
        const Grammar::Impl *impl,
        const llama_grammar_element *pos,
        std::vector<const llama_grammar_element *> continuations,
        std::vector<Grammar::Impl::GrammarState> &resolved_stacks)
    {
        const llama_grammar_element *current_pos = pos;
        while (true)
        {
            Grammar::Impl::resolve_states(impl, current_pos, continuations, resolved_stacks);

            // Scan forward to the next alternative at the current level
            int group_depth = 0;
            const llama_grammar_element *next_alt = current_pos;
            while (next_alt->type != llama_grammar_element::END && (next_alt->type != llama_grammar_element::ALT || group_depth > 0))
            {
                if (next_alt->type == llama_grammar_element::OPEN)
                    group_depth++;
                else if (next_alt->type == llama_grammar_element::CLOSE)
                    group_depth--;
                next_alt++;
            }

            if (next_alt->type == llama_grammar_element::ALT)
            {
                current_pos = next_alt + 1;
            }
            else
            {
                break; // No more alternatives
            }
        }
    }

    // ============================================================================
    // GBNF Parser (Reused from original implementation)
    // ============================================================================

    class GBNFParser
    {
    public:
        GBNFParser(const std::string &grammar_str,
                   std::vector<std::string> &literal_values,
                   std::vector<std::pair<std::set<char>, bool>> &char_classes)
            : pos_(grammar_str.data()),
              end_(grammar_str.data() + grammar_str.size()),
              literal_values_(literal_values),
              char_classes_(char_classes) {}

        std::map<std::string, uint32_t> parse_rules(std::vector<grammar_rule> &rules)
        {
            std::map<std::string, uint32_t> symbol_ids;

            const char *scan_pos = pos_;
            while (scan_pos < end_)
            {
                while (scan_pos < end_ && (std::isspace(*scan_pos) || *scan_pos == '#'))
                {
                    if (*scan_pos == '#')
                    {
                        while (scan_pos < end_ && *scan_pos != '\n' && *scan_pos != '\r')
                            scan_pos++;
                    }
                    else
                    {
                        scan_pos++;
                    }
                }
                if (scan_pos >= end_)
                    break;

                const char *name_start = scan_pos;
                while (scan_pos < end_ && (std::isalnum(*scan_pos) || *scan_pos == '-' || *scan_pos == '_'))
                {
                    scan_pos++;
                }
                if (name_start == scan_pos)
                {
                    scan_pos++;
                    continue;
                }
                std::string name(name_start, scan_pos - name_start);

                while (scan_pos < end_ && std::isspace(*scan_pos))
                    scan_pos++;

                if (scan_pos + 3 <= end_ && std::strncmp(scan_pos, "::=", 3) == 0)
                {
                    if (symbol_ids.find(name) == symbol_ids.end())
                    {
                        symbol_ids[name] = static_cast<uint32_t>(rules.size());
                        rules.emplace_back();
                    }
                    scan_pos += 3;
                }
            }

            while (pos_ < end_)
            {
                skip_whitespace();
                if (pos_ == end_)
                    break;

                std::string name = parse_rule_name();
                skip_whitespace();
                expect("::=");
                skip_whitespace();

                uint32_t rule_id = symbol_ids[name];
                rules[rule_id] = parse_alternatives(symbol_ids, rules);
            }
            return symbol_ids;
        }

    private:
        const char *pos_;
        const char *const end_;
        std::vector<std::string> &literal_values_;
        std::vector<std::pair<std::set<char>, bool>> &char_classes_;

        void expect(const std::string &str)
        {
            if (static_cast<size_t>(end_ - pos_) < str.size() ||
                std::strncmp(pos_, str.data(), str.size()) != 0)
            {
                throw std::runtime_error("GBNF parse error: expected '" + str + "' at " + std::string(pos_, 10));
            }
            pos_ += str.size();
        }

        void skip_whitespace()
        {
            while (pos_ < end_)
            {
                if (std::isspace(*pos_))
                {
                    pos_++;
                }
                else if (*pos_ == '#')
                {
                    while (pos_ < end_ && *pos_ != '\n' && *pos_ != '\r')
                        pos_++;
                }
                else
                {
                    break;
                }
            }
        }

        std::string parse_rule_name()
        {
            const char *start = pos_;
            while (pos_ < end_ && (std::isalnum(*pos_) || *pos_ == '-' || *pos_ == '_'))
            {
                pos_++;
            }
            if (start == pos_)
                throw std::runtime_error("GBNF parse error: expected rule name");
            return std::string(start, pos_ - start);
        }

        grammar_rule parse_alternatives(const std::map<std::string, uint32_t> &symbol_ids, const std::vector<grammar_rule> &rules)
        {
            grammar_rule rule;
            grammar_rule sequence = parse_sequence(symbol_ids, rules);
            rule.insert(rule.end(), sequence.begin(), sequence.end());

            while (true)
            {
                skip_whitespace();
                if (pos_ >= end_ || *pos_ != '|')
                    break;
                pos_++;
                skip_whitespace();
                rule.push_back({llama_grammar_element::ALT, 0});
                sequence = parse_sequence(symbol_ids, rules);
                rule.insert(rule.end(), sequence.begin(), sequence.end());
            }
            rule.push_back({llama_grammar_element::END, 0});
            return rule;
        }

        grammar_rule parse_sequence(const std::map<std::string, uint32_t> &symbol_ids, const std::vector<grammar_rule> &rules)
        {
            grammar_rule rule;
            while (true)
            {
                skip_whitespace();
                if (pos_ >= end_ || *pos_ == '|' || *pos_ == ')' || *pos_ == ']')
                    break;

                if (std::isalnum(*pos_) || *pos_ == '-' || *pos_ == '_')
                {
                    const char *peek = pos_;
                    while (peek < end_ && (std::isalnum(*peek) || *peek == '-' || *peek == '_'))
                        peek++;
                    while (peek < end_ && (*peek == ' ' || *peek == '\t'))
                        peek++;
                    if (peek + 3 <= end_ && std::strncmp(peek, "::=", 3) == 0)
                    {
                        break;
                    }
                }

                grammar_rule term = parse_term(symbol_ids, rules);
                rule.insert(rule.end(), term.begin(), term.end());
            }
            return rule;
        }

        grammar_rule parse_term(const std::map<std::string, uint32_t> &symbol_ids, const std::vector<grammar_rule> &rules)
        {
            grammar_rule term;
            if (*pos_ == '"')
            {
                term = parse_quoted_literal();
            }
            else if (*pos_ == '(')
            {
                term = parse_group(symbol_ids, rules);
            }
            else if (*pos_ == '[')
            {
                term = parse_char_class();
            }
            else
            {
                std::string name = parse_rule_name();
                if (symbol_ids.count(name))
                {
                    term.push_back({llama_grammar_element::RULE_REF, symbol_ids.at(name)});
                }
                else
                {
                    throw std::runtime_error("GBNF parse error: unknown rule name '" + name + "' ");
                }
            }

            skip_whitespace();
            if (pos_ < end_)
            {
                if (*pos_ == '*')
                {
                    pos_++;
                    term = wrap_with_operator(term, llama_grammar_element::OP_STAR);
                }
                else if (*pos_ == '+')
                {
                    pos_++;
                    grammar_rule repeated = wrap_with_operator(term, llama_grammar_element::OP_STAR);
                    term.insert(term.end(), repeated.begin(), repeated.end());
                }
                else if (*pos_ == '?')
                {
                    pos_++;
                    term = wrap_with_operator(term, llama_grammar_element::OP_OPT);
                }
            }
            return term;
        }

        grammar_rule parse_quoted_literal()
        {
            pos_++;
            std::stringstream ss;
            while (pos_ < end_ && *pos_ != '"')
            {
                if (*pos_ == '\\')
                {
                    pos_++;
                    if (pos_ == end_)
                        throw std::runtime_error("Unterminated escape sequence");
                    switch (*pos_)
                    {
                    case 'n':
                        ss << '\n';
                        break;
                    case 'r':
                        ss << '\r';
                        break;
                    case 't':
                        ss << '\t';
                        break;
                    case '"':
                        ss << '"';
                        break;
                    case '\\':
                        ss << '\\';
                        break;
                    default:
                        ss << *pos_;
                    }
                }
                else
                {
                    ss << *pos_;
                }
                pos_++;
            }
            if (pos_ == end_)
                throw std::runtime_error("Unterminated string literal");
            pos_++;

            literal_values_.push_back(ss.str());
            return {{llama_grammar_element::LITERAL, static_cast<uint32_t>(literal_values_.size() - 1)}};
        }

        grammar_rule parse_char_class()
        {
            pos_++; // consume '['
            bool negated = false;
            if (pos_ < end_ && *pos_ == '^')
            {
                negated = true;
                pos_++;
            }

            std::set<char> chars;
            while (pos_ < end_ && *pos_ != ']')
            {
                char c = *pos_;
                pos_++;

                // Check for range (e.g., a-z)
                if (pos_ + 1 < end_ && *pos_ == '-' && *(pos_ + 1) != ']')
                {
                    pos_++; // consume '-'
                    char end_char = *pos_;
                    pos_++;
                    for (char ch = c; ch <= end_char; ch++)
                    {
                        chars.insert(ch);
                    }
                }
                else
                {
                    chars.insert(c);
                }
            }

            if (pos_ >= end_)
                throw std::runtime_error("GBNF parse error: unterminated character class");
            pos_++; // consume ']'

            char_classes_.push_back({chars, negated});
            return {{llama_grammar_element::CHAR_CLASS, static_cast<uint32_t>(char_classes_.size() - 1)}};
        }

        grammar_rule wrap_with_operator(const grammar_rule &inner, llama_grammar_element::Type op)
        {
            grammar_rule result;
            result.push_back({llama_grammar_element::OPEN, 0});
            result.insert(result.end(), inner.begin(), inner.end());
            result.push_back({llama_grammar_element::CLOSE, 0});
            result.push_back({op, 0});
            return result;
        }

        grammar_rule parse_group(const std::map<std::string, uint32_t> &symbol_ids, const std::vector<grammar_rule> &rules)
        {
            pos_++;
            grammar_rule inner = parse_alternatives(symbol_ids, rules);
            if (!inner.empty() && inner.back().type == llama_grammar_element::END)
                inner.pop_back();
            expect(")");

            grammar_rule result;
            result.push_back({llama_grammar_element::OPEN, 0});
            result.insert(result.end(), inner.begin(), inner.end());
            result.push_back({llama_grammar_element::CLOSE, 0});
            return result;
        }
    };

    // ============================================================================
    // Grammar Class Implementation
    // ============================================================================

    Grammar::Grammar() : impl_(std::make_unique<Impl>()) {}
    Grammar::~Grammar() = default;

    std::unique_ptr<Grammar> Grammar::parse_impl(
        const std::string &gbnf_str,
        const std::map<std::string, std::vector<int32_t>> &pretokenized_literals)
    {

        std::unique_ptr<Grammar> grammar(new Grammar());
        try
        {
            grammar->impl_->pretokenized_literals = pretokenized_literals; // Store the map
            GBNFParser parser(gbnf_str, grammar->impl_->literal_values, grammar->impl_->char_classes);
            grammar->impl_->symbol_ids = parser.parse_rules(grammar->impl_->rules);

            if (grammar->impl_->rules.empty() || grammar->impl_->symbol_ids.find("root") == grammar->impl_->symbol_ids.end())
            {
                return nullptr;
            }

            grammar->reset();
            return grammar;
        }
        catch (const std::exception &e)
        {
            std::cerr << "GBNF parse error: " << e.what() << std::endl;
            return nullptr;
        }
    }

    std::vector<int32_t> Grammar::get_valid_tokens() const
    {
        std::cout << "\n[get_valid_tokens] Inspecting " << impl_->stacks.size() << " states..." << std::endl;
        std::set<int32_t> valid_token_set;
        for (const auto &state : impl_->stacks)
        {
            if (!state.pos)
                continue;

            if (state.pos->type == llama_grammar_element::LITERAL)
            {
                const std::string &literal_str = impl_->literal_values[state.pos->value];
                std::cout << "  - State points to LITERAL '" << literal_str << "', token_idx: " << state.token_idx << std::endl;
                auto it = impl_->pretokenized_literals.find(literal_str);
                if (it != impl_->pretokenized_literals.end())
                {
                    const auto &tokens = it->second;
                    if (state.token_idx < tokens.size())
                    {
                        valid_token_set.insert(tokens[state.token_idx]);
                    }
                }
            }
            else if (state.pos->type == llama_grammar_element::CHAR_CLASS)
            {
                const auto &char_class = impl_->char_classes[state.pos->value];
                const std::set<char> &allowed_chars = char_class.first;
                bool is_negated = char_class.second;

                std::cout << "  - State points to CHAR_CLASS with " << allowed_chars.size()
                          << " chars, negated=" << is_negated << std::endl;

                // Find all tokens whose literal starts with an allowed character
                for (const auto &pair : impl_->pretokenized_literals)
                {
                    const std::string &literal = pair.first;
                    if (literal.empty())
                        continue;

                    char first_char = literal[0];
                    bool char_in_set = allowed_chars.count(first_char) > 0;
                    bool is_allowed = is_negated ? !char_in_set : char_in_set;

                    if (is_allowed)
                    {
                        const auto &tokens = pair.second;
                        if (!tokens.empty())
                        {
                            valid_token_set.insert(tokens[0]);
                        }
                    }
                }
            }
        }
        std::cout << "[get_valid_tokens] Found " << valid_token_set.size() << " valid tokens." << std::endl;
        return {valid_token_set.begin(), valid_token_set.end()};
    }

    std::vector<int32_t> Grammar::get_valid_tokens(const std::vector<std::string> &vocab) const
    {
        std::set<int32_t> valid_token_set;
        for (const auto &state : impl_->stacks)
        {
            if (!state.pos)
                continue;

            if (state.pos->type == llama_grammar_element::LITERAL)
            {
                const std::string &literal_str = impl_->literal_values[state.pos->value];
                auto it = impl_->pretokenized_literals.find(literal_str);
                if (it != impl_->pretokenized_literals.end())
                {
                    const auto &tokens = it->second;
                    if (state.token_idx < tokens.size())
                    {
                        int32_t token_id = tokens[state.token_idx];
                        // Verify token_id is within vocab bounds
                        if (token_id >= 0 && static_cast<size_t>(token_id) < vocab.size())
                        {
                            valid_token_set.insert(token_id);
                        }
                    }
                }
            }
            else if (state.pos->type == llama_grammar_element::CHAR_CLASS)
            {
                const auto &char_class = impl_->char_classes[state.pos->value];
                const std::set<char> &allowed_chars = char_class.first;
                bool is_negated = char_class.second;

                // Scan vocab directly: index IS the token ID
                for (size_t i = 0; i < vocab.size(); ++i)
                {
                    const std::string &token_str = vocab[i];
                    if (token_str.empty())
                        continue;

                    char first_char = token_str[0];
                    bool char_in_set = allowed_chars.count(first_char) > 0;
                    bool is_allowed = is_negated ? !char_in_set : char_in_set;

                    if (is_allowed)
                    {
                        valid_token_set.insert(static_cast<int32_t>(i));
                    }
                }
            }
        }
        return {valid_token_set.begin(), valid_token_set.end()};
    }

    void Grammar::accept_token(int32_t token_id)
    {
        if (!impl_)
            return;
        decltype(impl_->stacks) new_stacks;

        for (const auto &state : impl_->stacks)
        {
            if (!state.pos)
                continue;

            if (state.pos->type == llama_grammar_element::LITERAL)
            {
                const std::string &literal_str = impl_->literal_values[state.pos->value];
                auto it = impl_->pretokenized_literals.find(literal_str);
                if (it != impl_->pretokenized_literals.end())
                {
                    const auto &tokens = it->second;
                    if (state.token_idx < tokens.size() && tokens[state.token_idx] == token_id)
                    {
                        // Token matches, check if literal is complete
                        if (state.token_idx + 1 < tokens.size())
                        {
                            // Not complete, advance token index but keep same grammar position
                            new_stacks.push_back({state.pos, state.token_idx + 1, state.continuations});
                        }
                        else
                        {
                            // Literal complete. Find what comes next.
                            const llama_grammar_element *next_pos = state.pos + 1;
                            if (next_pos->type == llama_grammar_element::ALT || next_pos->type == llama_grammar_element::END)
                            {
                                // This alternative is done. Check continuations or mark as accepting.
                                if (!state.continuations.empty())
                                {
                                    const llama_grammar_element *return_pos = state.continuations.back();
                                    std::vector<const llama_grammar_element *> remaining_continuations(
                                        state.continuations.begin(), state.continuations.end() - 1);
                                    Impl::resolve_states(impl_.get(), return_pos, remaining_continuations, new_stacks);
                                }
                                else
                                {
                                    const llama_grammar_element *rule_end = Impl::find_rule_end(state.pos, impl_.get());
                                    if (rule_end)
                                    {
                                        new_stacks.push_back({rule_end, 0, {}});
                                    }
                                }
                            }
                            else
                            {
                                // This was part of a sequence, explore what's next.
                                Impl::explore_alternatives(impl_.get(), next_pos, state.continuations, new_stacks);
                            }
                        }
                    }
                }
            }
            else if (state.pos->type == llama_grammar_element::CHAR_CLASS)
            {
                const auto &char_class = impl_->char_classes[state.pos->value];
                const std::set<char> &allowed_chars = char_class.first;
                bool is_negated = char_class.second;

                // Find which literal this token corresponds to
                for (const auto &pair : impl_->pretokenized_literals)
                {
                    const std::string &literal = pair.first;
                    const auto &tokens = pair.second;

                    if (tokens.empty() || tokens[0] != token_id)
                        continue;

                    if (literal.empty())
                        continue;

                    char first_char = literal[0];
                    bool char_in_set = allowed_chars.count(first_char) > 0;
                    bool is_allowed = is_negated ? !char_in_set : char_in_set;

                    if (is_allowed)
                    {
                        // Character class matched. Advance to next grammar element.
                        const llama_grammar_element *next_pos = state.pos + 1;
                        if (next_pos->type == llama_grammar_element::ALT || next_pos->type == llama_grammar_element::END)
                        {
                            if (!state.continuations.empty())
                            {
                                const llama_grammar_element *return_pos = state.continuations.back();
                                std::vector<const llama_grammar_element *> remaining_continuations(
                                    state.continuations.begin(), state.continuations.end() - 1);
                                Impl::resolve_states(impl_.get(), return_pos, remaining_continuations, new_stacks);
                            }
                            else
                            {
                                const llama_grammar_element *rule_end = Impl::find_rule_end(state.pos, impl_.get());
                                if (rule_end)
                                {
                                    new_stacks.push_back({rule_end, 0, {}});
                                }
                            }
                        }
                        else
                        {
                            Impl::explore_alternatives(impl_.get(), next_pos, state.continuations, new_stacks);
                        }
                        break; // Found the matching literal, no need to continue
                    }
                }
            }
        }


// --- DEBUG: dump state after accept ---
std::cerr << "[accept] token_id=" << token_id
          << " old_stacks=" << impl_->stacks.size()
          << " new_stacks=" << new_stacks.size() << std::endl;
for (size_t i = 0; i < new_stacks.size(); ++i) {
    auto& s = new_stacks[i];
    if (s.pos) {
        std::cerr << "  [" << i << "] type=" << s.pos->type
                  << " token_idx=" << s.token_idx;
        if (s.pos->type == llama_grammar_element::LITERAL) {
            std::cerr << " literal=\"" << impl_->literal_values[s.pos->value] << "\"";
            auto it = impl_->pretokenized_literals.find(impl_->literal_values[s.pos->value]);
            std::cerr << " in_map=" << (it != impl_->pretokenized_literals.end());
        }
        std::cerr << " continuations=" << s.continuations.size() << std::endl;
    }
}


        impl_->stacks = new_stacks;
    }

    void Grammar::accept_token(int32_t token_id, const std::vector<std::string> &vocab)
    {
        if (!impl_)
            return;

        // Get the string representation of this token
        std::string token_str;
        if (token_id >= 0 && static_cast<size_t>(token_id) < vocab.size())
        {
            token_str = vocab[token_id];
        }

        decltype(impl_->stacks) new_stacks;

        for (const auto &state : impl_->stacks)
        {
            if (!state.pos)
                continue;

            if (state.pos->type == llama_grammar_element::LITERAL)
            {
                const std::string &literal_str = impl_->literal_values[state.pos->value];
                auto it = impl_->pretokenized_literals.find(literal_str);
                if (it != impl_->pretokenized_literals.end())
                {
                    const auto &tokens = it->second;
                    if (state.token_idx < tokens.size() && tokens[state.token_idx] == token_id)
                    {
                        // Token matches, check if literal is complete
                        if (state.token_idx + 1 < tokens.size())
                        {
                            // Not complete, advance token index but keep same grammar position
                            new_stacks.push_back({state.pos, state.token_idx + 1, state.continuations});
                        }
                        else
                        {
                            // Literal complete. Find what comes next.
                            const llama_grammar_element *next_pos = state.pos + 1;
                            if (next_pos->type == llama_grammar_element::ALT || next_pos->type == llama_grammar_element::END)
                            {
                                // This alternative is done. Check continuations or mark as accepting.
                                if (!state.continuations.empty())
                                {
                                    const llama_grammar_element *return_pos = state.continuations.back();
                                    std::vector<const llama_grammar_element *> remaining_continuations(
                                        state.continuations.begin(), state.continuations.end() - 1);
                                    Impl::resolve_states(impl_.get(), return_pos, remaining_continuations, new_stacks);
                                }
                                else
                                {
                                    const llama_grammar_element *rule_end = Impl::find_rule_end(state.pos, impl_.get());
                                    if (rule_end)
                                    {
                                        new_stacks.push_back({rule_end, 0, {}});
                                    }
                                }
                            }
                            else
                            {
                                // This was part of a sequence, explore what's next.
                                Impl::explore_alternatives(impl_.get(), next_pos, state.continuations, new_stacks);
                            }
                        }
                    }
                }
            }
            else if (state.pos->type == llama_grammar_element::CHAR_CLASS)
            {
                const auto &char_class = impl_->char_classes[state.pos->value];
                const std::set<char> &allowed_chars = char_class.first;
                bool is_negated = char_class.second;

                // Use the token string from vocab directly
                if (!token_str.empty())
                {
                    char first_char = token_str[0];
                    bool char_in_set = allowed_chars.count(first_char) > 0;
                    bool is_allowed = is_negated ? !char_in_set : char_in_set;

                    if (is_allowed)
                    {
                        // Character class matched. Advance to next grammar element.
                        const llama_grammar_element *next_pos = state.pos + 1;
                        if (next_pos->type == llama_grammar_element::ALT || next_pos->type == llama_grammar_element::END)
                        {
                            if (!state.continuations.empty())
                            {
                                const llama_grammar_element *return_pos = state.continuations.back();
                                std::vector<const llama_grammar_element *> remaining_continuations(
                                    state.continuations.begin(), state.continuations.end() - 1);
                                Impl::resolve_states(impl_.get(), return_pos, remaining_continuations, new_stacks);
                            }
                            else
                            {
                                const llama_grammar_element *rule_end = Impl::find_rule_end(state.pos, impl_.get());
                                if (rule_end)
                                {
                                    new_stacks.push_back({rule_end, 0, {}});
                                }
                            }
                        }
                        else
                        {
                            Impl::explore_alternatives(impl_.get(), next_pos, state.continuations, new_stacks);
                        }
                    }
                }
            }
        }
        impl_->stacks = new_stacks;
    }

    void Grammar::reset()
    {
        if (!impl_ || impl_->rules.empty())
            return;
        impl_->stacks.clear();
        std::cout << "\n[reset] Cleared stacks." << std::endl;

        uint32_t root_id = impl_->symbol_ids.at("root");
        const grammar_rule &root_rule = impl_->rules[root_id];
        std::cout << "[reset] Starting resolution from root rule..." << std::endl;
        Impl::explore_alternatives(impl_.get(), root_rule.data(), {}, impl_->stacks);
        std::cout << "[reset] Resolution complete. Found " << impl_->stacks.size() << " initial states." << std::endl;
    }

    bool Grammar::is_accepting_state() const
    {
        if (!impl_)
            return false;
        for (const auto &state : impl_->stacks)
        {
            if (state.pos && state.pos->type == llama_grammar_element::END)
            {
                return true;
            }
        }
        return false;
    }

} // namespace qwen3