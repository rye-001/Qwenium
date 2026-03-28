#ifndef TOKEN_GRAMMAR_H
#define TOKEN_GRAMMAR_H

#include <string>
#include <vector>
#include <map>
#include <memory>

// Forward-declare for testing purposes
class GrammarDev_SingleLiteral_Test;
class GrammarDev_MultiTokenLiteralSequence_Test;
class GrammarDev_Alternatives_Test;
class GrammarDev_SimpleRuleRef_Test;
class GrammarDev_RuleRefInSequence_Test;
class GrammarDev_OptionalQuantifier_Test;
class GrammarDev_StarQuantifier_Test;
class GrammarDev_PlusQuantifier_Test;
class GrammarDev_RuleRefWithAlternatives_Test;
class GrammarDev_NestedQuantifiers_Test;
class GrammarDev_SimpleRecursion_Test;
class GrammarDev_ChainedOptionals_Test;
class GrammarDev_ManyAlternatives_Test;
class GrammarDev_MutualRecursion_Test;
class GrammarDev_CharClass_Test;
class GrammarDev_CharClassNegated_Test;
class GrammarDev_CharClassStar_Test;
class GrammarDev_QuotedString_Test;
class GrammarDev_DSLIntegration_Test;

namespace qwen3
{

    class Grammar
    {
    public:
        Grammar();
        ~Grammar();

        // Parse GBNF grammar with pretokenized literals map
        static std::unique_ptr<Grammar> parse_impl(
            const std::string &gbnf_str,
            const std::map<std::string, std::vector<int32_t>> &pretokenized_literals);

        // Get valid tokens - original interface (for backward compatibility)
        std::vector<int32_t> get_valid_tokens() const;

        // Get valid tokens - vocab-aware interface (for sampling integration)
        // vocab[i] is the string representation of token ID i
        std::vector<int32_t> get_valid_tokens(const std::vector<std::string> &vocab) const;

        // Accept token - original interface (for backward compatibility)
        void accept_token(int32_t token_id);

        // Accept token - vocab-aware interface (for sampling integration)
        // vocab[token_id] gives the string that was accepted
        void accept_token(int32_t token_id, const std::vector<std::string> &vocab);

        bool is_accepting_state() const;
        void reset();

    private:
        friend class ::GrammarDev_SingleLiteral_Test;

        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace qwen3

#endif // GRAMMAR_H