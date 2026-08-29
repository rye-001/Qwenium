// Corpus parse-check (Phase 1 correctness guard).
// Feeds each accepted JS example char-by-char through GrammarVocab and
// asserts is_accepting_state() at end. Usage:
//   test-corpus-parse <grammar.gbnf> <corpus.txt>
// Exit 0 iff every example parses; prints per-example PASS/FAIL.

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/sampling/grammar_vocab.h"

static std::string slurp(const std::string& p) {
    std::ifstream f(p);
    if (!f) { std::cerr << "cannot open " << p << "\n"; std::exit(2); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

// Split the corpus into JS code blocks. Structure repeats:
//   <sep>\n [NN] name\n User: ...\n <sep>\n <CODE...> \n  (then next <sep>)
// A separator line is all '='. The segment right after a header segment
// (one containing "User:") is the code to test.
static std::vector<std::string> extract_examples(const std::string& corpus) {
    std::vector<std::string> lines;
    {
        std::stringstream ss(corpus);
        std::string ln;
        while (std::getline(ss, ln)) lines.push_back(ln);
    }
    auto is_sep = [](const std::string& s) {
        if (s.size() < 4) return false;
        for (char c : s) if (c != '=') return false;
        return true;
    };
    std::vector<std::string> segments;
    std::string cur;
    for (const auto& ln : lines) {
        if (is_sep(ln)) { segments.push_back(cur); cur.clear(); }
        else { cur += ln; cur += '\n'; }
    }
    segments.push_back(cur);

    std::vector<std::string> examples;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (segments[i].find("User:") != std::string::npos &&
            segments[i].find('[') != std::string::npos) {
            if (i + 1 < segments.size()) {
                // Trim leading/trailing whitespace-only lines.
                std::string code = segments[i + 1];
                size_t b = code.find_first_not_of(" \t\r\n");
                size_t e = code.find_last_not_of(" \t\r\n");
                if (b != std::string::npos)
                    examples.push_back(code.substr(b, e - b + 1));
            }
        }
    }
    return examples;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <grammar.gbnf> <corpus.txt>\n";
        return 2;
    }
    std::string gbnf   = slurp(argv[1]);
    std::string corpus = slurp(argv[2]);

    // Single-char vocab: vocab[c] == std::string(1, c) for 0..255.
    std::vector<std::string> vocab(256);
    for (int c = 0; c < 256; ++c) vocab[c] = std::string(1, static_cast<char>(c));

    auto examples = extract_examples(corpus);
    if (examples.empty()) { std::cerr << "no examples extracted\n"; return 2; }

    int pass = 0, fail = 0, idx = 0;
    for (const auto& ex : examples) {
        ++idx;
        auto g = qinf::GrammarVocab::parse_impl(gbnf);
        if (!g) { std::cerr << "[" << idx << "] FAIL grammar parse\n"; ++fail; continue; }
        bool dead = false;
        for (unsigned char ch : ex) {
            g->accept_token(static_cast<int32_t>(ch), vocab);
            // Heuristic dead-state check: if no token can advance AND not
            // accepting, the path is dead. get_valid_tokens needs a trie;
            // instead we rely on the final accepting check — a wrong grammar
            // simply won't be accepting at the end.
            (void)dead;
        }
        bool ok = g->is_accepting_state();
        std::cout << "[" << (idx < 10 ? "0" : "") << idx << "] "
                  << (ok ? "PASS" : "FAIL")
                  << "  (" << ex.size() << " chars)\n";
        if (ok) ++pass; else { ++fail;
            std::cout << "    first 60: «" << ex.substr(0, 60) << "»\n"; }
    }
    std::cout << "------------------------------------------------------\n";
    std::cout << " corpus: " << pass << " passed, " << fail << " failed ("
              << examples.size() << " examples)\n";
    return fail == 0 ? 0 : 1;
}
