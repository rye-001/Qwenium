// capture_tool_chat_ref_llamacpp.cpp — the second tool-request render
// reference (docs/plan-tool-calling.md gate G1): llama.cpp's own Jinja engine
// (common/jinja), run directly on each fixture directory's template.jinja with
// the same variables py/capture_tool_chat_ref.py gives Hugging Face's renderer.
// None of llama.cpp's chat-format workarounds run — only the template. Writes
// <case>.llamacpp.txt (or .llamacpp.err) next to the .hf.txt files.
//
// Built out of tree against a llama.cpp checkout (no model, no libllama):
//   L=~/dev/projects/llama/llama.cpp
//   c++ -std=c++17 -O1 -I$L/common -I$L/vendor -I$L/ggml/include \
//       scripts/capture_tool_chat_ref_llamacpp.cpp $L/common/jinja/*.cpp \
//       $L/common/json.cpp $L/common/unicode.cpp -o $SCRATCH/capture-llamacpp
//   $SCRATCH/capture-llamacpp tests/fixtures/chat_tools <llama.cpp commit>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "json.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"

namespace fs = std::filesystem;

// common/json.cpp asserts through GGML_ASSERT; this tool links no ggml, so the
// one symbol it needs is defined here: print and abort, as ggml does.
extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
    std::fprintf(stderr, "%s:%d: ggml_abort (%s)\n", file, line, fmt);
    std::abort();
}

static std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("open '" + p.string() + "': expected readable, actual: failed");
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

static void spit(const fs::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary);
    f << s;
}

// "bos='<bos>' eos='<turn|>'" from the first line of models.txt.
static std::string field(const std::string& line, const std::string& key) {
    const std::string k = key + "='";
    const size_t a = line.find(k);
    if (a == std::string::npos) throw std::runtime_error("models.txt: expected " + key + "='...', actual: " + line);
    const size_t b = line.find('\'', a + k.size());
    return line.substr(a + k.size(), b - a - k.size());
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s FIXTURE_DIR LLAMACPP_COMMIT\n", argv[0]);
        return 2;
    }
    const fs::path fx = argv[1];
    const common_json cases = common_json::parse(slurp(fx / "cases.json"));
    int n_dirs = 0;
    for (const auto& ent : fs::directory_iterator(fx)) {
        if (!ent.is_directory() || !fs::exists(ent.path() / "template.jinja")) continue;
        const std::string src = slurp(ent.path() / "template.jinja");
        std::string first;
        std::getline(std::istringstream(slurp(ent.path() / "models.txt")) >> std::ws, first);
        const std::string bos = field(first, "bos"), eos = field(first, "eos");

        jinja::lexer lexer;
        const jinja::lexer_result lexed = lexer.tokenize(src);
        const jinja::program prog = jinja::parse_from_tokens(lexed);

        for (size_t i = 0; i < cases["cases"].size(); ++i) {
            const common_json& c = cases["cases"][i];
            const std::string name = c["name"].get<std::string>();
            common_json inp = common_json::object();
            inp["messages"] = c["messages"];
            // declare_tools false: tools undefined, the template's no-tools branch.
            if (c.contains("tools") && c.value("declare_tools", true)) inp["tools"] = cases["tool_sets"][c["tools"].get<std::string>()];
            inp["add_generation_prompt"] = true;
            inp["bos_token"] = bos;
            inp["eos_token"] = eos;
            if (c.contains("kwargs"))
                for (auto it = c["kwargs"].begin(); it != c["kwargs"].end(); ++it)
                    inp[it.key()] = it.value();
            const fs::path base = ent.path() / name;
            fs::remove(base.string() + ".llamacpp.txt");
            fs::remove(base.string() + ".llamacpp.err");
            try {
                jinja::context ctx(lexed.source);
                jinja::global_from_json(ctx, inp, /*mark_input=*/false);
                jinja::runtime rt(ctx);
                const jinja::value out = rt.execute(prog);
                // Keep the gathered value alive: ranging over a member of the
                // returned temporary would dangle.
                const jinja::value_string parts = jinja::runtime::gather_string_parts(out);
                std::string text;
                for (const auto& part : parts->val_str.parts) text += part.val;
                spit(base.string() + ".llamacpp.txt", text);
            } catch (const std::exception& e) {
                std::string msg = e.what();
                for (char& ch : msg) if (ch == '\n') ch = ' ';
                spit(base.string() + ".llamacpp.err", msg + "\n");
            }
        }
        spit(ent.path() / "llamacpp_commit.txt", std::string(argv[2]) + "\n");
        ++n_dirs;
    }
    std::printf("%d templates x %zu cases rendered\n", n_dirs, cases["cases"].size());
    return 0;
}
