#include "embed_mode.h"

#include "engine/embed.h"
#include "engine/model.h"
#include "../loader/tokenizer.h"
#include "../models/model_registry.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

std::string embed_mode_refusal(const CliArgs& args)
{
    struct Conflict { bool set; const char* flag; };
    const Conflict conflicts[] = {
        {args.chat_mode,                       "--chat"},
        {!args.image_path.empty(),             "--image"},
        {!args.mmproj_path.empty(),            "--mmproj"},
        {!args.grammar_file.empty(),           "--grammar-file"},
        {args.speculative,                     "--speculative"},
        {!args.save_session_path.empty(),      "--save-session"},
        {!args.load_session_path.empty(),      "--load-session"},
        {args.persistent_graph,                "--persistent-graph"},
        {!args.prefix_cache_dir.empty(),       "--prefix-cache"},
        {!args.image_prefix_cache_dir.empty(), "--image-prefix-cache"},
        {!args.image_embed_cache_dir.empty(),  "--image-embed-cache"},
        {args.kv_type != GGML_TYPE_F32,        "--kv-type / --kv-f16"},
        {!args.system_prompt.empty(),          "--system-prompt"},
    };
    for (const Conflict& c : conflicts)
        if (c.set)
            return std::string("--embed: expected no ") + c.flag +
                   " (an embedding has no chat, decode, cache or image path), actual: " +
                   c.flag + " given";
    const bool has_prompt = !args.prompt.empty();
    const bool has_file   = !args.embed_in_path.empty();
    if (has_prompt == has_file)
        return std::string("--embed: expected exactly one input, -p TEXT or --embed-in FILE, actual: ") +
               (has_prompt ? "both" : "neither");
    return "";
}

std::vector<std::string> read_embed_inputs(std::istream& in, const std::string& source)
{
    std::vector<std::string> texts;
    std::string line;
    size_t n = 0;
    while (std::getline(in, line)) {
        ++n;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty())
            throw std::runtime_error(
                "--embed-in: '" + source + "' line " + std::to_string(n) +
                ": expected one text, actual: an empty line");
        texts.push_back(line);
    }
    if (texts.empty())
        throw std::runtime_error(
            "--embed-in: '" + source + "': expected at least one line, actual: none");
    return texts;
}

std::string format_embedding_line(size_t index, size_t n_tokens,
                                  const std::vector<float>& embedding)
{
    std::string out = "{\"index\":" + std::to_string(index) +
                      ",\"n_tokens\":" + std::to_string(n_tokens) +
                      ",\"dim\":" + std::to_string(embedding.size()) +
                      ",\"embedding\":[";
    char buf[32];
    for (size_t i = 0; i < embedding.size(); ++i) {
        std::snprintf(buf, sizeof(buf), i ? ",%.9g" : "%.9g", embedding[i]);
        out += buf;
    }
    out += "]}";
    return out;
}

int run_embed(Model& model, const CliArgs& args)
{
    std::vector<std::string> texts;
    try {
        if (!args.prompt.empty()) {
            texts.push_back(args.prompt);
        } else {
            std::ifstream f(args.embed_in_path);
            if (!f) {
                std::cerr << "--embed-in: '" << args.embed_in_path
                          << "': expected a readable file, actual: cannot open\n";
                return 1;
            }
            texts = read_embed_inputs(f, args.embed_in_path);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    const ModelMetadata& meta = model.get_metadata();
    register_builtin_models();
    std::unique_ptr<ForwardPassBase> fp =
        create_forward_pass(model, &meta, args.context_length, 1);
    if (!dynamic_cast<IEmbeddingOutput*>(fp.get())) {
        std::cerr << "--embed: architecture '" << meta.architecture
                  << "' expected an embedding model, actual: a generation model\n";
        return 1;
    }
    if (args.flash_attn) {
        if (!fp->supports_flash_attn()) {
            std::cerr << "--flash-attn: architecture '" << meta.architecture
                      << "' does not support flash attention\n";
            return 1;
        }
        fp->set_prefill_attn_impl(ForwardPassBase::AttnImpl::Flash);
    }

    // Open the output only once the inputs and the model are known good, so a
    // refused run never truncates an existing file.
    std::ofstream file_out;
    if (!args.embed_out_path.empty()) {
        file_out.open(args.embed_out_path, std::ios::trunc);
        if (!file_out) {
            std::cerr << "--embed-out: '" << args.embed_out_path
                      << "': expected a writable file, actual: cannot open\n";
            return 1;
        }
    }
    std::ostream& out = file_out.is_open() ? static_cast<std::ostream&>(file_out) : std::cout;

    const TokenizerConfig cfg = lookup_tokenizer_config(meta.architecture);
    for (size_t i = 0; i < texts.size(); ++i) {
        const std::vector<int32_t> tokens = qinf::engine::embed::frame(
            cfg, meta, model.get_tokenizer()->encode(texts[i]));
        const std::vector<float> v = qinf::engine::embed::run(
            *fp, model.get_scheduler(), tokens, meta.architecture);
        out << format_embedding_line(i, tokens.size(), v) << "\n";
        if (args.verbose)
            std::cerr << "[embed] " << i << ": " << tokens.size() << " tokens\n";
    }
    out.flush();
    if (!out) {
        std::cerr << "--embed: writing the output expected to succeed, actual: write error\n";
        return 1;
    }
    return 0;
}
