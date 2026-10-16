#include "embed_mode.h"

#include "engine/embed.h"
#include "engine/model.h"
#include "../loader/tokenizer.h"
#include "../models/model_registry.h"
#include "../models/i_image_embeddable.h"
#include "image/image_loader.h"
#include "vision/vision_loader.h"
#include "vision/vision_model.h"
#include "vision/vision_profile.h"

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
                   " (an embedding has no chat, decode or cache path), actual: " +
                   c.flag + " given";
    // An image is embedded with the embedding model's own image tower
    // (docs/plan-embed-image.md G6): --image and --mmproj come together.
    const bool has_image  = !args.image_path.empty();
    const bool has_mmproj = !args.mmproj_path.empty();
    if (has_image != has_mmproj)
        return has_image
            ? std::string("--embed --image: expected --mmproj (the embedding model's image tower), actual: none")
            : std::string("--embed --mmproj: expected --image PATH (the image to embed), actual: none");
    const bool has_prompt = !args.prompt.empty();
    const bool has_file   = !args.embed_in_path.empty();
    const int n_inputs = (has_prompt ? 1 : 0) + (has_file ? 1 : 0) + (has_image ? 1 : 0);
    if (n_inputs != 1)
        return std::string("--embed: expected exactly one input, -p TEXT, --embed-in FILE or "
                           "--image PATH, actual: ") +
               (n_inputs == 0 ? "none" : std::to_string(n_inputs));
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
        if (!args.image_path.empty()) {
            // one image: no texts
        } else if (!args.prompt.empty()) {
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
    if (!args.image_path.empty()) {
        // The embedding model's image tower: refused unless it is gemma4v and
        // the recipe hosts images. Output: one record, index 0, n_tokens = the
        // framed sequence ([BOS] <|image> soft×n <image|> [EOS]).
        if (!dynamic_cast<IImageEmbeddable*>(fp.get())) {
            std::cerr << "--embed --image: architecture '" << meta.architecture
                      << "' expected an embedding model that hosts images, actual: text only\n";
            return 1;
        }
        qinf::vision::VisionModel vmodel;
        qinf::vision::VisionLoader vloader;
        vloader.parse_metadata(args.mmproj_path, vmodel);
        if (vmodel.config().projector_type != qinf::vision::VisionProjectorType::Gemma4V) {
            std::cerr << "--mmproj: '" << args.mmproj_path
                      << "' expected a gemma4v projector (an embedding model's image tower), actual: "
                      << qinf::vision::to_string(vmodel.config().projector_type) << "\n";
            return 1;
        }
        ggml_backend_t backend = model.get_backend_metal() ? model.get_backend_metal()
                                                           : model.get_backend_cpu();
        vloader.load_tensors(vmodel, backend);
        qinf::vision::VisionProfile prof = qinf::vision::make_vision_profile(
            vmodel, backend, meta.id_to_token, "--embed: parameter '--mmproj'");
        const auto bmp = qinf::image::load_image_to_bitmap(args.image_path, prof.preprocess);
        const uint32_t n = prof.encoder->mm_tokens_for(bmp);
        qinf::engine::embed::EmbedInput in = qinf::engine::embed::image_input(
            cfg, meta, prof.boi_id, prof.soft_id, prof.eoi_id, prof.encoder->encode(bmp), n);
        const size_t n_tokens = in.tokens.size();
        const std::vector<float> v =
            qinf::engine::embed::run(*fp, model.get_scheduler(), std::move(in), meta.architecture);
        out << format_embedding_line(0, n_tokens, v) << "\n";
        if (args.verbose)
            std::cerr << "[embed] image '" << args.image_path << "': " << n << " soft tokens, "
                      << n_tokens << " framed\n";
    }
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
