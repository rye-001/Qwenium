#include "embedding_service.h"

#include "engine/embed.h"
#include "engine/model.h"
#include "loader/tokenizer.h"
#include "models/model_registry.h"
#include "models/i_embedding_output.h"
#include "models/i_image_embeddable.h"
#include "image/image_loader.h"
#include "image_data_uri.h"
#include "vision/vision_loader.h"
#include "vision/vision_model.h"
#include "vision/vision_profile.h"

#include <cstring>
#include <stdexcept>

using json = nlohmann::json;

namespace qinf::server {

namespace {

HttpReply refuse(int status, const std::string& message) {
    return HttpReply{status, json{{"error", message}}};
}

}  // namespace

std::string base64_f32(const std::vector<float>& v)
{
    static const char* tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    // Apple Silicon and x86 are little-endian; the wire format is too.
    std::vector<unsigned char> bytes(v.size() * sizeof(float));
    std::memcpy(bytes.data(), v.data(), bytes.size());
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < bytes.size(); i += 3) {
        const uint32_t n = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
        out += tbl[(n >> 18) & 63]; out += tbl[(n >> 12) & 63];
        out += tbl[(n >> 6) & 63];  out += tbl[n & 63];
    }
    if (i < bytes.size()) {
        uint32_t n = bytes[i] << 16;
        if (i + 1 < bytes.size()) n |= bytes[i + 1] << 8;
        out += tbl[(n >> 18) & 63]; out += tbl[(n >> 12) & 63];
        out += (i + 1 < bytes.size()) ? tbl[(n >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

HttpReply handle_embeddings(const std::string& request_body,
                            const std::string& model_id, uint32_t dim,
                            uint32_t max_tokens,
                            const TokenizeFn& tokenize, const EmbedFn& embed,
                            const ImageFns* images)
{
    json body;
    try {
        body = json::parse(request_body);
    } catch (const json::parse_error&) {
        return refuse(400, "Invalid JSON");
    }
    if (!body.is_object()) return refuse(400, "request body: expected a JSON object");

    // input: a string, or a non-empty array of texts and (with an image tower)
    // image items.
    if (!body.contains("input"))
        return refuse(400, "'input': expected a string or an array of strings, actual: absent");
    const json& input = body["input"];
    struct Item { bool image = false; std::string value; };   // value: the text, or image_url.url
    std::vector<Item> items;
    if (input.is_string()) {
        items.push_back({false, input.get<std::string>()});
    } else if (input.is_array()) {
        if (input.empty())
            return refuse(400, "'input': expected at least one text, actual: an empty array");
        if (input.size() > kMaxInputs)
            return refuse(413, "'input': expected at most " + std::to_string(kMaxInputs) +
                               " texts, actual: " + std::to_string(input.size()));
        for (size_t i = 0; i < input.size(); ++i) {
            const std::string at = "'input[" + std::to_string(i) + "]'";
            const json& it = input[i];
            if (it.is_number_integer() || it.is_array())
                return refuse(400, "'input': expected text, actual: token ids (not supported — "
                                   "send the text)");
            if (it.is_object()) {
                if (it.value("type", std::string()) != "image_url" || !it.contains("image_url") ||
                    !it["image_url"].is_object() || !it["image_url"].contains("url") ||
                    !it["image_url"]["url"].is_string())
                    return refuse(400, at + ": expected a string or {\"type\":\"image_url\","
                                       "\"image_url\":{\"url\":\"data:...\"}}, actual: " + it.dump());
                if (!images)
                    return refuse(400, at + ": expected a text (this server has no image tower; "
                                       "start it with --embed --mmproj), actual: an image_url item");
                items.push_back({true, it["image_url"]["url"].get<std::string>()});
                continue;
            }
            if (!it.is_string())
                return refuse(400, at + ": expected a string, actual: " + std::string(it.type_name()));
            items.push_back({false, it.get<std::string>()});
        }
    } else {
        return refuse(400, "'input': expected a string or an array of strings, actual: " +
                           std::string(input.type_name()));
    }
    for (size_t i = 0; i < items.size(); ++i)
        if (!items[i].image && items[i].value.empty())
            return refuse(400, "'input[" + std::to_string(i) + "]': expected a non-empty text, "
                               "actual: an empty string");

    const std::string format = body.value("encoding_format", std::string("float"));
    if (format != "float" && format != "base64")
        return refuse(400, "'encoding_format': expected \"float\" or \"base64\", actual: \"" +
                           format + "\"");
    if (body.contains("dimensions")) {
        const json& d = body["dimensions"];
        if (!d.is_number_integer() || d.get<int64_t>() != static_cast<int64_t>(dim))
            return refuse(400, "'dimensions': expected " + std::to_string(dim) +
                               " (this model's width; truncation is not supported), actual: " +
                               d.dump());
    }

    // Tokenize every text and prepare every image first: a refused request
    // costs no forward pass.
    std::vector<std::vector<int32_t>> tokens(items.size());
    size_t total = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        size_t n = 0;
        if (items[i].image) {
            try {
                n = images->prepare(items[i].value);
            } catch (const std::exception& e) {
                return refuse(400, "'input[" + std::to_string(i) + "]': " + e.what());
            }
        } else {
            tokens[i] = tokenize(items[i].value);
            n = tokens[i].size();
        }
        if (n > max_tokens)
            return refuse(413, "'input[" + std::to_string(i) + "]': expected at most " +
                               std::to_string(max_tokens) + " tokens (--ctx), actual: " +
                               std::to_string(n));
        total += n;
    }

    json data = json::array();
    try {
        for (size_t i = 0; i < items.size(); ++i) {
            const std::vector<float> v = items[i].image ? images->embed(items[i].value)
                                                        : embed(tokens[i]);
            json item = {{"object", "embedding"}, {"index", i}};
            if (format == "base64") item["embedding"] = base64_f32(v);
            else                    item["embedding"] = v;
            data.push_back(std::move(item));
        }
    } catch (const std::exception& e) {
        return refuse(500, std::string("embedding failed: ") + e.what());
    }

    return HttpReply{200, json{
        {"object", "list"},
        {"model", model_id},
        {"data", std::move(data)},
        {"usage", {{"prompt_tokens", total}, {"total_tokens", total}}},
    }};
}

// ── EmbeddingService ─────────────────────────────────────────────────────────

struct EmbeddingService::Vision {
    qinf::vision::VisionModel   model;
    qinf::vision::VisionLoader  loader;
    qinf::vision::VisionProfile profile;
};

EmbeddingService::EmbeddingService(const std::string& model_path, uint32_t max_tokens, bool flash,
                                   const std::string& mmproj_path)
    : model_(std::make_unique<Model>()), max_tokens_(max_tokens)
{
    register_builtin_models();
    model_->load_metadata(model_path);
    model_->load_tensors();
    const ModelMetadata& meta = model_->get_metadata();
    fp_ = create_forward_pass(*model_, &meta, max_tokens_, 1);
    auto* emb = dynamic_cast<IEmbeddingOutput*>(fp_.get());
    if (!emb)
        throw std::runtime_error(
            "--embed: architecture '" + meta.architecture +
            "' expected an embedding model, actual: a generation model (start without --embed)");
    if (flash) fp_->set_prefill_attn_impl(ForwardPassBase::AttnImpl::Flash);
    dim_ = emb->embedding_dim();

    if (!mmproj_path.empty()) {
        if (!dynamic_cast<IImageEmbeddable*>(fp_.get()))
            throw std::runtime_error(
                "--mmproj: architecture '" + meta.architecture +
                "' expected an embedding model that hosts images (IImageEmbeddable), actual: text only");
        vision_ = std::make_unique<Vision>();
        vision_->loader.parse_metadata(mmproj_path, vision_->model);
        if (vision_->model.config().projector_type != qinf::vision::VisionProjectorType::Gemma4V)
            throw std::runtime_error(
                "--mmproj: '" + mmproj_path + "' expected a gemma4v projector (an embedding model's "
                "image tower), actual: " + qinf::vision::to_string(vision_->model.config().projector_type));
        ggml_backend_t backend = model_->get_backend_metal() ? model_->get_backend_metal()
                                                             : model_->get_backend_cpu();
        vision_->loader.load_tensors(vision_->model, backend);
        vision_->profile = qinf::vision::make_vision_profile(vision_->model, backend, meta.id_to_token,
                                                             "--embed: parameter '--mmproj'");
    }

    // The id clients see in "model": the file name without its directory or .gguf.
    model_id_ = model_path.substr(model_path.find_last_of('/') + 1);
    if (model_id_.size() > 5 && model_id_.compare(model_id_.size() - 5, 5, ".gguf") == 0)
        model_id_.resize(model_id_.size() - 5);
}

EmbeddingService::~EmbeddingService() = default;

HttpReply EmbeddingService::handle(const std::string& request_body)
{
    std::lock_guard<std::mutex> lock(mu_);
    const ModelMetadata& meta = model_->get_metadata();
    const TokenizerConfig cfg = lookup_tokenizer_config(meta.architecture);
    ImageFns image_fns;
    if (vision_) {
        auto bitmap_of = [this](const std::string& url) {
            const qinf::DecodedImage img = qinf::decode_image_data_uri(url);
            return qinf::image::load_image_to_bitmap_from_memory(img.bytes.data(), img.bytes.size(),
                                                                 vision_->profile.preprocess);
        };
        image_fns.prepare = [this, bitmap_of, &cfg](const std::string& url) -> size_t {
            const size_t n = vision_->profile.encoder->mm_tokens_for(bitmap_of(url));
            return n + 2 + (cfg.add_bos_token ? 1 : 0) + (cfg.add_eos_token ? 1 : 0);
        };
        image_fns.embed = [this, bitmap_of, &cfg, &meta](const std::string& url) {
            const auto bmp = bitmap_of(url);
            const auto& prof = vision_->profile;
            return qinf::engine::embed::run(
                *fp_, model_->get_scheduler(),
                qinf::engine::embed::image_input(cfg, meta, prof.boi_id, prof.soft_id, prof.eoi_id,
                                                 prof.encoder->encode(bmp), prof.encoder->mm_tokens_for(bmp)),
                meta.architecture);
        };
    }
    HttpReply reply = handle_embeddings(
        request_body, model_id_, dim_, max_tokens_,
        [&](const std::string& text) {
            return qinf::engine::embed::frame(cfg, meta, model_->get_tokenizer()->encode(text));
        },
        [&](const std::vector<int32_t>& tokens) {
            return qinf::engine::embed::run(*fp_, model_->get_scheduler(), tokens, meta.architecture);
        },
        vision_ ? &image_fns : nullptr);
    if (reply.status == 200) ++requests_served_;
    return reply;
}

}  // namespace qinf::server
