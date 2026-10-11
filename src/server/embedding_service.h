#pragma once
// embedding_service.h — POST /v1/embeddings for an embedding model
// (qwenium-server --embed).
//
// Responsibility: the OpenAI embeddings contract over engine/embed.
//   Request:  {"input": "text" | ["text", ...], "model"?: any,
//              "encoding_format"?: "float" | "base64", "dimensions"?: int,
//              "user"?: any}
//   Response: {"object":"list","model":<id>,
//              "data":[{"object":"embedding","index":i,"embedding":[...] | "<b64>"}],
//              "usage":{"prompt_tokens":N,"total_tokens":N}}
//   "base64" is the little-endian float32 bytes of the vector — what the
//   OpenAI Python client requests by default and decodes.
//   Refused (400): token-id inputs, an empty input or text, a non-string item,
//   an unknown encoding_format, `dimensions` other than the model's (no
//   truncation — the engine returns the vector the model was trained to give).
//   Refused (413): more than kMaxInputs texts, or a text longer than the
//   server's --ctx in tokens (named by index and count). Every input is
//   validated and tokenized before any is embedded, so a refused request
//   costs no compute.
//   Errors keep this server's shape: {"error": "<message>"}.
//
// One request embeds its texts one after another on the server's single
// scheduler; concurrent requests serialise on a mutex. Batching several texts
// into one pass is a later optimisation, not a contract.
//
// Unit test: tests/unit/test_embedding_service.cpp (the handler through fake
//   tokenize/embed callbacks; the service end to end with
//   EMBEDDINGGEMMA2_MODEL_PATH).

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

class Model;
class ForwardPassBase;

namespace qinf::server {

constexpr size_t kMaxInputs = 2048;   // OpenAI's own per-request ceiling

struct HttpReply {
    int status = 200;
    nlohmann::json body;
};

// text → framed token ids (BOS/EOS as the architecture wants).
using TokenizeFn = std::function<std::vector<int32_t>(const std::string&)>;
// framed token ids → one L2-normalised vector of length `dim`.
using EmbedFn = std::function<std::vector<float>(const std::vector<int32_t>&)>;

// The whole request → reply mapping, model-free.
HttpReply handle_embeddings(const std::string& request_body,
                            const std::string& model_id, uint32_t dim,
                            uint32_t max_tokens,
                            const TokenizeFn& tokenize, const EmbedFn& embed);

// Little-endian float32 bytes, standard base64 with padding.
std::string base64_f32(const std::vector<float>& v);

// Owns the loaded embedding model for the server's lifetime.
class EmbeddingService {
public:
    // Loads `model_path`; fail-loud (throws) unless its recipe implements
    // IEmbeddingOutput. `max_tokens` bounds one text; `flash` selects flash
    // attention for the pass.
    EmbeddingService(const std::string& model_path, uint32_t max_tokens, bool flash);
    ~EmbeddingService();

    HttpReply handle(const std::string& request_body);

    const std::string& model_id() const { return model_id_; }
    uint32_t dim() const { return dim_; }
    uint32_t max_tokens() const { return max_tokens_; }
    uint64_t requests_served() const { return requests_served_.load(); }

private:
    std::unique_ptr<Model>           model_;
    std::unique_ptr<ForwardPassBase> fp_;
    std::string                      model_id_;
    uint32_t                         dim_ = 0;
    uint32_t                         max_tokens_ = 0;
    std::atomic<uint64_t>            requests_served_{0};   // read by /health, unlocked
    std::mutex                       mu_;
};

}  // namespace qinf::server
