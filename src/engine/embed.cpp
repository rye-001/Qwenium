#include "embed.h"

#include "graph_compute.h"

#include "ggml.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace qinf::engine::embed {

std::vector<int32_t> frame(const TokenizerConfig& cfg, const ModelMetadata& meta,
                           std::vector<int32_t> ids)
{
    if (cfg.add_bos_token) {
        if (meta.bos_token_id < 0)
            throw std::runtime_error(
                "embed::frame: 'bos_token_id': expected >= 0 (add_bos_token is true), got: " +
                std::to_string(meta.bos_token_id));
        ids.insert(ids.begin(), meta.bos_token_id);
    }
    if (cfg.add_eos_token) {
        if (meta.eos_token_id < 0)
            throw std::runtime_error(
                "embed::frame: 'eos_token_id': expected >= 0 (add_eos_token is true), got: " +
                std::to_string(meta.eos_token_id));
        ids.push_back(meta.eos_token_id);
    }
    return ids;
}

std::vector<float> pool_and_normalize(const std::vector<float>& per_token,
                                      uint32_t dim, uint32_t n_tokens,
                                      IEmbeddingOutput::Pooling pooling)
{
    if (dim == 0 || n_tokens == 0)
        throw std::runtime_error(
            "embed::pool_and_normalize: 'dim, n_tokens': expected both > 0, got: " +
            std::to_string(dim) + ", " + std::to_string(n_tokens));
    if (per_token.size() != static_cast<size_t>(dim) * n_tokens)
        throw std::runtime_error(
            "embed::pool_and_normalize: 'per_token' size: expected dim * n_tokens = " +
            std::to_string(static_cast<size_t>(dim) * n_tokens) + ", got: " +
            std::to_string(per_token.size()));

    std::vector<double> acc(dim, 0.0);
    if (pooling == IEmbeddingOutput::Pooling::Mean) {
        for (uint32_t t = 0; t < n_tokens; ++t)
            for (uint32_t d = 0; d < dim; ++d)
                acc[d] += per_token[static_cast<size_t>(t) * dim + d];
        for (double& a : acc) a /= static_cast<double>(n_tokens);
    } else {
        const size_t last = static_cast<size_t>(n_tokens - 1) * dim;
        for (uint32_t d = 0; d < dim; ++d) acc[d] = per_token[last + d];
    }

    double ss = 0.0;
    for (double a : acc) ss += a * a;
    const double norm = std::sqrt(ss);
    if (!(norm > 0.0) || !std::isfinite(norm))
        throw std::runtime_error(
            "embed::pool_and_normalize: pooled vector L2 norm: expected finite and > 0, got: " +
            std::to_string(norm));
    std::vector<float> out(dim);
    for (uint32_t d = 0; d < dim; ++d) out[d] = static_cast<float>(acc[d] / norm);
    return out;
}

std::vector<float> run(ForwardPassBase& fp, ggml_backend_sched_t sched,
                       const std::vector<int32_t>& tokens,
                       const std::string& architecture)
{
    auto* emb = dynamic_cast<IEmbeddingOutput*>(&fp);
    if (!emb)
        throw std::runtime_error(
            "embed::run: architecture '" + architecture +
            "' IEmbeddingOutput: expected implemented, got: not implemented (not an embedding model)");

    ggml_backend_sched_reset(sched);
    ggml_cgraph* gf = emb->build_embedding_graph(tokens);
    if (!ggml_backend_sched_alloc_graph(sched, gf))
        throw std::runtime_error("embed::run: graph alloc expected to succeed, actual failure");
    fp.set_prefill_inputs(gf, tokens, /*pos=*/0);
    require_compute_success(ggml_backend_sched_graph_compute(sched, gf), "embed::run");

    ggml_tensor* out = ggml_graph_get_tensor(gf, "embd_tokens");
    const uint32_t dim = emb->embedding_dim();
    const uint32_t n   = static_cast<uint32_t>(tokens.size());
    if (!out || out->type != GGML_TYPE_F32 || out->ne[0] != (int64_t)dim || out->ne[1] != (int64_t)n)
        throw std::runtime_error(
            "embed::run: graph output 'embd_tokens': expected F32 [" + std::to_string(dim) + ", " +
            std::to_string(n) + "], got: " +
            (out ? std::string(ggml_type_name(out->type)) + " [" + std::to_string(out->ne[0]) +
                   ", " + std::to_string(out->ne[1]) + "]"
                 : std::string("absent")));
    std::vector<float> per_token(static_cast<size_t>(dim) * n);
    ggml_backend_tensor_get(out, per_token.data(), 0, per_token.size() * sizeof(float));
    return pool_and_normalize(per_token, dim, n, emb->pooling());
}

EmbedInput image_input(const TokenizerConfig& cfg, const ModelMetadata& meta,
                       int32_t boi_id, int32_t soft_id, int32_t eoi_id,
                       std::vector<float> embd, uint32_t n_tokens)
{
    if (boi_id < 0 || soft_id < 0 || eoi_id < 0)
        throw std::runtime_error(
            "embed::image_input: 'boi_id, soft_id, eoi_id': expected all >= 0, got: " +
            std::to_string(boi_id) + ", " + std::to_string(soft_id) + ", " + std::to_string(eoi_id));
    if (n_tokens == 0)
        throw std::runtime_error("embed::image_input: 'n_tokens': expected > 0, got: 0");
    std::vector<int32_t> inner;
    inner.reserve(n_tokens + 2);
    inner.push_back(boi_id);
    inner.insert(inner.end(), n_tokens, soft_id);
    inner.push_back(eoi_id);
    EmbedInput in;
    in.tokens = frame(cfg, meta, std::move(inner));
    const int32_t lead = cfg.add_bos_token ? 1 : 0;
    in.images.push_back(EmbedImageSpan{lead + 1, n_tokens, std::move(embd)});
    return in;
}

std::vector<float> run(ForwardPassBase& fp, ggml_backend_sched_t sched,
                       EmbedInput input, const std::string& architecture)
{
    if (input.images.empty())
        return run(fp, sched, input.tokens, architecture);
    if (input.images.size() > 1)
        throw std::runtime_error(
            "embed::run: 'images': expected at most 1 per input (Seam B arms one span per "
            "build), got: " + std::to_string(input.images.size()));
    auto* img = dynamic_cast<IImageEmbeddable*>(&fp);
    if (!img)
        throw std::runtime_error(
            "embed::run: architecture '" + architecture +
            "' IImageEmbeddable: expected implemented (an image was given), got: not implemented");
    EmbedImageSpan& span = input.images.front();
    img->set_image_embeddings(std::move(span.embd), span.span_start, span.n_tokens);
    return run(fp, sched, input.tokens, architecture);
}

}  // namespace qinf::engine::embed
