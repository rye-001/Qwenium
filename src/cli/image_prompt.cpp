#include "image_prompt.h"

#include <stdexcept>
#include <string>

namespace qinf::cli {

ExpandedImagePrompt expand_image_markers(
    const std::vector<int32_t>& tokens,
    int32_t  start_of_image_id,
    int32_t  soft_token_id,
    int32_t  end_of_image_id,
    uint32_t n_image_tokens)
{
    // Locate the single start-of-image marker.
    size_t marker = 0;
    uint32_t count = 0;
    for (size_t i = 0; i < tokens.size(); ++i)
        if (tokens[i] == start_of_image_id) { ++count; marker = i; }

    if (count != 1)
        throw std::runtime_error(
            "expand_image_markers: parameter 'tokens': expected exactly 1 "
            "start_of_image token (id " + std::to_string(start_of_image_id) +
            "; single-image scope, multi-image is Phase 7), got: " +
            std::to_string(count));

    ExpandedImagePrompt out;
    out.tokens.reserve(tokens.size() + n_image_tokens + 1);

    // Everything up to and including the start-of-image marker.
    out.tokens.insert(out.tokens.end(), tokens.begin(),
                      tokens.begin() + static_cast<long>(marker) + 1);

    // The soft-token span begins right after the start-of-image marker.
    out.span_start = static_cast<int32_t>(out.tokens.size());
    out.span_len   = n_image_tokens;
    out.tokens.insert(out.tokens.end(), n_image_tokens, soft_token_id);

    // Close the block, then the remainder of the original stream.
    out.tokens.push_back(end_of_image_id);
    out.tokens.insert(out.tokens.end(),
                      tokens.begin() + static_cast<long>(marker) + 1, tokens.end());

    return out;
}

}  // namespace qinf::cli
