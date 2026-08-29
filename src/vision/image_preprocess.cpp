#include "image_preprocess.h"

namespace qinf::vision {

ImagePreprocess gemma3_preprocess(int target) {
    ImagePreprocess pp;
    pp.sizing       = ImagePreprocess::Sizing::FixedSquarePadCeil;
    pp.fixed_target = target;
    pp.mean[0]   = pp.mean[1]   = pp.mean[2]   = 0.5f;
    pp.stddev[0] = pp.stddev[1] = pp.stddev[2] = 0.5f;
    return pp;
}

ImagePreprocess gemma4uv_preprocess(int align, int min_tokens, int max_tokens) {
    ImagePreprocess pp;
    pp.sizing     = ImagePreprocess::Sizing::DynSmartResize;
    pp.align      = align;
    pp.min_tokens = min_tokens;
    pp.max_tokens = max_tokens;
    pp.mean[0]   = pp.mean[1]   = pp.mean[2]   = 0.0f;   // verified mmproj kv [0,0,0]
    pp.stddev[0] = pp.stddev[1] = pp.stddev[2] = 1.0f;   // verified mmproj kv [1,1,1]
    return pp;
}

ImagePreprocess qwen3vl_preprocess(int align, int min_tokens, int max_tokens) {
    ImagePreprocess pp;
    pp.sizing     = ImagePreprocess::Sizing::DynSmartResize;
    pp.align      = align;
    pp.min_tokens = min_tokens;
    pp.max_tokens = max_tokens;
    pp.mean[0]   = pp.mean[1]   = pp.mean[2]   = 0.5f;   // clip.vision.image_mean
    pp.stddev[0] = pp.stddev[1] = pp.stddev[2] = 0.5f;   // clip.vision.image_std
    return pp;
}

}  // namespace qinf::vision
