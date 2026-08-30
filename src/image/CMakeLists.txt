# Host-side image pipeline: decode → resample → normalize → Bitmap, plus
# token-level marker expansion. Deliberately NOT src/vision/ (that is the
# encoder subsystem) and not src/cli/ — both front ends consume this.
#
# Self-contained by design: image_loader needs only stb, image_prompt only its
# own header. No ggml, no model, no vision. That is what lets it be a leaf
# library every consumer can link rather than a source each one recompiles,
# which is what cli/, server/ and three test targets were doing until
# 2026-08-30.
add_library(qinf-image STATIC
    image_loader.cpp
    image_prompt.cpp
)

target_include_directories(qinf-image PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
)

set_target_properties(qinf-image PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
