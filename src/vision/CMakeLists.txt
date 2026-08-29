# Vision subsystem — SigLIP encoder + mmproj loader + projection stage.
#
# Standalone, connected to the text model only at the embedding boundary
# (Phase 3's ImageEmbeddingsInput). See docs/plan-gemma-vision-impl.md.
#
# C3 contract: this subsystem is its own ggml graph, its own ggml context,
# its own scheduler. It shares the engine's Metal backend (one Metal device,
# multiple schedulers) but nothing else.

add_library(qinf-vision STATIC
    vision_model.cpp
    vision_loader.cpp
    siglip_encoder.cpp
    gemma4uv_encoder.cpp
    qwen3vl_encoder.cpp
    image_preprocess.cpp
    vision_profile.cpp
)

target_include_directories(qinf-vision PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(qinf-vision PUBLIC ggml qinf-loader)

set_target_properties(qinf-vision PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
