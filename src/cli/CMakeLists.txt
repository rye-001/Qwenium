# Qwen-3 CLI Executable
add_executable(qwenium-cli
    complete.cpp
    chat.cpp
    main.cpp
    session_mode.cpp
    image_loader.cpp
    image_prompt.cpp
    ${CMAKE_SOURCE_DIR}/src/core/decode_step.cpp
    ${CMAKE_SOURCE_DIR}/src/core/decode_plan.cpp
    ${CMAKE_SOURCE_DIR}/src/core/multimodal_prefill.cpp
    ${CMAKE_SOURCE_DIR}/src/core/image_embedding_cache.cpp
    ${CMAKE_SOURCE_DIR}/src/core/slot_snapshot.cpp
)

target_link_libraries(qwenium-cli PRIVATE
    qinf-models
    qinf-vision
    qinf-state
    qinf-session
    core
    sampling
    ggml
    Threads::Threads
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(qwenium-cli PRIVATE OpenMP::OpenMP_CXX)
endif()

# Set executable properties
set_target_properties(qwenium-cli PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "qwenium"
)

# Platform-specific linking
if(APPLE)
    target_link_libraries(qwenium-cli PRIVATE "-framework Accelerate")
elseif(UNIX)
    target_link_libraries(qwenium-cli PRIVATE m dl)
endif()

# Static linking option
if(QWEN3_STATIC_LINK)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_link_options(qwenium-cli PRIVATE -static-libgcc -static-libstdc++)
    endif()
endif()