# Qwen-3 CLI Executable
add_executable(qwenium-cli
    complete.cpp
    chat.cpp
    main.cpp
)

target_link_libraries(qwenium-cli PRIVATE
    qinf-models
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