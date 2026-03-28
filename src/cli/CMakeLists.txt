# Qwen-3 CLI Executable
add_executable(qwen3-cli
    complete.cpp
    chat.cpp
    main.cpp
)

target_link_libraries(qwen3-cli PRIVATE
    qwen3-core
    qwen3-kv-cache
    sampling
    ggml
    Threads::Threads
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(qwen3-cli PRIVATE OpenMP::OpenMP_CXX)
endif()

# Set executable properties
set_target_properties(qwen3-cli PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "qwen3"
)

# Platform-specific linking
if(APPLE)
    target_link_libraries(qwen3-cli PRIVATE "-framework Accelerate")
elseif(UNIX)
    target_link_libraries(qwen3-cli PRIVATE m dl)
endif()

# Static linking option
if(QWEN3_STATIC_LINK)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_link_options(qwen3-cli PRIVATE -static-libgcc -static-libstdc++)
    endif()
endif()