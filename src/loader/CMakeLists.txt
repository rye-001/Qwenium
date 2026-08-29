# GGUF loader, tokenizer, chat templates, and the GGUF value bag / mmap wrapper.

add_library(qinf-loader STATIC
    gguf_loader.cpp
    tokenizer.cpp
    chat_template.cpp
    multimodal_check.cpp
    channel_filter.cpp
)

target_include_directories(qinf-loader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(qinf-loader PUBLIC ggml)

set_target_properties(qinf-loader PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
