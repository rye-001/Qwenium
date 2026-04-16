# tests/CMakeLists.txt
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# Define the unit test executable
add_executable(qwen3-unit-tests
    unit/test_tokenizer.cpp
    unit/test_vocab_file_loading.cpp
    unit/test_qwen35_metadata.cpp
    unit/test_qwen35_tensor_loading.cpp
    unit/test_ssm_state_cache.cpp
    unit/test_qwen35_tokenizer.cpp
    unit/test_qwen35_hybrid_cache.cpp
    unit/test_qwen35_forward_attn.cpp
    unit/test_qwen35_generation.cpp
    unit/test_grammar_vocab.cpp
    unit/test_turboquant_generation.cpp
)

# Link libraries for unit tests
target_link_libraries(qwen3-unit-tests PRIVATE
    GTest::gtest_main
    qwen3-core
    sampling
    ggml
)

# Discover and add unit tests to CTest
include(GoogleTest)
gtest_discover_tests(qwen3-unit-tests)

# Define the integration test executable
add_executable(qwen3-integration-tests
    integration/test_e2e_generation.cpp
)

# Link libraries for integration tests
target_link_libraries(qwen3-integration-tests PRIVATE
    GTest::gtest_main
    qwen3-core
    sampling
    ggml
)

# Discover and add integration tests to CTest
gtest_discover_tests(qwen3-integration-tests)

# Define the standalone order management test (Non-GTest)
add_executable(qwen3-order-mngmt-test
    integration/test_order_management_generation.cpp
)

# Link libraries
target_link_libraries(qwen3-order-mngmt-test PRIVATE
    qwen3-core
    sampling
    ggml
)

# Define the standalone order management test (Non-GTest)
add_executable(qwen3-parallel-order-mngmt-test
    integration/test_parallel_order_management.cpp
)

# Link libraries
target_link_libraries(qwen3-parallel-order-mngmt-test PRIVATE
    qwen3-core
    sampling
    ggml
)

# TurboQuant unit tests (GTest, no model needed)
add_executable(qwen3-turboquant-tests
    unit/test_turboquant.cpp
)

target_link_libraries(qwen3-turboquant-tests PRIVATE
    GTest::gtest_main
    qwen3-kv-cache
)

gtest_discover_tests(qwen3-turboquant-tests)

# Compressed KV store unit tests (GTest, no model needed)
add_executable(qwen3-compressed-kv-tests
    unit/test_compressed_kv_store.cpp
)

target_link_libraries(qwen3-compressed-kv-tests PRIVATE
    GTest::gtest_main
    qwen3-kv-cache
)

gtest_discover_tests(qwen3-compressed-kv-tests)

# SnapKV CLI config tests (GTest, no model needed)
add_executable(qwen3-snapkv-cli-tests
    unit/test_snapkv_cli.cpp
)

target_link_libraries(qwen3-snapkv-cli-tests PRIVATE
    GTest::gtest_main
)

gtest_discover_tests(qwen3-snapkv-cli-tests)

# SnapKV eviction + compaction unit tests (GTest, no model needed)
add_executable(qwen3-snapkv-eviction-tests
    unit/test_snapkv_eviction.cpp
)

target_link_libraries(qwen3-snapkv-eviction-tests PRIVATE
    GTest::gtest_main
    qwen3-kv-cache
    ggml
)

gtest_discover_tests(qwen3-snapkv-eviction-tests)

# SnapKV position tracking tests (GTest)
add_executable(qwen3-snapkv-position-tests
    unit/test_snapkv_position.cpp
)

target_link_libraries(qwen3-snapkv-position-tests PRIVATE
    GTest::gtest_main
    qwen3-kv-cache
    qwen3-core
    ggml
)

gtest_discover_tests(qwen3-snapkv-position-tests)

# TokenTrie unit tests (GTest, no model needed)
add_executable(qwen3-token-trie-tests
    unit/test_token_trie.cpp
)

target_link_libraries(qwen3-token-trie-tests PRIVATE
    GTest::gtest_main
    sampling
)

gtest_discover_tests(qwen3-token-trie-tests)

# Parallel grammar test (Non-GTest)
add_executable(qwen3-parallel-grammar-test
    integration/test_parallel_grammar.cpp
)

target_link_libraries(qwen3-parallel-grammar-test PRIVATE
    qwen3-core
    sampling
    ggml
)

# HTTP Server Integration Test (GTest)
add_executable(qwen3-http-server-test
    integration/test_http_server.cpp
)

target_link_libraries(qwen3-http-server-test PRIVATE
    inference_server
    httplib::httplib
    nlohmann_json::nlohmann_json
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

target_compile_features(qwen3-http-server-test PRIVATE cxx_std_17)
gtest_discover_tests(qwen3-http-server-test)

# Grammar regression tests (non-GTest standalone executables)
add_subdirectory(grammar)
