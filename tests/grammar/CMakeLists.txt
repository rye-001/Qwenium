# tests/grammar/CMakeLists.txt
# Grammar regression tests (non-GTest, standalone executables)
# These test the GrammarVocab API from src/sampling/grammar_vocab.h

add_executable(grammar-test-dynamic-strings test_dynamic_strings.cpp)
target_link_libraries(grammar-test-dynamic-strings PRIVATE sampling)

add_executable(grammar-test-js-subset test_js_subset.cpp)
target_link_libraries(grammar-test-js-subset PRIVATE sampling)

add_executable(grammar-test-multitoken test_multitoken.cpp)
target_link_libraries(grammar-test-multitoken PRIVATE sampling)

add_executable(grammar-test-sampling-integration test_sampling_integration.cpp)
target_link_libraries(grammar-test-sampling-integration PRIVATE sampling)

add_executable(grammar-test-resolve-once test_resolve_once.cpp)
target_link_libraries(grammar-test-resolve-once PRIVATE sampling)
