// test_http_server.cpp
// Integration tests for the HTTP inference server
//
// These tests use a mock inference backend to verify:
// 1. HTTP endpoint correctness
// 2. Request/response flow
// 3. Streaming (SSE) behavior
// 4. Concurrent request handling
// 5. Error handling and edge cases

#include "gtest/gtest.h"
#include "inference_server.h"

// Include httplib in implementation mode for test client
#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include "httplib.h"
#include "nlohmann/json.hpp"

#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <sstream>

using json = nlohmann::json;

// =============================================================================
// Mock Inference Backend
// =============================================================================
class MockInferenceBackend {
public:
    MockInferenceBackend() {
        // Simple vocabulary for testing
        vocab_ = {"<eos>", "Hello", ",", " ", "world", "!", "How", "are", "you", "?"};
        eos_token_ = 0;
    }

    void configure_server(qwenium::InferenceServer& server) {
        server.set_tokenize([this](const std::string& text) {
            return tokenize(text);
        });

        server.set_detokenize([this](int token_id) {
            return detokenize(token_id);
        });

        server.set_prefill([this](int slot_id, const std::vector<int32_t>& tokens, int start_pos) {
            (void)slot_id; (void)start_pos;
            prefill_count_++;
            // Return first token of response sequence
            return response_tokens_.empty() ? eos_token_ : response_tokens_[0];
        });

        server.set_batched_decode([this](const std::vector<int32_t>& tokens,
                                         const std::vector<int>& slot_ids) {
            decode_count_++;
            std::vector<int> next_tokens;
            for (size_t i = 0; i < slot_ids.size(); ++i) {
                (void)tokens[i];
                int slot_id = slot_ids[i];
                int pos = slot_positions_[slot_id]++;
                
                if (pos + 1 < (int)response_tokens_.size()) {
                    next_tokens.push_back(response_tokens_[pos + 1]);
                } else {
                    next_tokens.push_back(eos_token_);
                }
            }
            return next_tokens;
        });

        server.set_clear_slot([this](int slot_id) {
            slot_positions_.erase(slot_id);
        });

        server.set_clone_slot([](int from_slot, int to_slot) {
            (void)from_slot; (void)to_slot;
        });

        server.set_get_eos_token([this]() {
            return eos_token_;
        });
    }

    // Set the response that will be generated
    void set_response(const std::vector<int>& tokens) {
        response_tokens_ = tokens;
    }

    // Set response as text (tokenized)
    void set_response_text(const std::string& text) {
        response_tokens_ = tokenize(text);
    }

    int prefill_count() const { return prefill_count_; }
    int decode_count() const { return decode_count_; }
    void reset_counts() { prefill_count_ = 0; decode_count_ = 0; }

private:
    std::vector<int32_t> tokenize(const std::string& text) {
        // Simple character-level tokenization for testing
        std::vector<int32_t> tokens;
        for (char c : text) {
            tokens.push_back(static_cast<int32_t>(c));
        }
        return tokens;
    }

    std::string detokenize(int token_id) {
        if (token_id == eos_token_) return "";
        // Return the character
        return std::string(1, static_cast<char>(token_id));
    }

    std::vector<std::string> vocab_;
    std::vector<int> response_tokens_;
    std::map<int, int> slot_positions_;
    int eos_token_;
    std::atomic<int> prefill_count_{0};
    std::atomic<int> decode_count_{0};
};

// =============================================================================
// Test Fixture
// =============================================================================
class HttpServerTest : public ::testing::Test {
protected:
    static constexpr int TEST_PORT = 18080;
    static constexpr auto STARTUP_DELAY = std::chrono::milliseconds(100);
    static constexpr auto SHUTDOWN_DELAY = std::chrono::milliseconds(50);

    void SetUp() override {
        mock_ = std::make_unique<MockInferenceBackend>();
        
        // Set default response: "Hello!"
        mock_->set_response({'H', 'e', 'l', 'l', 'o', '!', 0});

        qwenium::InferenceServer::Config config;
        config.max_slots = 4;
        config.max_queue_depth = 10;
        config.request_timeout = std::chrono::seconds(5);

        server_ = std::make_unique<qwenium::InferenceServer>(config);
        mock_->configure_server(*server_);

        // Start inference thread
        inference_thread_ = std::thread([this]() {
            server_->run();
        });

        // Start HTTP server thread
        http_server_ = std::make_unique<httplib::Server>();
        setup_test_routes();
        
        http_thread_ = std::thread([this]() {
            http_server_->listen("127.0.0.1", TEST_PORT);
        });

        // Wait for servers to start
        std::this_thread::sleep_for(STARTUP_DELAY);
    }

    void TearDown() override {
        // Stop servers
        server_->stop();
        http_server_->stop();

        if (inference_thread_.joinable()) {
            inference_thread_.join();
        }
        if (http_thread_.joinable()) {
            http_thread_.join();
        }

        std::this_thread::sleep_for(SHUTDOWN_DELAY);
    }

    void setup_test_routes() {
        // Health endpoint
        http_server_->Get("/health", [this](const httplib::Request&, httplib::Response& res) {
            json response = {
                {"status", "ok"},
                {"active_slots", server_->stats().active_slots.load()},
                {"queue_depth", server_->stats().queue_depth.load()}
            };
            res.set_content(response.dump(), "application/json");
        });

        // Completions endpoint
        http_server_->Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
            json body;
            try {
                body = json::parse(req.body);
            } catch (...) {
                res.status = 400;
                res.set_content(R"({"error": "Invalid JSON"})", "application/json");
                return;
            }

            std::string prompt = body.value("prompt", "");
            if (prompt.empty()) {
                res.status = 400;
                res.set_content(R"({"error": "Missing prompt"})", "application/json");
                return;
            }

            auto inf_req = std::make_shared<qwenium::InferenceRequest>();
            inf_req->prompt = prompt;
            inf_req->max_tokens = body.value("max_tokens", 256);

            bool stream = body.value("stream", false);

            if (!server_->submit(inf_req)) {
                res.status = 503;
                res.set_content(R"({"error": "Queue full"})", "application/json");
                return;
            }

            if (stream) {
                res.set_header("Content-Type", "text/event-stream");
                res.set_header("Cache-Control", "no-cache");

                res.set_chunked_content_provider(
                    "text/event-stream",
                    [inf_req, this](size_t, httplib::DataSink& sink) {
                        while (true) {
                            int token_id = inf_req->token_queue->pop_blocking();
                            if (token_id == qwenium::TokenQueue::QUEUE_END) {
                                sink.write("data: [DONE]\n\n", 14);
                                sink.done();
                                return false;
                            }
                            std::string token_text = server_->decode_token(token_id);
                            json chunk = {{"choices", {{{"text", token_text}}}}};
                            std::string sse = "data: " + chunk.dump() + "\n\n";
                            sink.write(sse.c_str(), sse.size());
                        }
                        return true;
                    },
                    [inf_req](bool success) {
                        if (!success) inf_req->cancelled = true;
                    }
                );
            } else {
                std::string full_text;
                while (true) {
                    int token_id = inf_req->token_queue->pop_blocking();
                    if (token_id == qwenium::TokenQueue::QUEUE_END) break;
                    full_text += server_->decode_token(token_id);
                }
                json response = {{"choices", {{{"text", full_text}}}}};
                res.set_content(response.dump(), "application/json");
            }
        });
    }

    httplib::Client get_client() {
        return httplib::Client("127.0.0.1", TEST_PORT);
    }

    std::unique_ptr<MockInferenceBackend> mock_;
    std::unique_ptr<qwenium::InferenceServer> server_;
    std::unique_ptr<httplib::Server> http_server_;
    std::thread inference_thread_;
    std::thread http_thread_;
};

// =============================================================================
// Tests
// =============================================================================

TEST_F(HttpServerTest, HealthEndpoint) {
    auto client = get_client();
    auto res = client.Get("/health");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);

    auto body = json::parse(res->body);
    EXPECT_EQ(body["status"], "ok");
}

TEST_F(HttpServerTest, NonStreamingCompletion) {
    auto client = get_client();
    
    json request = {
        {"prompt", "Say hello"},
        {"max_tokens", 50},
        {"stream", false}
    };

    auto res = client.Post("/v1/completions", request.dump(), "application/json");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);

    auto body = json::parse(res->body);
    EXPECT_TRUE(body.contains("choices"));
    EXPECT_FALSE(body["choices"].empty());
    
    std::string text = body["choices"][0]["text"];
    EXPECT_EQ(text, "Hello!");
}

TEST_F(HttpServerTest, StreamingCompletion) {
    auto client = get_client();
    
    json request = {
        {"prompt", "Say hello"},
        {"max_tokens", 50},
        {"stream", true}
    };

    std::string accumulated;
    std::vector<std::string> events;
    
    httplib::Request req;
    req.method = "POST";
    req.path = "/v1/completions";
    req.body = request.dump();
    req.set_header("Content-Type", "application/json");
    
    req.content_receiver = [&](const char* data, size_t len, uint64_t /*offset*/, uint64_t /*total_len*/) {
        std::string chunk(data, len);
        
        // Parse SSE events
        std::istringstream stream(chunk);
        std::string line;
        while (std::getline(stream, line)) {
            if (line.substr(0, 6) == "data: ") {
                events.push_back(line.substr(6));
                if (line != "data: [DONE]") {
                    auto event_json = json::parse(line.substr(6));
                    if (event_json.contains("choices")) {
                        accumulated += event_json["choices"][0]["text"].get<std::string>();
                    }
                }
            }
        }
        return true;
    };

    auto res = client.send(req);

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(accumulated, "Hello!");
    EXPECT_FALSE(events.empty());
    EXPECT_EQ(events.back(), "[DONE]");
}

TEST_F(HttpServerTest, MissingPromptError) {
    auto client = get_client();
    
    json request = {
        {"max_tokens", 50}
    };

    auto res = client.Post("/v1/completions", request.dump(), "application/json");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(HttpServerTest, InvalidJsonError) {
    auto client = get_client();
    
    auto res = client.Post("/v1/completions", "not valid json", "application/json");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(HttpServerTest, ConcurrentRequests) {
    const int NUM_REQUESTS = 4;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};

    for (int i = 0; i < NUM_REQUESTS; ++i) {
        threads.emplace_back([this, &success_count, &error_count]() {
            auto client = get_client();
            json request = {
                {"prompt", "Test prompt"},
                {"max_tokens", 10},
                {"stream", false}
            };

            auto res = client.Post("/v1/completions", request.dump(), "application/json");
            
            if (res && res->status == 200) {
                success_count++;
            } else {
                error_count++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count, NUM_REQUESTS);
    EXPECT_EQ(error_count, 0);
}

TEST_F(HttpServerTest, StatsTracking) {
    auto client = get_client();
    
    // Make a few requests
    for (int i = 0; i < 3; ++i) {
        json request = {
            {"prompt", "Test"},
            {"max_tokens", 5},
            {"stream", false}
        };
        auto res = client.Post("/v1/completions", request.dump(), "application/json");
        ASSERT_TRUE(res);
    }

    // Check stats via health endpoint
    auto health = client.Get("/health");
    ASSERT_TRUE(health);
    
    auto body = json::parse(health->body);
    EXPECT_EQ(body["status"], "ok");
}

TEST_F(HttpServerTest, MaxTokensRespected) {
    // Set a longer response
    mock_->set_response({'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 0});
    
    auto client = get_client();
    
    json request = {
        {"prompt", "Test"},
        {"max_tokens", 3},
        {"stream", false}
    };

    auto res = client.Post("/v1/completions", request.dump(), "application/json");

    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);

    auto body = json::parse(res->body);
    std::string text = body["choices"][0]["text"];
    
    // Should be limited to 3 tokens (including the first one from prefill)
    EXPECT_LE(text.length(), 3u);
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}