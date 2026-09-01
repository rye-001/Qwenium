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

    void configure_server(qinf::InferenceServer& server) {
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

        server.set_is_stop_token([this](int token_id) {
            return token_id == eos_token_;
        });

        // Speculative decoding wiring (--speculative). Only registered when a
        // test explicitly opts in (enable_speculative), mirroring
        // QweniumServerIntegration: spec_ null => neither callback is
        // registered => decode_step() never leaves the batched path above.
        // Every pre-existing test in this file configures nothing here and
        // still passes unmodified — that IS the byte-identical-when-off gate.
        if (speculative_enabled_) {
            server.set_speculative_decode(
                [this](int slot_id, int32_t /*last_token*/,
                       const std::vector<int32_t>& /*prompt_tokens*/,
                       const std::vector<int32_t>& /*generated_tokens*/) {
                    speculative_step_count_++;
                    qinf::InferenceServer::SpeculativeStepResult out;
                    // Same index convention set_batched_decode uses:
                    // slot_positions_[slot_id] == index of the most recently
                    // DELIVERED token in response_tokens_ (0 right after
                    // prefill, which always delivers index 0). "Verification"
                    // here is trivial — the mock knows its own ground-truth
                    // sequence — this exercises the ENGINE's multi-token
                    // delivery + deferred-next contract, not real speculative
                    // verification (that's sampling/speculative.h's job,
                    // covered by tests/unit/test_speculative.cpp).
                    int& pos = slot_positions_[slot_id];
                    for (int i = 0; i < speculative_draft_width_; ++i) {
                        if (pos + 1 >= (int)response_tokens_.size()) break;
                        ++pos;
                        int tok = response_tokens_[pos];
                        out.delivered_tokens.push_back(tok);
                        if (tok == eos_token_) return out;  // never a next_token past eos
                    }
                    if (pos + 1 < (int)response_tokens_.size()) {
                        ++pos;
                        out.has_next = true;
                        out.next_token = response_tokens_[pos];
                    } else {
                        out.delivered_tokens.push_back(eos_token_);
                    }
                    return out;
                });
            server.set_speculative_eligible([this](int /*slot_id*/) {
                return speculative_eligible_;
            });
        }
    }

    // Test-only opt-in, mirroring QweniumServerIntegration::enable_speculative
    // (server startup, not per-request). `draft_width` caps how many tokens
    // one speculative_decode_ call delivers before deferring — analogous to
    // --suffix-max-draft / --mtp-max-draft.
    void enable_speculative(int draft_width = 4) {
        speculative_enabled_ = true;
        speculative_draft_width_ = draft_width;
    }
    // Toggle whether the (single) active slot is speculative-eligible —
    // stands in for "no grammar active on this slot" (the real gate).
    void set_speculative_eligible(bool eligible) { speculative_eligible_ = eligible; }
    int speculative_step_count() const { return speculative_step_count_; }

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

    // Speculative decoding test wiring (see enable_speculative above).
    bool speculative_enabled_ = false;
    bool speculative_eligible_ = true;
    int speculative_draft_width_ = 4;
    std::atomic<int> speculative_step_count_{0};
};

// =============================================================================
// Speculative decoding wiring (--speculative). Gates 2/3 of the server
// speculative-decoding task: single-slot engagement produces the SAME output
// as the batched path, and a concurrent (multi-slot) run behaves correctly.
// Gate 1 (byte-identical when OFF) is every HttpServerTest below: none of
// them touch speculative decoding at all, and they all still pass unmodified
// — decode_step()'s speculative branch is dead code unless
// set_speculative_decode() was called, which only enable_speculative() does.
//
// Speculative decoding test harness (bypasses HTTP entirely)
//
// Drives qinf::InferenceServer directly against MockInferenceBackend: submit
// a request, drain its token_queue, read output_text/finish_reason off it.
// Used only by the speculative tests below, which need
// MockInferenceBackend::enable_speculative() to run BEFORE configure_server()
// wires the callbacks — exactly like QweniumServerIntegration's
// enable_speculative() must run before configure_server() in production
// (registration is a one-time, startup-only decision, not a live toggle).
// The HTTP fixture above starts its server inside SetUp(), before a test
// body gets a chance to touch mock_, so it can't express that ordering
// without restarting httplib on a fixed port mid-test — which races the
// OS's socket teardown and was flaky in practice; this sidesteps it.
// =============================================================================
struct SpeculativeHarness {
    std::unique_ptr<MockInferenceBackend> mock;
    std::unique_ptr<qinf::InferenceServer> server;
    std::thread inference_thread;

    SpeculativeHarness() : mock(std::make_unique<MockInferenceBackend>()) {}

    ~SpeculativeHarness() {
        if (server) server->stop();
        if (inference_thread.joinable()) inference_thread.join();
    }

    // Call once mock is fully configured (set_response, enable_speculative,
    // set_speculative_eligible, ...).
    void start(int max_slots = 4) {
        qinf::InferenceServer::Config config;
        config.max_slots = max_slots;
        config.max_queue_depth = 10;
        config.max_context = 64;
        config.request_timeout = std::chrono::seconds(5);
        server = std::make_unique<qinf::InferenceServer>(config);
        mock->configure_server(*server);
        inference_thread = std::thread([this]() { server->run(); });
    }

    // Build + submit a request; returns immediately (does not wait for it).
    std::shared_ptr<qinf::InferenceRequest> submit(const std::string& prompt,
                                                    int max_tokens = 50) {
        auto req = std::make_shared<qinf::InferenceRequest>();
        req->prompt = prompt;
        req->max_tokens = max_tokens;
        server->submit(req);
        return req;
    }

    // Block until a submitted request completes (mirrors the HTTP fixture's
    // non-streaming drain loop, minus the HTTP/JSON wrapping).
    static void drain(const std::shared_ptr<qinf::InferenceRequest>& req) {
        while (req->token_queue->pop_blocking() != qinf::TokenQueue::QUEUE_END) {
        }
    }

    // submit() + drain() for the common single-request case.
    std::shared_ptr<qinf::InferenceRequest> run(const std::string& prompt,
                                                int max_tokens = 50) {
        auto req = submit(prompt, max_tokens);
        drain(req);
        return req;
    }
};

// Single active slot, speculative eligible: the engine must engage
// decode_step_speculative() (speculative_step_count() > 0) and the resulting
// text must be identical to what the batched path produces for the same
// response sequence.
TEST(SpeculativeDecoding, SingleSlotMatchesBatchedOutput) {
    SpeculativeHarness h;
    h.mock->set_response({'H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!', 0});
    h.mock->enable_speculative(/*draft_width=*/4);
    h.start();

    auto req = h.run("hi");
    EXPECT_EQ(req->output_text, "Hello, world!");
    EXPECT_EQ(req->finish_reason, "stop");
    // Engaged at least once, and delivered more than one token per step at
    // least once (13 response tokens over 13+ steps would mean it never
    // actually batched anything).
    EXPECT_GT(h.mock->speculative_step_count(), 0);
    EXPECT_LT(h.mock->speculative_step_count(), 13);
}

// Same response, same request, run once with speculative off and once on (two
// separate harnesses — MockInferenceBackend has no live on/off toggle, by
// design: production doesn't either): the two completions must be
// token-identical (gate 2, greedy).
TEST(SpeculativeDecoding, OnMatchesOff) {
    SpeculativeHarness off;
    off.mock->set_response({'H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!', 0});
    off.start();
    auto off_req = off.run("hi");

    SpeculativeHarness on;
    on.mock->set_response({'H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!', 0});
    on.mock->enable_speculative(4);
    on.start();
    auto on_req = on.run("hi");

    EXPECT_EQ(off_req->output_text, on_req->output_text);
    EXPECT_EQ(off_req->finish_reason, on_req->finish_reason);
    EXPECT_EQ(off_req->completion_tokens, on_req->completion_tokens);
    EXPECT_GT(on.mock->speculative_step_count(), 0);
}

// Ineligible slot (stands in for a per-request grammar): speculative is
// enabled server-wide but must never engage on this slot — the batched path
// serves the request instead, and the output is still correct. Matches the
// CLI's own behavior (--speculative + --grammar-file disables speculation
// outright).
TEST(SpeculativeDecoding, IneligibleSlotFallsBackToBatched) {
    SpeculativeHarness h;
    h.mock->set_response({'H', 'e', 'l', 'l', 'o', '!', 0});
    h.mock->enable_speculative(4);
    h.mock->set_speculative_eligible(false);
    h.start();

    auto req = h.run("hi");
    EXPECT_EQ(req->output_text, "Hello!");
    EXPECT_EQ(h.mock->speculative_step_count(), 0);
    EXPECT_GT(h.mock->decode_count(), 0);
}

// Two concurrent requests of EQUAL response length, submitted back-to-back
// from ONE thread (so both land in the request queue together — assign_
// requests_to_slots() drains the whole queue per iteration, so this makes it
// overwhelmingly likely both are assigned in the SAME iteration rather than
// racing each other in): once both are assigned, both slots stay active until
// they complete on the same batched step, so the >1-active-slot fallback in
// decode_step() covers the whole run — the multi-slot path is exactly what
// it was before this task (gate 3). Both completions still have to be
// correct, and the batched path (not the speculative one) must be what
// served them.
TEST(SpeculativeDecoding, MultiSlotStillCorrect) {
    SpeculativeHarness h;
    h.mock->set_response({'A', 'A', 'A', 'A', 0});
    h.mock->enable_speculative(4);
    h.start();

    auto req0 = h.submit("concurrent");
    auto req1 = h.submit("concurrent");
    std::thread t0([&]() { SpeculativeHarness::drain(req0); });
    std::thread t1([&]() { SpeculativeHarness::drain(req1); });
    t0.join();
    t1.join();

    EXPECT_EQ(req0->output_text, "AAAA");
    EXPECT_EQ(req1->output_text, "AAAA");
    EXPECT_GT(h.mock->decode_count(), 0);
    EXPECT_EQ(h.mock->speculative_step_count(), 0);
}

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

        qinf::InferenceServer::Config config;
        config.max_slots = 4;
        config.max_queue_depth = 10;
        config.max_context = 64;  // small ceiling so the fail-loud test can exceed it
        config.request_timeout = std::chrono::seconds(5);

        server_ = std::make_unique<qinf::InferenceServer>(config);
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

            auto inf_req = std::make_shared<qinf::InferenceRequest>();
            inf_req->prompt = prompt;
            inf_req->max_tokens = body.value("max_tokens", 256);

            if (body.contains("stop")) {
                const auto& stop = body["stop"];
                if (stop.is_string()) {
                    inf_req->stop.push_back(stop.get<std::string>());
                } else if (stop.is_array()) {
                    for (const auto& s : stop) {
                        if (s.is_string()) inf_req->stop.push_back(s.get<std::string>());
                    }
                }
            }

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
                            if (token_id == qinf::TokenQueue::QUEUE_END) {
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
                // Mirror production: the server owns the canonical output_text,
                // token counts, and finish_reason. Drain only to wait for the end.
                while (inf_req->token_queue->pop_blocking() != qinf::TokenQueue::QUEUE_END) {
                }
                if (!inf_req->error_message.empty()) {
                    res.status = 413;
                    res.set_content(json({{"error", inf_req->error_message}}).dump(),
                                    "application/json");
                    return;
                }
                json response = {
                    {"choices", {{
                        {"text", inf_req->output_text},
                        {"finish_reason", inf_req->finish_reason}
                    }}},
                    {"usage", {
                        {"prompt_tokens", inf_req->prompt_tokens},
                        {"completion_tokens", inf_req->completion_tokens},
                        {"total_tokens", inf_req->prompt_tokens + inf_req->completion_tokens}
                    }}
                };
                res.set_content(response.dump(), "application/json");
            }
        });
    }

    httplib::Client get_client() {
        return httplib::Client("127.0.0.1", TEST_PORT);
    }

    std::unique_ptr<MockInferenceBackend> mock_;
    std::unique_ptr<qinf::InferenceServer> server_;
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
// Delegation-contract smoke tests (items 1-6)
//
// These exercise the server-owned request lifecycle that the production HTTP
// handler depends on: canonical output_text, stop-string truncation,
// finish_reason, usage counts, slot reset between requests, and the fail-loud
// oversized-prompt guard. The mock backend is deterministic by construction, so
// any cross-request variation here is a server-logic bug, not a model effect.
// =============================================================================

// Item 1 (determinism): the same request body submitted twice yields a
// byte-identical completion and finish_reason. Proves the slot is reset to a
// clean state between independent requests.
TEST_F(HttpServerTest, DeterminismSameRequestTwice) {
    mock_->set_response({'H', 'e', 'l', 'l', 'o', 0});
    auto client = get_client();
    json request = {{"prompt", "hi"}, {"max_tokens", 50}, {"stream", false}};

    auto a = client.Post("/v1/completions", request.dump(), "application/json");
    auto b = client.Post("/v1/completions", request.dump(), "application/json");

    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    ASSERT_EQ(a->status, 200);
    ASSERT_EQ(b->status, 200);
    auto ja = json::parse(a->body);
    auto jb = json::parse(b->body);
    EXPECT_EQ(ja["choices"][0]["text"], jb["choices"][0]["text"]);
    EXPECT_EQ(ja["choices"][0]["text"], "Hello");
    EXPECT_EQ(ja["choices"][0]["finish_reason"], jb["choices"][0]["finish_reason"]);
}

// Item 4 (slot isolation): request B must not inherit request A's per-slot
// state. The mock advances a per-slot decode cursor that is only reset by
// clear_slot(); if isolation were broken, B would resume mid-sequence (or hit
// EOS immediately) instead of producing its own full response.
TEST_F(HttpServerTest, SlotIsolationBetweenRequests) {
    auto client = get_client();

    mock_->set_response({'A', 'A', 'A', 0});
    auto a = client.Post("/v1/completions",
                         json({{"prompt", "first"}, {"max_tokens", 50}}).dump(),
                         "application/json");
    ASSERT_TRUE(a);
    EXPECT_EQ(json::parse(a->body)["choices"][0]["text"], "AAA");

    mock_->set_response({'B', 'B', 'B', 'B', 0});
    auto b = client.Post("/v1/completions",
                         json({{"prompt", "second"}, {"max_tokens", 50}}).dump(),
                         "application/json");
    ASSERT_TRUE(b);
    // Full fresh response, not truncated by A's stale cursor.
    EXPECT_EQ(json::parse(b->body)["choices"][0]["text"], "BBBB");
}

// Item 2 (stop sequences): output ends *before* the stop string, not after,
// and the stop string itself is excluded.
TEST_F(HttpServerTest, StopSequenceTruncatesOutput) {
    mock_->set_response({'H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!', 0});
    auto client = get_client();
    json request = {
        {"prompt", "hi"},
        {"max_tokens", 100},
        {"stop", json::array({"world"})}
    };

    auto res = client.Post("/v1/completions", request.dump(), "application/json");
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200);
    auto body = json::parse(res->body);
    EXPECT_EQ(body["choices"][0]["text"], "Hello, ");
    EXPECT_EQ(body["choices"][0]["finish_reason"], "stop");
}

// Item 3 (max_tokens): generation is bounded and the truncation is reported as
// finish_reason "length" (not a generic "stop").
TEST_F(HttpServerTest, MaxTokensReportsLengthFinishReason) {
    mock_->set_response({'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 0});
    auto client = get_client();
    json request = {{"prompt", "hi"}, {"max_tokens", 3}, {"stream", false}};

    auto res = client.Post("/v1/completions", request.dump(), "application/json");
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200);
    auto body = json::parse(res->body);
    EXPECT_LE(std::string(body["choices"][0]["text"]).length(), 3u);
    EXPECT_EQ(body["choices"][0]["finish_reason"], "length");
    EXPECT_EQ(body["usage"]["completion_tokens"], 3);
}

// Item 6 (usage counts): prompt/completion/total token counts are present and
// internally consistent.
TEST_F(HttpServerTest, UsageCountsReported) {
    mock_->set_response({'x', 'y', 'z', 0});
    auto client = get_client();
    json request = {{"prompt", "abcd"}, {"max_tokens", 50}, {"stream", false}};

    auto res = client.Post("/v1/completions", request.dump(), "application/json");
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200);
    auto body = json::parse(res->body);
    ASSERT_TRUE(body.contains("usage"));
    int prompt_tokens     = body["usage"]["prompt_tokens"];
    int completion_tokens = body["usage"]["completion_tokens"];
    int total_tokens      = body["usage"]["total_tokens"];
    EXPECT_EQ(prompt_tokens, 4);          // "abcd" -> 4 char tokens
    EXPECT_EQ(completion_tokens, 3);      // "xyz"
    EXPECT_EQ(total_tokens, prompt_tokens + completion_tokens);
}

// Item 5 (fail-loud): an oversized prompt is rejected with a named error
// (slot, expected ceiling, actual count) instead of overflowing the KV cache.
TEST_F(HttpServerTest, OversizedPromptFailsLoud) {
    auto client = get_client();
    std::string huge(200, 'a');  // 200 char-tokens > config.max_context (64)
    json request = {{"prompt", huge}, {"max_tokens", 10}, {"stream", false}};

    auto res = client.Post("/v1/completions", request.dump(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 413);
    auto body = json::parse(res->body);
    ASSERT_TRUE(body.contains("error"));
    std::string err = body["error"];
    EXPECT_NE(err.find("prompt too large"), std::string::npos);
    EXPECT_NE(err.find("64"), std::string::npos);   // expected ceiling
    EXPECT_NE(err.find("200"), std::string::npos);  // actual count
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}