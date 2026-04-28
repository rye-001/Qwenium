# Qwen HTTP Inference Server

Minimal HTTP server for serving Qwen inference with OpenAI-compatible API.

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  HTTP Handlers  │────────▶│   RequestQueue   │
│  (cpp-httplib)  │         │  (thread-safe)   │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         │ blocks on                 │ pops requests
         │ TokenQueue                ▼
         │                  ┌──────────────────┐
         │◀─────────────────│ Inference Thread │
         │  pushes tokens   │  (owns model)    │
         └──────────────────└──────────────────┘
```

## Files

```
src/server/
├── inference_server.h   # Core server: queues, slots, InferenceServer class
└── http_server.cpp      # HTTP routes, main(), model integration

tests/integration/
└── test_http_server.cpp # Integration tests with mock backend
```

## Integration Steps

### 1. Add Dependencies

**cpp-httplib** (header-only):
```bash
# Clone or download httplib.h
wget https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h
```

**nlohmann/json** (header-only):
```bash
# Clone or download json.hpp
wget https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
mv json.hpp nlohmann/json.hpp
```

Or use CMake FetchContent (see CMakeLists_server.txt).

### 2. Wire Up Your Forward Pass

Edit `http_server.cpp` `QwenServerIntegration` class to match your API:

```cpp
int run_prefill(int slot_id, const std::vector<int32_t>& tokens, int start_pos) {
    // Your existing prefill code
    // Need to add slot_id parameter to build_graph if using per-slot KV cache
    ggml_cgraph* gf = forward_pass_->build_graph(tokens, start_pos, slot_id);
    // ... execute, sample first token ...
    return first_token;
}

std::vector<int> run_batched_decode(const std::vector<int32_t>& tokens,
                                    const std::vector<int>& slot_ids) {
    // Your existing batched decode
    ggml_cgraph* gf = forward_pass_->build_decoding_graph(tokens, slot_ids);
    // ... execute, sample per-slot ...
    return next_tokens;
}
```

### 3. Build

Add to your CMakeLists.txt:
```cmake
add_executable(http_server src/server/http_server.cpp)
target_link_libraries(http_server PRIVATE
    your_qwen_lib
    httplib::httplib  # or just include the header
    nlohmann_json::nlohmann_json
    Threads::Threads
)
```

## Usage

### Start Server

```bash
./http_server --port 8080 --model ./qwen2.5-coder-14b-instruct-q4_0.gguf
```

### API Endpoints

**Health Check**
```bash
curl http://localhost:8080/health
```
```json
{"status": "ok", "active_slots": 2, "queue_depth": 0, "tokens_generated": 1234}
```

**Non-Streaming Completion**
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a hello world in Python:", "max_tokens": 100}'
```
```json
{
  "choices": [{"text": "print(\"Hello, World!\")", "finish_reason": "stop"}],
  "usage": {"completion_tokens": 15}
}
```

**Streaming Completion (SSE)**
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Count to 5:", "max_tokens": 50, "stream": true}'
```
```
data: {"choices": [{"text": "1"}]}

data: {"choices": [{"text": ","}]}

data: {"choices": [{"text": " "}]}

data: {"choices": [{"text": "2"}]}

...

data: [DONE]
```

**List Models**
```bash
curl http://localhost:8080/v1/models
```

## Configuration

In `http_server.cpp` main():

```cpp
qwenium::InferenceServer::Config config;
config.max_slots = 10;           // Max concurrent generations
config.max_queue_depth = 100;    // Reject requests if queue exceeds this
config.request_timeout = 120s;   // Kill requests after this duration
config.system_prompt_len = 0;    // If using prefix caching
```

## Key Design Decisions

1. **Single inference thread**: All model operations happen on one thread. HTTP threads block on per-request `TokenQueue`. This avoids ggml synchronization complexity.

2. **Callback-based integration**: `InferenceServer` doesn't depend on your model types. Wire up via `set_prefill()`, `set_batched_decode()`, etc.

3. **Static batching**: All active slots decode together. Simpler than continuous batching, sufficient for 10 concurrent users.

4. **Request rejection**: When queue is full, return 503. Client can retry. Better than unbounded queuing.

## Running Tests

```bash
# Build with tests enabled
cmake -B build -DBUILD_TESTING=ON
cmake --build build

# Run tests
cd build && ctest --output-on-failure
```

Tests use a mock backend - no GPU/model required.

## TODO for Production

- [ ] Request ID tracking for logging
- [ ] Prometheus metrics endpoint
- [ ] Graceful drain on shutdown
- [ ] Rate limiting per client
- [ ] Authentication (API key header)
- [ ] CORS headers for browser access