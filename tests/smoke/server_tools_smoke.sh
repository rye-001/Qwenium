#!/usr/bin/env bash
# server_tools_smoke.sh — OpenAI tool calling on /v1/chat/completions, end to
# end on one real model (docs/plan-tool-calling.md gate G5). Run it once per
# family, one model resident at a time:
#
#   MODEL=models/Qwen3-0.6B-Q8_0.gguf       tests/smoke/server_tools_smoke.sh   # hermes-json
#   MODEL=models/Qwen3.5-0.8B-BF16.gguf     tests/smoke/server_tools_smoke.sh   # qwen-xml
#   MODEL=models/gemma-4-12B-it-Q8_0.gguf   tests/smoke/server_tools_smoke.sh   # gemma4
#
# Temperature 0. What it checks, each by name:
#   format       the server picked a tool format for this model (startup log)
#   call         an unambiguous two-tool question → 200, finish "tool_calls",
#                exactly get_weather({"city": "Paris"}), an OpenAI call id
#   answer       the tool's result sent back → a plain answer that uses it ("17")
#   stopped      the reply ends at the call: for Gemma 4 this is the
#                <|tool_response> stop — a model that went on to invent the
#                tool's answer would be text after the call, a 502
#   none         tool_choice "none" (tools undeclared to the model) → no call;
#                an answer that ends on its own and is about the question
#   required     tool_choice "required" on a prompt with no reason to call → a call
#   named        tool_choice naming get_time on a weather question → get_time
#   think        thinking on + "required" → reasoning_content and the call
#   refusals     stream / conversation_id / a bad tool name → 400, named
#   plain        a request without tools still answers as before
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SERVER="${SERVER:-build-metal/bin/qwenium-server}"
MODEL="${MODEL:-models/Qwen3-0.6B-Q8_0.gguf}"
PORT="${PORT:-18095}"
CTX="${CTX:-4096}"

for f in "$SERVER" "$MODEL"; do
  [[ -e "$f" ]] || { echo "FAIL: missing '$f'"; exit 1; }
done

WORK="$(mktemp -d /tmp/qinf_srv_tools_smoke.XXXXXX)"
SRV_LOG="$WORK/server.log"
SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT

echo "Starting server: $SERVER -m $MODEL -c $CTX -s 1 -p $PORT"
"$SERVER" -m "$MODEL" -c "$CTX" -s 1 -p "$PORT" >"$SRV_LOG" 2>&1 &
SERVER_PID=$!

echo -n "Waiting for server to come up"
for _ in $(seq 1 300); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo " up."; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo " FAIL: server died"; sed -n '1,40p' "$SRV_LOG"; exit 1; fi
  echo -n "."; sleep 1
done

URL="http://127.0.0.1:$PORT/v1/chat/completions"
pass=0
check() { if eval "$2"; then echo "[$1] -> PASS"; else echo "[$1] -> FAIL"; pass=1; fi; }

FORMAT="$(grep -o 'Tool calling: [a-z0-9-]*' "$SRV_LOG" | head -1 | sed 's/Tool calling: //')"
echo "  format -> '$FORMAT'"
check "format" '[[ "$FORMAT" == "hermes-json" || "$FORMAT" == "qwen-xml" || "$FORMAT" == "gemma4" ]]'

# The request bodies, built in Python so the JSON is exact. $1 = the case.
body() { python3 - "$1" <<'PY'
import json, sys
case = sys.argv[1]
tools = [
  {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
   "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                  "required": ["city"]}}},
  {"type": "function", "function": {"name": "get_time", "description": "Current local time in an IANA time zone.",
   "parameters": {"type": "object", "properties": {"tz": {"type": "string", "description": "IANA zone, e.g. Europe/Paris"}},
                  "required": ["tz"]}}},
]
ask = [{"role": "user", "content": "What is the weather in Paris right now?"}]
b = {"messages": ask, "tools": tools, "temperature": 0.0, "max_tokens": 200, "enable_thinking": False}
if case == "answer":
    b["messages"] = ask + [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "17 C, light rain"}]
elif case == "none":
    b["tool_choice"] = "none"
    b["max_tokens"] = 400   # a toolless answer explains itself (Qwen3.5-0.8B: 246 tokens)
elif case == "required":
    b["messages"] = [{"role": "user", "content": "Tell me a short joke."}]
    b["tool_choice"] = "required"
elif case == "named":
    b["tool_choice"] = {"type": "function", "function": {"name": "get_time"}}
elif case == "think":
    b["tool_choice"] = "required"
    b["enable_thinking"] = True
    b["max_tokens"] = 1500
elif case == "stream":
    b["stream"] = True
elif case == "conversation":
    b["conversation_id"] = "new"
elif case == "badname":
    b["tools"] = [{"type": "function", "function": {"name": "get weather"}}]
elif case == "plain":
    b = {"messages": [{"role": "user", "content": "Name one ocean. Answer in one word."}],
         "temperature": 0.0, "max_tokens": 40, "enable_thinking": False}
print(json.dumps(b))
PY
}

# POST a case; print "<http code> <json>" on one line.
post() { curl -sS -o "$WORK/$1.json" -w '%{http_code}' "$URL" -H 'Content-Type: application/json' --data "$(body "$1")"; }
# A field of the saved reply, by a Python expression over d (the JSON).
get() { python3 -c "import json,sys; d=json.load(open('$WORK/$1.json')); print($2)"; }

CODE="$(post call)"
CALL="$(get call 'json.dumps([(t["function"]["name"], json.loads(t["function"]["arguments"])) for t in d["choices"][0]["message"].get("tool_calls", [])])' 2>/dev/null || echo '?')"
FIN="$(get call 'd["choices"][0]["finish_reason"]' 2>/dev/null || echo '?')"
ID="$(get call 'd["choices"][0]["message"]["tool_calls"][0]["id"]' 2>/dev/null || echo '?')"
TOKS="$(get call 'd["usage"]["completion_tokens"]' 2>/dev/null || echo '?')"
echo "  call -> HTTP $CODE finish=$FIN calls=$CALL id=$ID tokens=$TOKS"
check "call" '[[ "$CODE" == "200" && "$FIN" == "tool_calls" && "$CALL" == "[[\"get_weather\", {\"city\": \"Paris\"}]]" && "$ID" =~ ^call_[0-9a-f]{24}$ ]]'
check "stopped" '[[ "$CODE" == "200" && "$TOKS" =~ ^[0-9]+$ && "$TOKS" -lt 80 ]]'

CODE="$(post answer)"
ANS="$(get answer 'd["choices"][0]["message"]["content"]' 2>/dev/null || echo '?')"
FIN="$(get answer 'd["choices"][0]["finish_reason"]' 2>/dev/null || echo '?')"
echo "  answer -> HTTP $CODE finish=$FIN content='${ANS:0:120}'"
check "answer" '[[ "$CODE" == "200" && "$FIN" == "stop" && "$ANS" == *17* ]]'

CODE="$(post none)"
NCALLS="$(get none 'len(d["choices"][0]["message"].get("tool_calls", []))' 2>/dev/null || echo '?')"
NTEXT="$(get none 'd["choices"][0]["message"]["content"] or ""' 2>/dev/null || echo '')"
NFIN="$(get none 'd["choices"][0]["finish_reason"]' 2>/dev/null || echo '?')"
echo "  none -> HTTP $CODE finish=$NFIN calls=$NCALLS content='${NTEXT:0:100}'"
# A real answer to the question, not just any text: it ends on its own and
# talks about Paris or the weather (a model that degenerates under the ban
# fills max_tokens with junk).
check "none" '[[ "$CODE" == "200" && "$NCALLS" == "0" && "$NFIN" == "stop" ]] && printf "%s" "$NTEXT" | grep -qiE "paris|weather"'

CODE="$(post required)"
RCALL="$(get required 'd["choices"][0]["message"]["tool_calls"][0]["function"]["name"]' 2>/dev/null || cat "$WORK/required.json")"
echo "  required -> HTTP $CODE call=$RCALL"
check "required" '[[ "$CODE" == "200" && ( "$RCALL" == "get_weather" || "$RCALL" == "get_time" ) ]]'

CODE="$(post named)"
NAMED="$(get named 'd["choices"][0]["message"]["tool_calls"][0]["function"]["name"]' 2>/dev/null || cat "$WORK/named.json")"
echo "  named -> HTTP $CODE call=$NAMED"
check "named" '[[ "$CODE" == "200" && "$NAMED" == "get_time" ]]'

CODE="$(post think)"
TCALL="$(get think 'd["choices"][0]["message"]["tool_calls"][0]["function"]["name"]' 2>/dev/null || cat "$WORK/think.json")"
TREAS="$(get think 'len(d["choices"][0]["message"].get("reasoning_content") or "")' 2>/dev/null || echo 0)"
echo "  think -> HTTP $CODE call=$TCALL reasoning=${TREAS} chars"
check "think" '[[ "$CODE" == "200" && "$TCALL" == "get_weather" && "$TREAS" =~ ^[0-9]+$ && "$TREAS" -gt 0 ]]'

R1="$(post stream)"; R2="$(post conversation)"; R3="$(post badname)"
echo "  refusals -> stream $R1 '$(get stream 'd["error"]' 2>/dev/null | cut -c1-60)', conversation $R2, badname $R3"
check "refusals" '[[ "$R1" == "400" && "$R2" == "400" && "$R3" == "400" ]] && grep -q "stream" "$WORK/stream.json" && grep -q "conversation_id" "$WORK/conversation.json" && grep -q "function.name" "$WORK/badname.json"'

CODE="$(post plain)"
PLAIN="$(get plain 'd["choices"][0]["message"]["content"]' 2>/dev/null || echo '')"
echo "  plain -> HTTP $CODE content='${PLAIN:0:60}'"
check "plain" '[[ "$CODE" == "200" && -n "$(printf "%s" "$PLAIN" | tr -d "[:space:]")" ]]'
check "server-alive" 'kill -0 "$SERVER_PID" 2>/dev/null'

[[ "$pass" -eq 0 ]] || { echo "================ SERVER TOOLS SMOKE FAIL ($MODEL) ================"; exit 1; }
echo "================ SERVER TOOLS SMOKE PASS ($MODEL) ================"
