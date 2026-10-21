#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# serve_tools_smoke.sh — end-to-end smoke test for OpenAI tool-calling in
# `pantheon serve`. Runs on the host CPU path (no K80). Exercises:
#   1. non-stream tool call    -> message.tool_calls + finish_reason tool_calls
#   2. multi-step tool loop     -> feed the tool result back, get a final answer
#   3. streaming tool call      -> OpenAI tool-call delta sequence + [DONE]
#
# Usage:
#   tests/serve_tools_smoke.sh [MODEL_GGUF] [PORT]
# With no MODEL_GGUF, uses $PANTHEON_TEST_MODEL, else the first .gguf in models/.
# Default port 8082.
#
# Requires: a built ./pantheon, curl, python3.
set -u

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Model: first argument, else $PANTHEON_TEST_MODEL, else the first .gguf in models/.
MODEL="${1:-${PANTHEON_TEST_MODEL:-$(ls "$REPO_DIR"/models/*.gguf 2>/dev/null | head -1)}}"
PORT="${2:-8082}"
HOST="127.0.0.1"
BASE="http://${HOST}:${PORT}"
BIN="${REPO_DIR}/pantheon"

fail() { echo "FAIL: $*" >&2; cleanup; exit 1; }
pass() { echo "PASS: $*"; }

SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

[ -x "$BIN" ] || fail "pantheon binary not built (run: make USE_BLAS=1 pantheon)"
[ -f "$MODEL" ] || fail "model not found: $MODEL"

# A single-function tool the model can obviously use.
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name, e.g. Paris"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}}]'

echo "Starting server: $BIN serve --model $MODEL --port $PORT"
"$BIN" serve --model "$MODEL" --host "$HOST" --port "$PORT" >/tmp/serve_tools_smoke.log 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
    grep -q "listening on" /tmp/serve_tools_smoke.log && break
    kill -0 "$SERVER_PID" 2>/dev/null || fail "server exited early; see /tmp/serve_tools_smoke.log"
    sleep 1
done
grep -q "listening on" /tmp/serve_tools_smoke.log || fail "server did not come up in time"
pass "server is listening"

# ---------------------------------------------------------------------------
# 1. Non-streaming tool call
# ---------------------------------------------------------------------------
REQ1="$(python3 -c '
import json,sys
tools=json.loads(sys.argv[1])
print(json.dumps({
  "messages":[
    {"role":"system","content":"You are a helpful assistant. Use tools when appropriate."},
    {"role":"user","content":"What is the weather in Paris in celsius? Use the get_weather tool."}
  ],
  "tools":tools,"tool_choice":"auto","temperature":0,"max_tokens":512
}))' "$TOOLS")"

RESP1="$(curl -s "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ1")"
echo "--- non-stream response ---"
echo "$RESP1" | python3 -m json.tool 2>/dev/null || echo "$RESP1"

echo "$RESP1" | python3 -c '
import sys,json
d=json.load(sys.stdin)
ch=d["choices"][0]
msg=ch["message"]
tc=msg.get("tool_calls")
assert tc, "no tool_calls in message: %r"%msg
call=tc[0]
assert call["type"]=="function"
assert call["id"].startswith("call_"), call["id"]
fn=call["function"]
assert fn["name"]=="get_weather", fn["name"]
args=json.loads(fn["arguments"])   # arguments MUST be a JSON string
assert "location" in args, args
assert "paris" in args["location"].lower(), args
assert ch["finish_reason"]=="tool_calls", ch["finish_reason"]
print("  tool_call id:", call["id"])
print("  function    :", fn["name"])
print("  arguments   :", fn["arguments"])
print("  finish      :", ch["finish_reason"])
' || fail "non-stream tool call not emitted correctly"
pass "non-stream tool call (finish_reason=tool_calls)"

# Grab the emitted call to feed back.
CALL_JSON="$(echo "$RESP1" | python3 -c 'import sys,json; print(json.dumps(json.load(sys.stdin)["choices"][0]["message"]["tool_calls"][0]))')"

# ---------------------------------------------------------------------------
# 2. Multi-step: feed the tool result back as role:"tool"
# ---------------------------------------------------------------------------
REQ2="$(python3 -c '
import json,sys
tools=json.loads(sys.argv[1]); call=json.loads(sys.argv[2])
print(json.dumps({
  "messages":[
    {"role":"system","content":"You are a helpful assistant. Use tools when appropriate."},
    {"role":"user","content":"What is the weather in Paris in celsius? Use the get_weather tool."},
    {"role":"assistant","content":"","tool_calls":[call]},
    {"role":"tool","tool_call_id":call["id"],"content":"{\"temperature\":\"18\",\"unit\":\"celsius\",\"conditions\":\"partly cloudy\"}"}
  ],
  "tools":tools,"temperature":0,"max_tokens":512
}))' "$TOOLS" "$CALL_JSON")"

RESP2="$(curl -s "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ2")"
echo "--- multi-step (post tool-result) response ---"
echo "$RESP2" | python3 -m json.tool 2>/dev/null || echo "$RESP2"

echo "$RESP2" | python3 -c '
import sys,json
d=json.load(sys.stdin)
msg=d["choices"][0]["message"]
content=msg.get("content") or ""
assert content.strip(), "empty final content: %r"%msg
assert "18" in content or "cloud" in content.lower(), "answer should reflect tool result: %r"%content
print("  final answer:", repr(content[:160]))
' || fail "multi-step tool-result continuation failed"
pass "multi-step tool loop (tool result -> final answer)"

# ---------------------------------------------------------------------------
# 3. Streaming tool call
# ---------------------------------------------------------------------------
REQ3="$(python3 -c '
import json,sys
tools=json.loads(sys.argv[1])
print(json.dumps({
  "messages":[
    {"role":"user","content":"What is the weather in Tokyo? Use the get_weather tool."}
  ],
  "tools":tools,"temperature":0,"max_tokens":512,"stream":True
}))' "$TOOLS")"

STREAM="$(curl -sN "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ3")"
echo "--- streaming tool-call chunks ---"
echo "$STREAM" | grep -F '"tool_calls"'

echo "$STREAM" | grep -q '"delta":{"role":"assistant"}'  || fail "no opening role delta"
echo "$STREAM" | grep -q '"tool_calls":\[{"index":0,"id":"call_' || fail "no opening tool_call delta (index/id/type/name)"
echo "$STREAM" | grep -q '"function":{"name":"get_weather","arguments":""}' || fail "opening tool_call delta missing name / empty arguments"
echo "$STREAM" | grep -q '"function":{"arguments":' || fail "no streamed arguments fragment"
echo "$STREAM" | grep -q '"finish_reason":"tool_calls"' || fail "no tool_calls finish_reason"
echo "$STREAM" | grep -q '^data: \[DONE\]' || fail "missing [DONE] terminator"
pass "streaming tool call (delta sequence + [DONE])"

echo
echo "ALL TOOL SMOKE TESTS PASSED"
