#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# serve_smoke.sh — end-to-end smoke test for `pantheon serve`.
#
# Starts the OpenAI-compatible server on the host CPU path, then exercises:
#   1. GET  /v1/models                     -> OpenAI model list shape
#   2. POST /v1/chat/completions           -> non-streaming chat.completion
#   3. POST /v1/chat/completions (stream)  -> SSE chunks + [DONE] framing
#   4. error handling                      -> OpenAI-shaped error JSON
#
# Usage:
#   tests/serve_smoke.sh [MODEL_GGUF] [PORT]
# With no MODEL_GGUF, uses $PANTHEON_TEST_MODEL, else the first .gguf in models/.
# Default port 8081.
#
# Requires: a built ./pantheon, curl, python3 (for JSON assertions).
set -u

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Model: first argument, else $PANTHEON_TEST_MODEL, else the first .gguf in models/.
MODEL="${1:-${PANTHEON_TEST_MODEL:-$(ls "$REPO_DIR"/models/*.gguf 2>/dev/null | head -1)}}"
PORT="${2:-8081}"
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

echo "Starting server: $BIN serve --model $MODEL --port $PORT"
"$BIN" serve --model "$MODEL" --host "$HOST" --port "$PORT" >/tmp/serve_smoke.log 2>&1 &
SERVER_PID=$!

# Wait for the listen line (model load can take a while).
for _ in $(seq 1 120); do
    grep -q "listening on" /tmp/serve_smoke.log && break
    kill -0 "$SERVER_PID" 2>/dev/null || fail "server exited early; see /tmp/serve_smoke.log"
    sleep 1
done
grep -q "listening on" /tmp/serve_smoke.log || fail "server did not come up in time"
pass "server is listening"

# 1. GET /v1/models
MODELS="$(curl -s "${BASE}/v1/models")"
echo "$MODELS" | python3 -c 'import sys,json; d=json.load(sys.stdin); assert d["object"]=="list"; assert d["data"][0]["id"]; print("  model id:",d["data"][0]["id"])' \
    || fail "/v1/models did not return a valid OpenAI model list"
pass "/v1/models"

# 2. Non-streaming completion
RESP="$(curl -s "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"temperature":0,"max_tokens":200}')"
echo "$RESP" | python3 -c '
import sys,json
d=json.load(sys.stdin)
assert d["object"]=="chat.completion", d
msg=d["choices"][0]["message"]
assert msg["role"]=="assistant"
assert isinstance(msg["content"],str) and msg["content"].strip(), "empty content"
assert d["usage"]["completion_tokens"]>0
print("  content:",repr(msg["content"][:80]))
print("  finish:",d["choices"][0]["finish_reason"])
' || fail "non-streaming completion invalid"
pass "non-streaming /v1/chat/completions"

# 3. Streaming completion (SSE)
STREAM="$(curl -sN "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hello in 3 words."}],"temperature":0,"max_tokens":120,"stream":true}')"
echo "$STREAM" | grep -q '"object":"chat.completion.chunk"' || fail "no SSE chunk objects"
echo "$STREAM" | grep -q '"delta":{"role":"assistant"}' || fail "no opening role delta"
# Any terminal reason is fine here — this asserts SSE framing, not model
# behaviour. A reasoning model can legitimately hit "length" inside <think>.
echo "$STREAM" | grep -qE '"finish_reason":"(stop|length)"' || fail "no terminal finish_reason"
echo "$STREAM" | grep -q '^data: \[DONE\]'               || fail "missing [DONE] terminator"
# A reasoning model streams reasoning_content deltas until </think>; either
# kind proves deltas are flowing, which is what this test is checking.
echo "$STREAM" | grep -qE '"delta":\{"(reasoning_)?content":' || fail "no content deltas"
pass "streaming /v1/chat/completions (SSE + [DONE])"

# 4. Error handling
CODE="$(curl -s -o /tmp/serve_err.json -w '%{http_code}' "${BASE}/v1/chat/completions" \
    -H 'Content-Type: application/json' -d '{bad json')"
[ "$CODE" = "400" ] || fail "malformed JSON should be 400 (got $CODE)"
python3 -c 'import json; d=json.load(open("/tmp/serve_err.json")); assert d["error"]["type"]' \
    || fail "error response not OpenAI-shaped"
pass "error handling (400 + OpenAI error shape)"

echo
echo "ALL SMOKE TESTS PASSED"

# ----------------------------------------------------------------------------
# opencode provider config — point opencode at this server. In opencode.json:
#
#   {
#     "provider": {
#       "pantheon": {
#         "npm": "@ai-sdk/openai-compatible",
#         "name": "Pantheon (local)",
#         "options": { "baseURL": "http://127.0.0.1:8080/v1" },
#         "models": { "nex-agi_Nex-N2-mini-Q3_K_M.gguf": { "name": "Nex-N2-mini" } }
#       }
#     }
#   }
#
# The model id must match GET /v1/models (the gguf file name, or --served-name).
# ----------------------------------------------------------------------------
