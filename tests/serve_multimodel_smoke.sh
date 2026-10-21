#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# serve_multimodel_smoke.sh — end-to-end smoke test for Phase 7 multi-model
# serving (registry routing + runtime load/unload).
#
# Serves ONE small GGUF under TWO names (CPU path — die placement is exercised
# separately on the K80 test machine) and checks:
#   1. GET  /v1/models                    -> lists both resident models
#   2. routing: model field selects the entry; response echoes the routed id
#   3. unknown model                      -> 404 model_not_found
#   4. per-model serialization: busy model 429s while the OTHER model serves
#   5. admin: /admin/models/load a third alias -> chat to it -> unload -> 404
#   6. GET  /admin/dies                   -> die-manager snapshot shape
#   7. back-compat: single-model serve accepts ANY model value (old behavior)
#
# Usage:
#   tests/serve_multimodel_smoke.sh [MODEL_GGUF] [PORT]
# Defaults to the local qwen3-0.6b and port 8083. The 0.6b is a thinking
# model, so prompts use /no_think to keep `content` non-empty.
#
# Requires: a built ./pantheon, curl, python3.
set -u

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-${REPO_DIR}/models/qwen3-0.6b-Q8_0.gguf}"
PORT="${2:-8083}"
HOST="127.0.0.1"
BASE="http://${HOST}:${PORT}"
BIN="${REPO_DIR}/pantheon"
LOG=/tmp/serve_multimodel_smoke.log

fail() { echo "FAIL: $*" >&2; cleanup; exit 1; }
pass() { echo "PASS: $*"; }

SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

wait_listening() {
    for _ in $(seq 1 120); do
        grep -q "listening on" "$LOG" && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || fail "server exited early; see $LOG"
        sleep 1
    done
    fail "server did not come up in time"
}

chat() { # chat MODEL_FIELD PROMPT MAX_TOKENS -> body in /tmp/mm_chat.json, code in CHAT_CODE
    local model_json=""
    [ -n "$1" ] && model_json="\"model\":\"$1\","
    CHAT_CODE="$(curl -s -o /tmp/mm_chat.json -w '%{http_code}' \
        "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{${model_json}\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],\"temperature\":0,\"max_tokens\":$3}")"
}

[ -x "$BIN" ] || fail "pantheon binary not built (run: make USE_BLAS=1 pantheon)"
[ -f "$MODEL" ] || fail "model not found: $MODEL"

echo "Starting server: 2 aliases of $MODEL"
"$BIN" serve --model "$MODEL" --name alpha --model "$MODEL" --name beta \
    --host "$HOST" --port "$PORT" > "$LOG" 2>&1 &
SERVER_PID=$!
wait_listening
pass "server is listening (alpha + beta)"

# 1. /v1/models lists both
curl -s "${BASE}/v1/models" | python3 -c '
import sys,json
d=json.load(sys.stdin)
ids=sorted(m["id"] for m in d["data"])
assert ids==["alpha","beta"], ids
print("  models:",ids)' || fail "/v1/models does not list both models"
pass "/v1/models lists both resident models"

# 2. routing by the model field (response echoes the routed id)
for M in alpha beta; do
    chat "$M" 'Say only the word hello. /no_think' 60
    [ "$CHAT_CODE" = "200" ] || fail "chat to $M returned $CHAT_CODE"
    python3 < /tmp/mm_chat.json -c "
import sys,json
d=json.load(sys.stdin)
assert d['model']=='$M', d['model']
c=d['choices'][0]['message']['content']
assert isinstance(c,str) and c.strip(), 'empty content'
print('  $M ->',repr(c[:40]))" || fail "chat routed to $M invalid"
done
pass "routing by model field (alpha + beta)"

# 3. unknown model -> 404 model_not_found
chat "does-not-exist" 'hi' 8
[ "$CHAT_CODE" = "404" ] || fail "unknown model should 404 (got $CHAT_CODE)"
python3 -c 'import json; d=json.load(open("/tmp/mm_chat.json")); assert d["error"]["type"]=="model_not_found"' \
    || fail "404 error not model_not_found-shaped"
pass "unknown model -> 404 model_not_found"

# 4. per-model serialization: keep alpha busy (long think), then alpha=429 / beta=200
curl -s "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"alpha","messages":[{"role":"user","content":"Explain prime numbers."}],"temperature":0,"max_tokens":400,"stream":true}' \
    > /tmp/mm_busy.out &
BUSY_PID=$!
sleep 2
chat alpha 'hi /no_think' 8
[ "$CHAT_CODE" = "429" ] || fail "busy alpha should 429 (got $CHAT_CODE)"
chat beta 'Say only the word hello. /no_think' 60
[ "$CHAT_CODE" = "200" ] || fail "beta should serve while alpha is busy (got $CHAT_CODE)"
wait $BUSY_PID 2>/dev/null
pass "per-model serialization (busy alpha 429s, beta serves concurrently)"

# 5. runtime load/unload via the admin surface
CODE="$(curl -s -o /tmp/mm_admin.json -w '%{http_code}' "${BASE}/admin/models/load" \
    -H 'Content-Type: application/json' -d "{\"path\":\"${MODEL}\",\"name\":\"gamma\"}")"
[ "$CODE" = "200" ] || fail "/admin/models/load gamma failed ($CODE: $(cat /tmp/mm_admin.json))"
python3 -c 'import json; d=json.load(open("/tmp/mm_admin.json")); assert d["id"]=="gamma"' \
    || fail "load response not the gamma model row"
curl -s "${BASE}/v1/models" | python3 -c '
import sys,json
ids=sorted(m["id"] for m in json.load(sys.stdin)["data"])
assert ids==["alpha","beta","gamma"], ids' || fail "gamma not listed after load"
chat gamma 'Say only the word hello. /no_think' 60
[ "$CHAT_CODE" = "200" ] || fail "chat to freshly-loaded gamma failed ($CHAT_CODE)"

# duplicate name refused
CODE="$(curl -s -o /tmp/mm_admin.json -w '%{http_code}' "${BASE}/admin/models/load" \
    -H 'Content-Type: application/json' -d "{\"path\":\"${MODEL}\",\"name\":\"gamma\"}")"
[ "$CODE" = "409" ] || fail "duplicate load should 409 (got $CODE)"

CODE="$(curl -s -o /tmp/mm_admin.json -w '%{http_code}' "${BASE}/admin/models/unload" \
    -H 'Content-Type: application/json' -d '{"name":"gamma"}')"
[ "$CODE" = "200" ] || fail "/admin/models/unload gamma failed ($CODE)"
chat gamma 'hi' 8
[ "$CHAT_CODE" = "404" ] || fail "unloaded gamma should 404 (got $CHAT_CODE)"
curl -s "${BASE}/v1/models" | python3 -c '
import sys,json
ids=sorted(m["id"] for m in json.load(sys.stdin)["data"])
assert ids==["alpha","beta"], ids' || fail "gamma still listed after unload"
pass "runtime load/unload (live /v1/models, 409 duplicate, 404 after unload)"

# 6. /admin/dies shape (empty on the CPU host; occupancy on the K80 test machine)
curl -s "${BASE}/admin/dies" | python3 -c '
import sys,json
d=json.load(sys.stdin)
assert d["object"]=="list" and "n_dies" in d and isinstance(d["data"],list)
print("  n_dies:",d["n_dies"])' || fail "/admin/dies malformed"
pass "/admin/dies snapshot shape"

cleanup; SERVER_PID=""; sleep 1

# 7. back-compat: single-model serve routes ANY model value to the one model
echo "Restarting single-model for back-compat check"
"$BIN" serve --model "$MODEL" --host "$HOST" --port "$PORT" > "$LOG" 2>&1 &
SERVER_PID=$!
wait_listening
chat "some-stale-client-name" 'Say only the word hello. /no_think' 60
[ "$CHAT_CODE" = "200" ] || fail "single-model serve should accept any model value (got $CHAT_CODE)"
pass "single-model back-compat (any model value accepted)"

echo
echo "ALL MULTI-MODEL SMOKE TESTS PASSED"
