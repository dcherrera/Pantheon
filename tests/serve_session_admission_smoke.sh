#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# serve_session_admission_smoke.sh — regression test for "a batched model never
# enforced max_sessions, and a new session silently wiped another user's KV"
# (2026-11-05, K80 bank).
#
# The defect: the batched path admitted against the model's prefix cache, which
# holds nothing on that path, so every new session was admitted; it then took
# the least recently active idle decode slot, and the prefill erased the KV of
# the session warm there.  The scheduler now admits: at most max_sessions
# session families hold slots, and the next new one gets an immediate 429.
#
# Run against a LIVE serve whose model under test was loaded with
# --max-sessions 1 on the batched (CUDA) path.  Asserts:
#   1. session A is served;
#   2. a distinct session B gets 429 model_at_session_capacity at once (not
#      after the 30 s slot wait);
#   3. A's next turn is still warm (cached_tokens > 0): B did not take A's slot;
#   4. after POST /session/release of A, B is admitted.
#
# Usage (server must already be running):
#   BASE_URL=http://127.0.0.1:8398 MODEL=orn tests/serve_session_admission_smoke.sh
#
# Env:
#   BASE_URL     default http://127.0.0.1:8398
#   MODEL        default orn
#   REQ_TIMEOUT  per-request curl timeout, seconds (default 600)
#   FAST_S       how fast the refusal must come back, seconds (default 5)

set -u

BASE_URL="${BASE_URL:-http://127.0.0.1:8398}"
MODEL="${MODEL:-orn}"
REQ_TIMEOUT="${REQ_TIMEOUT:-600}"
FAST_S="${FAST_S:-5}"
TAG="adm-$$-$(date +%s)"
SA="$TAG-a"
SB="$TAG-b"
PASS=0
FAIL=0

ok()  { PASS=$((PASS+1)); echo "  PASS  $*"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL  $*"; }

# chat <sid> <user text>: prints "<http code> <seconds> <body>" on one line.
chat() {
    curl -s -m "$REQ_TIMEOUT" -o /dev/stdout -w '\n%{http_code} %{time_total}' \
        "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' \
        -H "X-Session-Id: $1" \
        -d "{\"model\":\"$MODEL\",\"max_tokens\":8,\"temperature\":0,\"messages\":[
             {\"role\":\"system\",\"content\":\"You are a terse assistant. Answer in one word.\"},
             {\"role\":\"user\",\"content\":\"$2\"}]}" |
        python3 -c '
import sys
raw = sys.stdin.read().rsplit("\n", 1)
body, tail = (raw[0], raw[1]) if len(raw) == 2 else ("", raw[0])
print(tail, body.replace("\n", " "))'
}

# field <body> <python expr over d>: a scalar from a JSON body ("" if absent).
field() {
    python3 -c '
import json, sys
try:
    d = json.loads(sys.argv[1]); v = eval(sys.argv[2])
except Exception:
    v = None
print("" if v is None else v)' "$1" "$2"
}

release() {
    curl -s -m 30 -o /dev/null -w '%{http_code}' "$BASE_URL/session/release" \
        -H 'Content-Type: application/json' -d "{\"name\":\"$MODEL\",\"session_id\":\"$1\"}"
}

trap 'release "$SA" >/dev/null; release "$SB" >/dev/null' EXIT

echo "session admission smoke: $BASE_URL model=$MODEL (expects --max-sessions 1)"

read -r code secs body <<< "$(chat "$SA" "Name a colour.")"
echo "  A turn 1: HTTP $code in ${secs}s"
[ "$code" = 200 ] && ok "1. session A is served" || bad "1. session A: HTTP $code $body"

read -r code secs body <<< "$(chat "$SB" "Name a fruit.")"
err=$(field "$body" 'd["error"]["type"]')
echo "  B: HTTP $code in ${secs}s ($err) $(field "$body" 'd["error"]["message"]')"
[ "$code" = 429 ] && [ "$err" = model_at_session_capacity ] &&
    ok "2. distinct session B refused 429 model_at_session_capacity" ||
    bad "2. B was not refused at capacity: HTTP $code $body"
awk -v s="$secs" -v f="$FAST_S" 'BEGIN { exit !(s < f) }' &&
    ok "2b. the refusal came back at once (${secs}s < ${FAST_S}s)" ||
    bad "2b. the refusal took ${secs}s (queued behind the slot wait?)"

read -r code secs body <<< "$(chat "$SA" "Name a colour. Now name another.")"
cached=$(field "$body" 'd["usage"]["prompt_tokens_details"]["cached_tokens"]')
echo "  A turn 2: HTTP $code in ${secs}s, cached_tokens=$cached"
[ "$code" = 200 ] && [ -n "$cached" ] && [ "$cached" -gt 0 ] &&
    ok "3. A's next turn is still warm (cached_tokens $cached)" ||
    bad "3. A's next turn is not warm: HTTP $code cached_tokens='$cached'"

rc=$(release "$SA")
echo "  release A: HTTP $rc"
read -r code secs body <<< "$(chat "$SB" "Name a fruit.")"
echo "  B after release: HTTP $code in ${secs}s"
[ "$code" = 200 ] && ok "4. after A's release B is admitted" ||
    bad "4. B still refused after A's release: HTTP $code $body"

echo "session admission smoke: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
