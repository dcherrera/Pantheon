#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# serve_concurrency_smoke.sh — prove `pantheon serve` handles connections
# concurrently while serializing generation. Reproduces opencode's traffic
# pattern against a LIVE server: a streaming chat in the background, and WHILE it
# generates, several GET /v1/models health checks plus a second chat.
#
# Asserts:
#   1. Health checks (GET /v1/models) return promptly *during* a generation
#      (i.e. the accept loop is not jammed by the in-flight chat).
#   2. The first (streaming) chat streams multiple chunks and reaches [DONE].
#   3. The second chat is either served (200) after, or cleanly rejected (429) —
#      never a hang / jam.
#   4. The server is still responsive at the end.
#
# Usage:
#   BASE_URL=http://127.0.0.1:8091 MODEL=nex-agi_Nex-N2-mini-Q3_K_M.gguf \
#     tests/serve_concurrency_smoke.sh
#
# The server must already be running (this test does not start or stop it).

set -u

BASE_URL="${BASE_URL:-http://127.0.0.1:8091}"
MODEL="${MODEL:-nex-agi_Nex-N2-mini-Q3_K_M.gguf}"
# A health check must come back well under this many seconds even mid-generation.
HEALTH_MAX_SECONDS="${HEALTH_MAX_SECONDS:-5}"
# Seconds to let the streaming chat warm up so the probes below land squarely
# mid-generation. (curl writing to a file is block-buffered, so we time the
# overlap with a fixed warmup rather than polling the buffered output.)
STREAM_WARMUP="${STREAM_WARMUP:-6}"
# Upper bound on the whole streaming chat.
STREAM_TOTAL_TIMEOUT="${STREAM_TOTAL_TIMEOUT:-600}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0
ok()   { echo "PASS: $*"; pass=$((pass+1)); }
bad()  { echo "FAIL: $*"; fail=$((fail+1)); }

echo "== serve concurrency smoke =="
echo "base_url=$BASE_URL model=$MODEL"

# Pre-flight: the server answers a models health check at all.
pre=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE_URL/v1/models")
if [ "$pre" != "200" ]; then
  echo "ABORT: server not reachable at $BASE_URL (GET /v1/models -> $pre)"
  exit 2
fi
echo "server reachable (pre-flight GET /v1/models -> 200)"

stream_body() {
  cat <<JSON
{"model":"$MODEL","stream":true,"max_tokens":400,"temperature":0.7,
 "messages":[{"role":"user","content":"Write a long detailed story about a dragon and a knight."}]}
JSON
}

second_body() {
  cat <<JSON
{"model":"$MODEL","stream":false,"max_tokens":32,"temperature":0.7,
 "messages":[{"role":"user","content":"Say hello in one short sentence."}]}
JSON
}

# --- 1. Launch the streaming chat in the background. ---------------------------
STREAM_OUT="$WORK/stream.out"
( curl -sN --max-time "$STREAM_TOTAL_TIMEOUT" -X POST \
    -H 'Content-Type: application/json' \
    --data "$(stream_body)" \
    "$BASE_URL/v1/chat/completions" > "$STREAM_OUT" 2>/dev/null ) &
STREAM_PID=$!
echo "launched streaming chat (pid $STREAM_PID)"

# Let generation warm up, then confirm it is genuinely still in flight so the
# probes below land mid-generation.
sleep "$STREAM_WARMUP"
if kill -0 "$STREAM_PID" 2>/dev/null; then
  ok "streaming chat still generating after ${STREAM_WARMUP}s warmup (generation in flight)"
else
  bad "streaming chat finished within ${STREAM_WARMUP}s — too fast to test overlap (raise max_tokens or lower STREAM_WARMUP)"
fi

# --- 2. WHILE generating: fire concurrent health checks, time each. -----------
if kill -0 "$STREAM_PID" 2>/dev/null; then
  echo "-- firing 5 concurrent GET /v1/models during generation --"
  health_pids=""
  for i in 1 2 3 4 5; do
    ( curl -s -o /dev/null -w '%{http_code} %{time_total}\n' \
        --max-time "$HEALTH_MAX_SECONDS" \
        "$BASE_URL/v1/models" > "$WORK/health.$i" 2>/dev/null ) &
    health_pids="$health_pids $!"
  done
  # Wait only for the health-check jobs, NOT the still-running stream job.
  for hp in $health_pids; do wait "$hp"; done

  health_ok=1
  for i in 1 2 3 4 5; do
    line=$(cat "$WORK/health.$i" 2>/dev/null)
    code=$(echo "$line" | awk '{print $1}')
    dur=$(echo "$line"  | awk '{print $2}')
    echo "   health #$i -> code=$code time=${dur}s"
    if [ "$code" != "200" ]; then health_ok=0; fi
  done
  if [ "$health_ok" = "1" ]; then
    ok "all 5 health checks returned 200 within ${HEALTH_MAX_SECONDS}s DURING generation (no accept-loop jam)"
  else
    bad "a health check did not return 200 within ${HEALTH_MAX_SECONDS}s during generation"
  fi

  # --- 3. WHILE still generating: a second chat. ----------------------------
  if kill -0 "$STREAM_PID" 2>/dev/null; then
    echo "-- firing a second POST /v1/chat/completions during generation --"
    sec_code=$(curl -s -o "$WORK/second.out" -w '%{http_code}' \
      --max-time "$STREAM_TOTAL_TIMEOUT" -X POST \
      -H 'Content-Type: application/json' \
      --data "$(second_body)" \
      "$BASE_URL/v1/chat/completions")
    echo "   second chat -> code=$sec_code"
    if [ "$sec_code" = "429" ]; then
      ok "second concurrent chat cleanly rejected with 429 (generation serialized)"
    elif [ "$sec_code" = "200" ]; then
      ok "second chat served (200) after the first released the generation lock"
    else
      bad "second chat returned unexpected code $sec_code"
    fi
  else
    echo "   (first chat already finished before second chat fired; skipping 2nd-chat concurrency assert)"
  fi
else
  echo "   (first chat finished too fast to overlap health checks; model likely tiny)"
fi

# --- 4. Wait for the streaming chat to complete; check it finished cleanly. ---
wait "$STREAM_PID" 2>/dev/null
chunks=$(grep -c '^data:' "$STREAM_OUT" 2>/dev/null || echo 0)
if grep -q '\[DONE\]' "$STREAM_OUT" 2>/dev/null; then
  ok "streaming chat reached [DONE] with $chunks SSE lines"
else
  bad "streaming chat did not reach [DONE] (got $chunks SSE lines)"
fi

# --- 5. Server still responsive afterwards. -----------------------------------
post=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE_URL/v1/models")
if [ "$post" = "200" ]; then
  ok "server still responsive after the run (GET /v1/models -> 200)"
else
  bad "server not responsive after the run (GET /v1/models -> $post)"
fi

echo "== result: $pass passed, $fail failed =="
[ "$fail" = "0" ]
