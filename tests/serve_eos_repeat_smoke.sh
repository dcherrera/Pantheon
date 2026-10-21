#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# serve_eos_repeat_smoke.sh — regression test for the "zero completion tokens
# after an EOS-terminated turn" defect.
#
# The defect: the GPU sparse top-k logits presentation kept its "is this buffer
# already -inf-filled" bookkeeping on the shared GPU context, keyed by the host
# logits buffer POINTER.  A session's pn_state is freed when its prefix-cache
# slot is recycled, and the allocator commonly hands the same address back to
# the next session — so the guard still matched, the full -inf refill was
# skipped, and the PREVIOUS sequence's last survivors lingered as finite
# entries in the NEW sequence's step-0 logits.  When the previous sequence
# ended on EOS, EOS was among those survivors and won at step 0: the new turn
# emitted zero tokens with finish_reason "stop" — which then re-armed the leak,
# latching the model until reload.
#
# Why the whole optimization campaign missed it: every benchmark request ran to
# max_tokens and never emitted EOS, so the survivors left behind were ordinary
# tokens.  This test therefore insists on turns that END ON EOS.
#
# Asserts, for N consecutive turns each ending on EOS:
#   1. Turn 1 really ends on EOS (finish_reason "stop", tokens < max_tokens) —
#      otherwise the test is not exercising the arming condition at all.
#   2. EVERY turn returns completion_tokens > 0 and non-empty text.
#   3. Holds with the session released + a fresh session id each turn (slot
#      release / re-acquire, i.e. the pn_state free + realloc path).
#   4. Holds at CONCURRENCY=1 and CONCURRENCY=2 (single-user and batched slot
#      drivers are separate code paths; the defect lived on the batched
#      prefill's logits fetch).
#
# Usage (server must already be running):
#   BASE_URL=http://127.0.0.1:8201 MODEL=saber-orch \
#     tests/serve_eos_repeat_smoke.sh
#
# Env:
#   BASE_URL     default http://127.0.0.1:8201
#   MODEL        default saber-orch
#   TURNS        turns per concurrency level (default 6)
#   CONCURRENCY  space-separated levels to run (default "1 2")
#   MAX_TOKENS   default 400 (must be roomy enough that a short answer ends on
#                EOS rather than on length — assertion 1 checks this)
#   PROMPT       default a prompt with a short, EOS-terminated answer
#   REQ_TIMEOUT  per-request curl timeout, seconds (default 180)

set -u

BASE_URL="${BASE_URL:-http://127.0.0.1:8201}"
MODEL="${MODEL:-saber-orch}"
TURNS="${TURNS:-6}"
CONCURRENCY="${CONCURRENCY:-1 2}"
MAX_TOKENS="${MAX_TOKENS:-400}"
PROMPT="${PROMPT:-Reply with only: ok}"
REQ_TIMEOUT="${REQ_TIMEOUT:-180}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0
ok()   { echo "PASS: $*"; pass=$((pass+1)); }
bad()  { echo "FAIL: $*" >&2; fail=$((fail+1)); }

command -v python3 >/dev/null || { echo "FAIL: python3 required" >&2; exit 1; }

# One turn: POST a chat completion under a fresh session id, then release the
# session (so the next turn re-acquires a recycled prefix-cache slot — the exact
# free/realloc path the defect rode in on).  Writes
# "<http_code> <completion_tokens> <finish_reason> <text_len>" to $2.
one_turn() {
    local sid="$1" out="$2" body code
    body="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"temperature":0}' \
            "$MODEL" "$PROMPT" "$MAX_TOKENS")"
    code="$(curl -s -m "$REQ_TIMEOUT" -o "$WORK/$sid.json" -w '%{http_code}' \
            "$BASE_URL/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -H "X-Session-Id: $sid" -d "$body")"
    curl -s -m 10 -o /dev/null "$BASE_URL/session/release" \
        -H 'Content-Type: application/json' \
        -d "$(printf '{"name":"%s","session_id":"%s"}' "$MODEL" "$sid")"
    python3 - "$code" "$WORK/$sid.json" >"$out" <<'PY'
import json, sys
code = sys.argv[1]
try:
    d = json.load(open(sys.argv[2]))
    ch = d["choices"][0]
    msg = ch.get("message", {}) or {}
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    print(code, d.get("usage", {}).get("completion_tokens", -1),
          ch.get("finish_reason", "?"), len(text))
except Exception:
    print(code, -1, "parse_error", 0)
PY
}

echo "== serve_eos_repeat_smoke: $MODEL @ $BASE_URL (turns=$TURNS max_tokens=$MAX_TOKENS)"

for k in $CONCURRENCY; do
    echo "-- concurrency K=$k"
    saw_eos=0
    turn=0
    while [ "$turn" -lt "$TURNS" ]; do
        sids=""
        j=0
        while [ "$j" -lt "$k" ] && [ "$turn" -lt "$TURNS" ]; do
            sid="eosrep-k${k}-$$-$turn"
            one_turn "$sid" "$WORK/$sid.res" &
            sids="$sids $sid"
            j=$((j+1)); turn=$((turn+1))
        done
        wait
        for sid in $sids; do
            read -r code toks finish tlen <"$WORK/$sid.res"
            if [ "$code" != "200" ]; then
                # A 429 means the session slots are full — a test-harness
                # problem (stale sticky sessions), NOT this defect.  Say so
                # loudly rather than reporting a false pass/fail.
                bad "$sid: HTTP $code (not 200) — release stale sessions and retry"
                continue
            fi
            if [ "$toks" -le 0 ] || [ "$tlen" -le 0 ]; then
                bad "$sid: zero-token reply (completion_tokens=$toks text_len=$tlen finish=$finish) — the EOS latch is back"
            else
                ok "$sid: completion_tokens=$toks finish=$finish"
            fi
            # Assertion 1: at least one turn must genuinely end on EOS,
            # otherwise this run never armed the defect.
            if [ "$finish" = "stop" ] && [ "$toks" -gt 0 ] && \
               [ "$toks" -lt "$MAX_TOKENS" ]; then
                saw_eos=1
            fi
        done
    done
    if [ "$saw_eos" -eq 1 ]; then
        ok "K=$k: at least one turn ended on EOS (the arming condition was exercised)"
    else
        bad "K=$k: no turn ended on EOS (every reply hit max_tokens) — this run could not have caught the defect; raise MAX_TOKENS or use a prompt with a short answer"
    fi
done

echo "== $pass passed, $fail failed"
[ "$fail" -eq 0 ]
