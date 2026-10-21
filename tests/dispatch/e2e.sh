#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# e2e.sh — dispatch-layer end-to-end test (Phase G1+G2), Mac-only, no GPU/CUDA.
#
# Starts the real `pantheon router`, spins up stub upstreams speaking minimal
# OpenAI /v1/chat/completions + SSE, registers them mirroring the live fleet
# (saber-orch, qwen36-base, nex-n2-mini, and TWO ornith instances), then asserts:
#   1. /v1/models is deduped (ornith once, live=2); /registry/fleet correct.
#   2. TTL eviction: stop heartbeating an instance -> it drops within ~TTL.
#   3. Least-busy: the freer ornith instance gets the prompt; round-robin on ties.
#   4. Failover: kill the chosen upstream -> router retries the other; unknown -> 503.
#   5. SSE pass-through: streamed chunks arrive unbuffered (staggered timestamps).
#
# Exit 0 = all pass. Requires: the `pantheon` binary (built), python3, curl.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PANTHEON="$ROOT/pantheon"
STUB="$HERE/stub_upstream.py"

ROUTER_PORT=18100
REG="http://127.0.0.1:$ROUTER_PORT"
PIDS=()
FAIL=0

pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1"; FAIL=1; }

cleanup() {
    for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done
    wait 2>/dev/null
}
trap cleanup EXIT

wait_port() {  # host port
    for _ in $(seq 1 50); do
        if (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; then exec 3>&- 3<&-; return 0; fi
        sleep 0.1
    done
    return 1
}

register() {  # model instance port capacity  [transport]
    curl -s -X POST "$REG/registry/register" -d \
      "{\"model_id\":\"$1\",\"instance_id\":\"$2\",\"node_id\":\"node0\",\"endpoint\":\"http://127.0.0.1:$3\",\"transport\":\"${5:-local}\",\"capacity\":$4}" >/dev/null
}

heartbeat() {  # instance active queue free healthy
    curl -s -X POST "$REG/registry/heartbeat" -d \
      "{\"instance_id\":\"$1\",\"load\":{\"active_slots\":$2,\"queue_depth\":$3,\"free_slots\":$4},\"healthy\":$5}" >/dev/null
}

# ---------------------------------------------------------------------------
[ -x "$PANTHEON" ] || { echo "FATAL: build the pantheon binary first ($PANTHEON)"; exit 2; }

echo "== dispatch-layer e2e =="

# 1. Start the router.
"$PANTHEON" router --host 127.0.0.1 --port "$ROUTER_PORT" >/tmp/e2e_router.log 2>&1 &
PIDS+=($!)
wait_port "$ROUTER_PORT" || { echo "FATAL: router did not start"; cat /tmp/e2e_router.log; exit 2; }

# 2. Start stub upstreams (ports mirror the live fleet 8201-8205 shifted +10000).
python3 "$STUB" --port 18201 --tag saber  >/dev/null 2>&1 & PIDS+=($!)
python3 "$STUB" --port 18202 --tag qwen36 >/dev/null 2>&1 & PIDS+=($!)
python3 "$STUB" --port 18203 --tag nex    >/dev/null 2>&1 & PIDS+=($!)
python3 "$STUB" --port 18204 --tag ornithA >/dev/null 2>&1 & PIDS+=($!)
python3 "$STUB" --port 18205 --tag ornithB >/dev/null 2>&1 & PIDS+=($!)
for p in 18201 18202 18203 18204 18205; do wait_port "$p" || { echo "FATAL: stub $p"; exit 2; }; done

# 3. Register the fleet (ornith = TWO instances of ONE model_id).
register saber-orch  saber-1  18201 4
register qwen36-base qwen36-1 18202 4
register nex-n2-mini nex-1    18203 4
register ornith      ornith-A 18204 4
register ornith      ornith-B 18205 4
# initial heartbeats so both ornith are live
heartbeat ornith-A 0 0 4 true
heartbeat ornith-B 0 0 4 true

# --- ASSERT 1: /v1/models deduped -----------------------------------------
MODELS="$(curl -s "$REG/v1/models")"
N_MODELS="$(grep -o '"object":"model"' <<<"$MODELS" | wc -l | tr -d ' ')"
N_ORNITH="$(grep -o '"id":"ornith"' <<<"$MODELS" | wc -l | tr -d ' ')"
[ "$N_MODELS" = "4" ] && pass "/v1/models has 4 deduped models" || fail "/v1/models model count = $N_MODELS (want 4): $MODELS"
[ "$N_ORNITH" = "1" ] && pass "ornith appears once" || fail "ornith count = $N_ORNITH (want 1)"
grep -q '"live":2' <<<"$MODELS" && pass "ornith live=2" || fail "ornith live count wrong: $MODELS"

FLEET="$(curl -s "$REG/registry/fleet")"
grep -q '"instance_id":"ornith-A"' <<<"$FLEET" && grep -q '"instance_id":"ornith-B"' <<<"$FLEET" \
  && pass "/registry/fleet lists both ornith instances" || fail "fleet missing an ornith instance: $FLEET"
grep -q '"total_free_slots":8' <<<"$FLEET" && pass "ornith total_free_slots=8" || fail "ornith total_free_slots wrong: $FLEET"

# --- ASSERT 3: least-busy --------------------------------------------------
# Make ornith-A clearly freer (4 free vs 1 free) -> all prompts route to A.
heartbeat ornith-A 0 0 4 true
heartbeat ornith-B 3 0 1 true
HITS_A=0; HITS_B=0
for _ in $(seq 1 6); do
  R="$(curl -s -X POST "$REG/v1/chat/completions" -d '{"model":"ornith","messages":[{"role":"user","content":"hi"}]}')"
  case "$R" in
    *answered-by:ornithA*) HITS_A=$((HITS_A+1));;
    *answered-by:ornithB*) HITS_B=$((HITS_B+1));;
  esac
done
[ "$HITS_A" = "6" ] && [ "$HITS_B" = "0" ] && pass "least-busy routed all 6 to freer ornith-A" \
  || fail "least-busy split A=$HITS_A B=$HITS_B (want 6/0)"

# round-robin on ties: equalize load, expect BOTH to be hit
heartbeat ornith-A 0 0 4 true
heartbeat ornith-B 0 0 4 true
HITS_A=0; HITS_B=0
for _ in $(seq 1 8); do
  R="$(curl -s -X POST "$REG/v1/chat/completions" -d '{"model":"ornith","messages":[]}')"
  case "$R" in *ornithA*) HITS_A=$((HITS_A+1));; *ornithB*) HITS_B=$((HITS_B+1));; esac
done
[ "$HITS_A" -gt 0 ] && [ "$HITS_B" -gt 0 ] && pass "round-robin spread ties across A=$HITS_A B=$HITS_B" \
  || fail "round-robin did not spread (A=$HITS_A B=$HITS_B)"

# --- ASSERT 3b: session affinity + X-Session-Id forwarding (PC0.2) ---------
# Both ornith replicas equally free: WITHOUT a session id, ties round-robin
# (asserted above). WITH a session id, every turn must stick to the SAME
# instance (its prefix cache holds the session's KV), and the router must
# forward X-Session-Id upstream (the stub echoes what it received).
heartbeat ornith-A 0 0 4 true
heartbeat ornith-B 0 0 4 true
SID="client-e2e-42"
FIRST=""
STICKY=1
FORWARDED=1
for _ in $(seq 1 6); do
  R="$(curl -s -X POST "$REG/v1/chat/completions" \
        -H "X-Session-Id: $SID" \
        -d '{"model":"ornith","messages":[{"role":"user","content":"hi"}]}')"
  case "$R" in
    *answered-by:ornithA*) WHO=ornithA;;
    *answered-by:ornithB*) WHO=ornithB;;
    *) WHO=none;;
  esac
  case "$R" in *"session:$SID"*) : ;; *) FORWARDED=0;; esac
  [ -z "$FIRST" ] && FIRST="$WHO"
  [ "$WHO" = "$FIRST" ] || STICKY=0
done
[ "$STICKY" = "1" ] && [ "$FIRST" != "none" ] \
  && pass "session affinity: all 6 turns of $SID stuck to $FIRST" \
  || fail "session affinity broke (first=$FIRST sticky=$STICKY)"
[ "$FORWARDED" = "1" ] && pass "X-Session-Id forwarded upstream" \
  || fail "X-Session-Id not forwarded to the upstream"

# A DIFFERENT session may land anywhere, but must also be self-consistent.
SID2="client-e2e-99"
R2="$(curl -s -X POST "$REG/v1/chat/completions" -H "X-Session-Id: $SID2" \
      -d '{"model":"ornith","messages":[]}')"
case "$R2" in
  *answered-by:ornith*) pass "a new session routes + records";;
  *) fail "new session did not route: $R2";;
esac

# --- ASSERT 4: failover + unknown model ------------------------------------
# Kill ornith-B's stub and replace it with a DEAD upstream (accepts+closes).
# Bias so B is chosen first, prove the router fails over to A.
kill "${PIDS[5]}" 2>/dev/null   # ornith-B stub (index: router=0, 5 stubs =1..5 -> B is [5])
wait_port 18205 >/dev/null 2>&1 || true
sleep 0.2
python3 "$STUB" --port 18205 --tag ornithB --mode dead >/dev/null 2>&1 & PIDS+=($!)
wait_port 18205 || { echo "FATAL: dead stub 18205"; exit 2; }
# Make B look freer so it is the PRIMARY pick; A is the failover target.
heartbeat ornith-A 3 0 1 true
heartbeat ornith-B 0 0 4 true
R="$(curl -s -X POST "$REG/v1/chat/completions" -d '{"model":"ornith","messages":[]}')"
case "$R" in
  *answered-by:ornithA*) pass "failover: dead primary (B) -> router retried A";;
  *) fail "failover did not reach a healthy instance: $R";;
esac

# unknown model -> 503
CODE="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$REG/v1/chat/completions" -d '{"model":"nonesuch","messages":[]}')"
[ "$CODE" = "503" ] && pass "unknown model -> 503" || fail "unknown model returned $CODE (want 503)"

# --- ASSERT 5: SSE pass-through (unbuffered) -------------------------------
# Stream from saber-orch; record arrival timestamps of the 3 chunks. If the
# router buffered, they'd all arrive together; the stub sleeps 0.15s between
# chunks, so an unbuffered relay shows staggered arrivals spanning >0.2s.
STAMPS="$(curl -s -N -X POST "$REG/v1/chat/completions" \
  -d '{"model":"saber-orch","stream":true,"messages":[]}' \
  | while IFS= read -r line; do
      [ -n "$line" ] && printf '%s %s\n' "$(python3 -c 'import time;print(f"{time.time():.3f}")')" "$line"
    done)"
N_CHUNKS="$(grep -c 'chat.completion.chunk' <<<"$STAMPS")"
grep -q '\[DONE\]' <<<"$STAMPS" && DONE_OK=1 || DONE_OK=0
FIRST_TS="$(grep 'chat.completion.chunk' <<<"$STAMPS" | head -1 | awk '{print $1}')"
LAST_TS="$(grep 'chat.completion.chunk' <<<"$STAMPS" | tail -1 | awk '{print $1}')"
SPAN="$(python3 -c "print(round(${LAST_TS:-0}-${FIRST_TS:-0},3))" 2>/dev/null || echo 0)"
[ "$N_CHUNKS" = "3" ] && pass "SSE relayed all 3 chunks" || fail "SSE chunk count = $N_CHUNKS (want 3)"
[ "$DONE_OK" = "1" ] && pass "SSE terminal [DONE] forwarded" || fail "SSE [DONE] not forwarded"
python3 -c "import sys; sys.exit(0 if ${SPAN:-0} > 0.2 else 1)" \
  && pass "SSE arrived unbuffered (chunk span ${SPAN}s > 0.2s)" \
  || fail "SSE chunks arrived together (span ${SPAN}s) -> buffered"

# --- ASSERT 2: TTL eviction ------------------------------------------------
# Register a throwaway instance, stop heartbeating it, wait > TTL, expect drop.
register ttl-model ttl-1 18201 1
heartbeat ttl-1 0 0 1 true
grep -q '"id":"ttl-model"' <<<"$(curl -s "$REG/v1/models")" && pass "TTL: instance present before expiry" || fail "TTL: instance not present"
echo "  (waiting ~17s for TTL eviction — TTL=15s, sweep=2s ...)"
sleep 17
MODELS2="$(curl -s "$REG/v1/models")"
grep -q '"id":"ttl-model"' <<<"$MODELS2" && fail "TTL: instance NOT evicted after expiry" || pass "TTL: instance evicted after ~TTL"

# ---------------------------------------------------------------------------
echo
if [ "$FAIL" = "0" ]; then echo "ALL DISPATCH E2E ASSERTIONS PASSED"; else echo "SOME DISPATCH E2E ASSERTIONS FAILED"; fi
exit "$FAIL"
