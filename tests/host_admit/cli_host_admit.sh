#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# cli_host_admit.sh — the CLI refuses / caps a host KV ring that does not fit,
# before allocating it. Uses the PN_HOST_ADMIT_TEST_AVAIL_MB hook to pretend
# the host has 3 GiB available (1.5 GiB above the default reserve).
#
#   1. explicit window that does not fit (PN_SESSION_MAX_CTX=20000, ~4.6 GB of
#      fp32 KV on qwen3-0.6b — the 2026-10-27 bank OOM) -> exit != 0, the
#      refusal names the ring, and peak RSS stays small (nothing allocated);
#   2. the same through --ctx-cap -> refused the same way;
#   3. default window (40000) -> capped with a log line, generation runs, and
#      peak RSS stays under the fake budget + weights;
#   4. PN_HOST_ADMIT=0 -> the escape hatch: no refusal (window kept small so
#      the check itself is cheap).
#
# RED on main: case 1 allocates the full ring (RSS ~5 GB) and succeeds.
# Run main's binary under foundry's metal/run_guarded.sh when proving red.
#
# Usage: tests/host_admit/cli_host_admit.sh [MODEL_GGUF] [BIN]
set -u

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${1:-${PANTHEON_TEST_MODEL:-$REPO_DIR/models/qwen3-0.6b-Q8_0.gguf}}"
BIN="${2:-$REPO_DIR/pantheon}"
OUT="$REPO_DIR/build/host_admit"
mkdir -p "$OUT"
FAILS=0

fail() { echo "FAIL: $*"; FAILS=$((FAILS + 1)); }
pass() { echo "PASS: $*"; }

[ -x "$BIN" ] || { echo "FAIL: binary not built: $BIN"; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: model not found: $MODEL"; exit 1; }

# Peak RSS in MiB of a command, written to $OUT/<tag>.rss; stderr to <tag>.log.
run_measured() {
    local tag="$1"; shift
    local timelog="$OUT/$tag.time"
    if [ "$(uname)" = "Darwin" ]; then
        /usr/bin/time -l "$@" >"$OUT/$tag.out" 2>"$OUT/$tag.log"
        local rc=$?
        awk '/maximum resident set size/ { printf "%d", $1 / 1048576 }' \
            "$OUT/$tag.log" >"$OUT/$tag.rss"
    else
        /usr/bin/time -v -o "$timelog" "$@" >"$OUT/$tag.out" 2>"$OUT/$tag.log"
        local rc=$?
        awk -F: '/Maximum resident set size/ { printf "%d", $2 / 1024 }' \
            "$timelog" >"$OUT/$tag.rss"
    fi
    return $rc
}

ARGS=(--model "$MODEL" --prompt "Hi" --max-tokens 2 --temperature 0)
export PN_HOST_ADMIT_TEST_AVAIL_MB=3072

# 1. explicit window via PN_SESSION_MAX_CTX
PN_SESSION_MAX_CTX=20000 run_measured explicit_env "$BIN" "${ARGS[@]}"
rc=$?
rss=$(cat "$OUT/explicit_env.rss")
[ $rc -ne 0 ] && pass "explicit 20000 window refused (exit $rc)" \
              || fail "explicit 20000 window was NOT refused (exit 0)"
grep -q "host KV ring (1 session x 20000 positions) needs 4.3 GiB" "$OUT/explicit_env.log" \
    && pass "refusal names the ring and its size" \
    || fail "refusal message missing: $(grep -m1 -i 'host' "$OUT/explicit_env.log")"
[ "${rss:-99999}" -lt 1536 ] && pass "peak RSS ${rss} MiB < 1536 (nothing allocated)" \
                            || fail "peak RSS ${rss} MiB >= 1536 (ring was allocated)"

# 2. explicit window via --ctx-cap
run_measured explicit_flag "$BIN" "${ARGS[@]}" --ctx-cap 20000
rc=$?
[ $rc -ne 0 ] && grep -q "needs 4.3 GiB" "$OUT/explicit_flag.log" \
    && pass "--ctx-cap 20000 refused with the reason" \
    || fail "--ctx-cap 20000 not refused (exit $rc)"

# 3. default window: capped, runs
run_measured default_cap "$BIN" "${ARGS[@]}"
rc=$?
rss=$(cat "$OUT/default_cap.rss")
[ $rc -eq 0 ] && pass "default window runs" || fail "default window failed (exit $rc)"
grep -q "host RAM caps the default context 40000 -> " "$OUT/default_cap.log" \
    && pass "default capped with a log line" || fail "no default-cap log line"
[ "${rss:-99999}" -lt 3584 ] && pass "peak RSS ${rss} MiB < 3584" \
                            || fail "peak RSS ${rss} MiB >= 3584"

# 4. escape hatch
PN_HOST_ADMIT=0 PN_SESSION_MAX_CTX=1024 PN_HOST_ADMIT_TEST_AVAIL_MB=1 \
    run_measured disabled "$BIN" "${ARGS[@]}"
rc=$?
[ $rc -eq 0 ] && ! grep -q "REFUSED" "$OUT/disabled.log" \
    && pass "PN_HOST_ADMIT=0 disables the check" \
    || fail "PN_HOST_ADMIT=0 still refused (exit $rc)"

echo "cli_host_admit: $FAILS failure(s)"
[ $FAILS -eq 0 ]
