#!/bin/bash
# tests/local/lib.sh — shared helpers for the Pantheon Local on-hardware scripts.
# Sourced, not run. Runs ON the box from the repo root.
#
#   PN_BIN    binary (default ./pantheon)
#   LT_PORT   test port (default 8411; never 8400 or a fleet port)
#   LT_DIR    state/log dir (default build/local_hw, gitignored)
#   LT_STORE  model store (default $LT_DIR/store)

PN_BIN=${PN_BIN:-./pantheon}
LT_PORT=${LT_PORT:-8411}
LT_DIR=${LT_DIR:-build/local_hw}
LT_STORE=${LT_STORE:-$LT_DIR/store}
LT_FAILS=0

lt_check() {   # lt_check <description> <command...>
    local what=$1; shift
    if "$@"; then echo "  ok   $what"; else echo "  FAIL $what"; LT_FAILS=$((LT_FAILS + 1)); fi
}

lt_start() {   # lt_start [extra pantheon local flags...]
    mkdir -p "$LT_DIR"
    # The inner shell writes its own pid, then execs pantheon: the pid file names
    # the box itself, not setsid's fork.
    setsid sh -c 'echo $$ > "$0"; exec "$@"' "$LT_DIR/pid" \
        "$PN_BIN" local --port "$LT_PORT" --state-dir "$LT_DIR/state" \
        --store "$LT_STORE" "$@" > "$LT_DIR/serve.log" 2>&1 < /dev/null &
    sleep 0.2
    for _ in $(seq 1 240); do
        curl -s -o /dev/null "http://127.0.0.1:$LT_PORT/" && return 0
        sleep 0.5
    done
    echo "pantheon local did not come up; log: $LT_DIR/serve.log"; return 1
}

lt_stop() {
    local pid
    pid=$(cat "$LT_DIR/pid" 2>/dev/null) || return 0
    kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 120); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.5; done
    kill -KILL "$pid" 2>/dev/null
}

lt_pair() {    # prints an admin key (empty if the box has no pairing)
    local code
    code=$(cat "$LT_DIR/state/pair_code" 2>/dev/null) || return 0
    curl -s -X POST "http://127.0.0.1:$LT_PORT/local/pair" \
        -d "{\"code\":\"$code\",\"label\":\"${1:-hwtest}\"}" | sed -n 's/.*"key":"\([^"]*\)".*/\1/p'
}

lt_get() {     # lt_get <key> <path>
    curl -s -H "Authorization: Bearer $1" "http://127.0.0.1:$LT_PORT$2"
}

lt_post() {    # lt_post <key> <path> <json>
    curl -s -X POST -H "Authorization: Bearer $1" "http://127.0.0.1:$LT_PORT$2" -d "$3"
}

lt_count() {   # lt_count <file> <literal> — occurrences of literal in file
    grep -o -- "$2" "$1" | wc -l | tr -d ' '
}

lt_done() {
    echo "$1: $LT_FAILS failure(s)"
    [ "$LT_FAILS" -eq 0 ]
}

# --- catalog fixture (TEST ONLY key; needs a `make LOCAL_TEST_CATALOG_KEY=1` build) ---
LT_REV=0123456789abcdef0123456789abcdef01234567
LT_HF_PORT=${LT_HF_PORT:-8412}
LT_CATALOG=tests/local/fixtures/catalog_e2e.json
LT_MODEL_ID=qwen3-0.6b-q8

lt_hf_start() {   # serve $LT_MODEL_FILE in the HF layout on 127.0.0.1:$LT_HF_PORT
    local www=$LT_DIR/www/teamide/test-fixtures/resolve/$LT_REV
    [ -x build/fixture_httpd ] || make -s build/fixture_httpd || return 1
    mkdir -p "$www"
    ln -sfn "$(readlink -f "${LT_MODEL_FILE:?path to qwen3-0.6b-Q8_0.gguf}")" "$www/qwen3-0.6b-Q8_0.gguf"
    setsid sh -c 'echo $$ > "$0"; exec "$@"' "$LT_DIR/hf.pid" \
        build/fixture_httpd "$LT_HF_PORT" "$LT_DIR/www" > "$LT_DIR/hf.log" 2>&1 < /dev/null &
    sleep 0.5
}

lt_hf_stop() {
    kill "$(cat "$LT_DIR/hf.pid" 2>/dev/null)" 2>/dev/null
}

lt_catalog_flags() {   # flags that point the box at the fixture catalog + HF stand-in
    echo "--allow-http --catalog-url file://$PWD/$LT_CATALOG --hf-base-url http://127.0.0.1:$LT_HF_PORT"
}

lt_wait() {    # lt_wait <seconds> <command...> — poll until the command succeeds
    local secs=$1; shift
    for _ in $(seq 1 "$secs"); do "$@" && return 0; sleep 1; done
    return 1
}

lt_installed() {   # true once the install of $LT_MODEL_ID is done
    lt_get "$KEY" "/local/models/install/$LT_MODEL_ID" | grep -q '"state":"done"'
}
