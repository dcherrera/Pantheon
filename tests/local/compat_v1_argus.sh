#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# tests/local/compat_v1_argus.sh — people registry P1.6, on the Mac (CPU).
# A box paired before the people registry (keys.json v1) is upgraded to this
# checkout's HEAD, and an Argus from before C6 keeps working with the keys it
# stored, with no step: it connects, lists, loads and chats.
#
#   tests/local/compat_v1_argus.sh [argus-repo] [foundry] [old-ref] [argus-ref]
#     argus-repo  an Argus clone (default ../Argus)
#     foundry     a foundry main checkout (default ../foundry)
#     old-ref     the Pantheon ref before the people registry (default origin/main)
#     argus-ref   an Argus ref before C6 (default origin/main)
#
# 1. Builds old-ref and HEAD in detached worktrees under build/compat_v1/, both
#    with LOCAL_TEST_CATALOG_KEY=1 (TEST ONLY: trusts the e2e fixture catalog),
#    and argus-ref (argus + local_probe) the same way.
# 2. Old box: Argus's local_probe (the TUI's /local path) pairs and loads
#    qwen3-0.6b-q8 -> a keys.json v1 and Argus's key store in build/compat_v1/home.
# 3. Control: the new box on an EMPTY state dir; the same Argus steps must fail
#    (proves the checks can fail).
# 4. Stops the old box, copies its state dir, starts the new box on the copy
#    (same port and store). Argus, same key store: unload, load, headless chat.
# Every box runs under foundry metal/run_guarded.sh. Exit status = failures.
# Output under build/compat_v1/ (run.txt when run detached); the worktrees go at the end.
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
ARGUS_REPO=$(cd "${1:-../Argus}" && pwd) || exit 1
FOUNDRY=$(cd "${2:-../foundry}" && pwd) || exit 1
OLD_REF=${3:-origin/main}
ARGUS_REF=${4:-origin/main}
OUT=$ROOT/build/compat_v1
ARGUS=$OUT/argus-src
PORT=${COMPAT_PORT:-18431}
HF_PORT=${COMPAT_HF_PORT:-18432}
MODEL_FILE=${PN_TEST_MODEL:-models/qwen3-0.6b-Q8_0.gguf}
MODEL=qwen3-0.6b-q8
REV=0123456789abcdef0123456789abcdef01234567
FAILS=0
export HOME=$OUT/home ARGUS_PROBE_LOG=$OUT/probe.log
export ARGUS_LOCAL_MDNS=127.0.0.1:9 GUARD_MIN_FREE_GB=${GUARD_MIN_FREE_GB:-2}

check() { local what=$1; shift; if "$@"; then echo "  ok   $what"; else echo "  FAIL $what"; FAILS=$((FAILS + 1)); fi; }
not() { ! "$@"; }
probe() { "$ARGUS/build/tools/local/local_probe" "$@" >> "$OUT/notes.txt" 2>&1; }
store_field() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['boxes'][0][sys.argv[2]])" \
                "$HOME/.argus/local_keys.json" "$1"; }

# Only ever signal a pid > 0 (a kill(-1) elsewhere once killed every process).
stop_pid() {
    local pid=$1
    [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null || return 0
    kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.5; done
    kill -KILL "$pid" 2>/dev/null
}

BOX_PID=0
start_box() {   # start_box <binary> <state dir> <log>
    "$FOUNDRY/metal/run_guarded.sh" "$1" local --host 127.0.0.1 --port "$PORT" --state-dir "$2" \
        --store "$OUT/store" --no-mdns --allow-http \
        --catalog-url "file://$ROOT/tests/local/fixtures/catalog_e2e.json" \
        --hf-base-url "http://127.0.0.1:$HF_PORT" > "$3" 2>&1 < /dev/null &
    BOX_PID=$!
    for _ in $(seq 1 120); do
        curl -s -o /dev/null "http://127.0.0.1:$PORT/" && return 0
        sleep 0.5
    done
    echo "box did not come up; log: $3"; return 1
}
stop_box() { stop_pid "$BOX_PID"; BOX_PID=0; }

build_ref() {   # build_ref <ref> <dir> (a kept tree at the same commit builds incrementally)
    if [ "$(git -C "$2" rev-parse HEAD 2>/dev/null)" != "$(git rev-parse "$1")" ]; then
        git worktree remove --force "$2" > /dev/null 2>&1
        git worktree add --detach "$2" "$1" > /dev/null 2>&1 || return 1
    fi
    make -C "$2" -j8 USE_BLAS=1 USE_THREADS=1 LOCAL_TEST_CATALOG_KEY=1 FOUNDRY="$FOUNDRY" pantheon \
        > "$2.build.log" 2>&1
}

build_argus() {
    if [ "$(git -C "$ARGUS" rev-parse HEAD 2>/dev/null)" != "$(git -C "$ARGUS_REPO" rev-parse "$ARGUS_REF")" ]; then
        git -C "$ARGUS_REPO" worktree remove --force "$ARGUS" > /dev/null 2>&1
        git -C "$ARGUS_REPO" worktree add --detach "$ARGUS" "$ARGUS_REF" > /dev/null 2>&1 || return 1
    fi
    make -C "$ARGUS" -j8 all local-probe > "$OUT/argus.build.log" 2>&1
}

chat() {   # argus --headless from a workspace: no --base, so the paired box is used.
    # Argus sends ~4k prompt tokens: ~260-290 s of CPU prefill for the 0.6B on an M4.
    mkdir -p "$OUT/work" && cd "$OUT/work" || return 1
    ARGUS_MAX_TOKENS=48 ARGUS_ENABLE_THINKING=0 perl -e 'alarm 900; exec @ARGV' \
        "$ARGUS/build/argus" --headless -m chat -p "Reply with one word: ok." > "$OUT/$1" 2>&1
    local rc=$?
    echo "    $1: exit $rc, $(tail -1 "$OUT/$1")"
    cd "$ROOT" || return 1
    return $rc
}

HF_PID=0
remove_worktrees() {   # COMPAT_KEEP=1 keeps the three source trees and binaries
    [ -n "${COMPAT_KEEP:-}" ] && return 0
    git worktree remove --force "$OUT/src-old" > /dev/null 2>&1
    git worktree remove --force "$OUT/src-new" > /dev/null 2>&1
    git -C "$ARGUS_REPO" worktree remove --force "$ARGUS" > /dev/null 2>&1
}
cleanup() { stop_box; stop_pid "$HF_PID"; remove_worktrees; }
trap cleanup EXIT INT TERM

[ -f "$MODEL_FILE" ] || { echo "compat_v1: $MODEL_FILE not found (set PN_TEST_MODEL)"; exit 1; }
rm -rf "$OUT/home" "$OUT/v1" "$OUT/v2" "$OUT/empty" "$OUT/work" "$OUT/store" "$OUT/www" \
       "$OUT"/*.txt "$OUT"/*.log
mkdir -p "$HOME" "$OUT/store" "$OUT/www/teamide/test-fixtures/resolve/$REV"
ln -s "$(cd "$(dirname "$MODEL_FILE")" && pwd)/$(basename "$MODEL_FILE")" \
      "$OUT/www/teamide/test-fixtures/resolve/$REV/qwen3-0.6b-Q8_0.gguf"

echo "--- build: $OLD_REF (before people) and HEAD, test catalog key; Argus pre-C6"
check "old build ($OLD_REF $(git rev-parse --short "$OLD_REF"))" build_ref "$OLD_REF" "$OUT/src-old"
check "new build (HEAD $(git rev-parse --short HEAD))" build_ref HEAD "$OUT/src-new"
check "Argus build ($ARGUS_REF $(git -C "$ARGUS_REPO" rev-parse --short "$ARGUS_REF"))" build_argus
check "Argus has no C6 (/local/me/device unused)" not grep -rq 'local/me/device' "$ARGUS/src"
make -s FOUNDRY="$FOUNDRY" build/fixture_httpd > /dev/null || exit 1
build/fixture_httpd "$HF_PORT" "$OUT/www" > "$OUT/httpd.log" 2>&1 &
HF_PID=$!

echo "--- old box: Argus pairs and loads (keys.json v1)"
start_box "$OUT/src-old/pantheon" "$OUT/v1/state" "$OUT/serve_old.log" || exit 1
check "paired" probe pair "127.0.0.1:$PORT" "$(cat "$OUT/v1/state/pair_code")"
LABEL=$(store_field label)
check "loaded $MODEL on the old box" probe load "$LABEL" "$MODEL"
check "the old box wrote keys.json v1" grep -q '"version":1' "$OUT/v1/state/keys.json"
check "chat on the old box (baseline)" chat chat_old.txt
stop_box
ADMIN=$(store_field admin_key)
INF=$(store_field inference_key)

echo "--- control: the new box on an empty state dir must refuse the stored keys"
mkdir -p "$OUT/empty/state"
start_box "$OUT/src-new/pantheon" "$OUT/empty/state" "$OUT/serve_empty.log" || exit 1
check "control: load refused" not probe load "$LABEL" "$MODEL"
check "control: chat fails" not chat chat_control.txt
stop_box

echo "--- upgrade: copy the v1 state dir, start HEAD on the copy"
cp -R "$OUT/v1" "$OUT/v2"
start_box "$OUT/src-new/pantheon" "$OUT/v2/state" "$OUT/serve_new.log" || exit 1
S=$OUT/v2/state
check "keys.json migrated to v2" grep -q '"version":2' "$S/keys.json"
check "keys.json.v1 kept, 0600" [ "$(stat -f %Lp "$S/keys.json.v1")" = 600 ]
check "person 1 is owner, admin" grep -q '"handle":"owner","name":"Owner","admin":true' "$S/people.json"
check "the model came back by itself" sh -c "for i in \$(seq 1 120); do curl -s -H 'Authorization: \
    Bearer $INF' http://127.0.0.1:$PORT/v1/models | grep -q '\"id\":\"$MODEL\"' && exit 0; sleep 0.5; done; exit 1"

echo "--- Argus with its stored keys, no step: lists, loads, chats"
code=$(curl -s -o "$OUT/keys_list.json" -w '%{http_code}' -H "Authorization: Bearer $ADMIN" \
       "http://127.0.0.1:$PORT/local/keys")
check "stored admin key lists keys (old roles kept)" sh -c "[ $code = 200 ] && grep -q \
    '\"role\":\"admin\"' '$OUT/keys_list.json' && grep -q '\"role\":\"inference\"' '$OUT/keys_list.json'"
check "unload (stored admin key)" probe unload "$LABEL" "$MODEL"
check "load again (lists catalog + loaded, then loads)" probe load "$LABEL" "$MODEL"
check "chat with the stored inference key (argus --headless)" chat chat_new.txt
check "the inference key still cannot reach admin routes" [ "$(curl -s -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer $INF" "http://127.0.0.1:$PORT/local/hardware")" = 403 ]
check "the admin key is a device of owner (/local/me)" sh -c "curl -s -H 'Authorization: Bearer $ADMIN' \
    http://127.0.0.1:$PORT/local/me | grep -q '\"handle\":\"owner\"'"
stop_box
check "no key in any log" sh -c "! grep -rq 'pnk_' '$OUT'/serve_*.log '$OUT/probe.log' '$OUT/work' \
    '$OUT'/chat_*.txt '$OUT/notes.txt'"
echo "compat_v1_argus: $FAILS failure(s)"
exit "$FAILS"
