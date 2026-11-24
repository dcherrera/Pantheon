#!/bin/bash
# tests/local/unload_with_watch.sh <catalog_id> <profile_id>
# Run ON the box, from the repo root, against the RUNNING `pantheon local` on
# $LT_PORT with <catalog_id> loaded. The Argus Unload bug on hardware: an open
# GET /session/watch (what every open Argus chat holds) must not block an unload.
#   1. idle watch: unload succeeds fast, the watch gets `event: model_unloaded`
#      and is closed, the model leaves the loaded set and its dies are released;
#   2. live streaming turn + watch: the unload waits, then cancels the turn (the
#      stream ends with a model_unloaded error) and succeeds.
# Reloads the model after each arm, so the box is left as it was found. Uses its
# own paired admin + inference keys and revokes both at the end.
cd "$(dirname "$0")/../.." || exit 1
. tests/local/lib.sh
model=${1:?catalog id}; profile=${2:?profile id}
url=http://127.0.0.1:$LT_PORT
code=$(cat "$LT_DIR/state/pair_code") || exit 1
pair=$(curl -s -X POST "$url/local/pair" -d "{\"code\":\"$code\",\"label\":\"unload-watch-test\"}")
AK=$(sed -n 's/.*"key":"\([^"]*\)".*/\1/p' <<< "$pair")
AK_ID=$(sed -n 's/.*"id":"\([^"]*\)".*/\1/p' <<< "$pair")
inf=$(lt_post "$AK" /local/keys '{"role":"inference","label":"unload-watch-test"}')
IK=$(sed -n 's/.*"key":"\([^"]*\)".*/\1/p' <<< "$inf")
IK_ID=$(sed -n 's/.*"id":"\([^"]*\)".*/\1/p' <<< "$inf")
[ -n "$AK" ] && [ -n "$IK" ] || { echo "could not pair / mint keys: $pair $inf"; exit 1; }

is_loaded() { lt_get "$AK" /local/models/loaded | grep -q "\"catalog_id\":\"$model\"[^}]*\"state\":\"loaded\""; }
is_gone() { ! lt_get "$AK" /local/models/loaded | grep -q "\"catalog_id\":\"$model\""; }
load() {
    lt_post "$AK" /local/models/load "{\"catalog_id\":\"$model\",\"profile_id\":\"$profile\"}"; echo
    lt_wait 900 is_loaded
}
open_watch() {   # open_watch <session_id> <out file>: held in the background, as Argus does
    curl -sN -H "Authorization: Bearer $IK" "$url/session/watch?name=$model&session_id=$1" > "$2" &
    WATCH_PID=$!
    lt_wait 10 grep -q "watch established" "$2"
}
watch_closed() { ! kill -0 "$WATCH_PID" 2>/dev/null; }
unload_timed() {   # sets UNLOAD_OUT and UNLOAD_S
    local t0 t1
    t0=$(date +%s.%N)
    UNLOAD_OUT=$(lt_post "$AK" /admin/models/unload "{\"name\":\"$model\"}")
    t1=$(date +%s.%N)
    UNLOAD_S=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }')
    echo "  unload answered in ${UNLOAD_S}s: $UNLOAD_OUT"
}

lt_check "$model is loaded to begin with" lt_wait 900 is_loaded

echo "1. idle watch"
w1=$LT_DIR/unload_watch_1.out
lt_check "watch established (idle chat tab)" open_watch unload-hw-1-chat "$w1"
unload_timed
lt_check "unload with the watch open succeeds" grep -q '"unloaded"' <<< "$UNLOAD_OUT"
lt_check "and answers in under 5 s" awk -v s="$UNLOAD_S" 'BEGIN { exit !(s < 5) }'
lt_check "the watch was closed by the box" lt_wait 5 watch_closed
lt_check "the watch got event: model_unloaded" grep -q "event: model_unloaded" "$w1"
lt_check "the model left the loaded set" is_gone
lt_check "its dies were released" grep -q "registry: unloaded '$model' (die budget released)" \
    "$LT_DIR/serve.log"
lt_check "reloads" load

echo "2. live streaming turn + watch"
w2=$LT_DIR/unload_watch_2.out; s2=$LT_DIR/unload_stream_2.out
lt_check "watch established" open_watch unload-hw-2-chat "$w2"
curl -sN -H "Authorization: Bearer $IK" -H "X-Session-Id: unload-hw-2-chat" \
    "$url/v1/chat/completions" -d "{\"model\":\"$model\",\"stream\":true,\"max_tokens\":4000,
    \"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 2000, one number per line.\"}]}" \
    > "$s2" &
STREAM_PID=$!
lt_check "the turn is streaming" lt_wait 300 grep -q '"content"' "$s2"
unload_timed
lt_check "unload during a live turn succeeds" grep -q '"unloaded"' <<< "$UNLOAD_OUT"
lt_check "after waiting for the turn, then cancelling it (< 20 s)" \
    awk -v s="$UNLOAD_S" 'BEGIN { exit !(s < 20) }'
lt_check "the stream ended with a model_unloaded error" lt_wait 10 grep -q model_unloaded "$s2"
lt_check "the stream connection closed" lt_wait 10 sh -c "! kill -0 $STREAM_PID 2>/dev/null"
lt_check "the watch got event: model_unloaded" lt_wait 5 grep -q "event: model_unloaded" "$w2"
lt_check "the model left the loaded set" is_gone
lt_check "reloads (left loaded as found)" load

lt_post "$AK" /local/keys/revoke "{\"id\":\"$IK_ID\"}"; echo
lt_post "$AK" /local/keys/revoke "{\"id\":\"$AK_ID\"}"; echo
lt_done unload_with_watch
