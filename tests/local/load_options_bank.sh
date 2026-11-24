#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# @file load_options_bank.sh
# @brief L5 on hardware: smart load options and replicas through `pantheon local`.
#
# Usage (on the card host, from the box checkout, the box itself stopped; a
# `make USE_CUDA=1|USE_TT=1 LOCAL_TEST_CATALOG_KEY=1` build):
#   tests/local/load_options_bank.sh k80  <test-signed catalog.json>
#   tests/local/load_options_bank.sh p150 <test-signed catalog.json>
# The catalog is catalog/catalog.json signed with the TEST key; its files must
# already be in the store (LT_STORE, default build/local_hw/store), so nothing is
# downloaded. State goes to build/local_ldopt (its own keys and loaded set).
#
# k80 (dies 0-3 visible: CUDA_VISIBLE_DEVICES): the catalog offers only fitting
# choices; off-grid and too-big choices are refused; 8k x 2 users loads and two
# sessions chat, device memory stays inside the memory model; the recommended
# 32k x 1 loads on dies 0,1, a second copy goes to dies 2,3 under the same name,
# two new sessions land one on each copy; a restart brings both copies back;
# unload one, then all.
# p150: 8k..128k for one user; Ornith at 32k loads, the card's DRAM matches the
# model, a chat answers, unload.
set -u
cd "$(dirname "$0")/../.." || exit 1
kind=${1:?k80|p150}; catalog=${2:?test-signed catalog.json}
export LT_DIR=${LT_DIR:-build/local_ldopt} LT_STORE=${LT_STORE:-build/local_hw/store}
. tests/local/lib.sh
OUT=$LT_DIR/out; mkdir -p "$OUT"
FLAGS=(--no-mdns --allow-http --catalog-url "file://$PWD/$catalog")
QWEN=qwen3.6-35b-a3b-q3km; ORN=ornith-1.5-9b-q8

has() { grep -q -- "$2" "$1"; }
status_of() { sed -n 's/^HTTP\/1.1 \([0-9]*\).*/\1/p' "$1" | tail -1; }

load() {   # load <out> <json body>: POST /local/models/load, response with headers in <out>
    curl -s -i -m 1800 -X POST -H "Authorization: Bearer $KEY" \
        "http://127.0.0.1:$LT_PORT/local/models/load" -d "$2" > "$OUT/$1"
}

unload() { lt_post "$KEY" /admin/models/unload "$1" > "$OUT/$2"; }

chat() {   # chat <name> <session> <out>
    curl -s -m 600 -H "Authorization: Bearer $KEY" -H "X-Session-Id: $2" \
        "http://127.0.0.1:$LT_PORT/v1/chat/completions" \
        -d "{\"model\":\"$1\",\"max_tokens\":24,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Name three colours.\"}]}" \
        > "$OUT/$3"
}

loaded_has() { lt_get "$KEY" /local/models/loaded > "$OUT/loaded.json"; has "$OUT/loaded.json" "$1"; }

die_peak_mib() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
                 awk -F', *' -v d="$1" '$1 == d { print $2 }'; }

start_box() {
    lt_start "${FLAGS[@]}" || exit 1
    KEY=$(lt_pair)
    [ -n "$KEY" ] || KEY=$(cat "$OUT/key" 2>/dev/null)
    echo "$KEY" > "$OUT/key"; chmod 600 "$OUT/key"
    lt_wait 60 sh -c "curl -s -H 'Authorization: Bearer $KEY' http://127.0.0.1:$LT_PORT/local/catalog | grep -q '\"version\":2'"
}

run_k80() {
    lt_get "$KEY" /local/catalog > "$OUT/catalog.json"
    lt_get "$KEY" /local/hardware > "$OUT/hardware.json"
    lt_check "catalog: 32k x 1 fits and is recommended" \
        has "$OUT/catalog.json" '"ctx_cap":32768,"max_sessions":1,"need_per_unit":12179210240,"recommended":true'
    lt_check "catalog: 8k x 2 and 16k x 2 offered" sh -c \
        "grep -q '\"ctx_cap\":8192,\"max_sessions\":2' $OUT/catalog.json && grep -q '\"ctx_cap\":16384,\"max_sessions\":2' $OUT/catalog.json"
    lt_check "catalog: 32k x 2 is not offered (it does not fit)" sh -c "! grep -q '\"ctx_cap\":32768,\"max_sessions\":2' $OUT/catalog.json"

    load refuse_ctx.txt "{\"catalog_id\":\"$QWEN\",\"profile_id\":\"k80-pair\",\"ctx_cap\":20000}"
    lt_check "off-grid context refused (409, offered ones named)" \
        sh -c "[ \"$(status_of "$OUT/refuse_ctx.txt")\" = 409 ] && grep -q 'not an offered context (8192, 16384, 32768)' $OUT/refuse_ctx.txt"
    load refuse_big.txt "{\"catalog_id\":\"$QWEN\",\"profile_id\":\"k80-pair\",\"ctx_cap\":32768,\"max_sessions\":2}"
    lt_check "32k x 2 refused with the memory it needs" \
        sh -c "[ \"$(status_of "$OUT/refuse_big.txt")\" = 409 ] && grep -q 'needs .* GB on each k80-gpu' $OUT/refuse_big.txt"

    load a.txt "{\"catalog_id\":\"$QWEN\",\"profile_id\":\"k80-pair\",\"ctx_cap\":8192,\"max_sessions\":2,\"units\":[0,1]}"
    lt_check "8k x 2 users loads on dies 0,1" sh -c "[ \"$(status_of "$OUT/a.txt")\" = 200 ] && grep -q '\"ctx_cap\":8192,\"max_sessions\":2' $OUT/a.txt"
    chat "$QWEN" two-a chat_a1.json & local c1=$!
    chat "$QWEN" two-b chat_a2.json; wait "$c1"   # never a bare wait: the box is our child
    lt_check "two users chat at once" sh -c "grep -q choices $OUT/chat_a1.json && grep -q choices $OUT/chat_a2.json"
    local p1; p1=$(die_peak_mib 1); echo "  die 1 now ${p1} MiB (8k x 2 model: 10850 MiB peak)"
    lt_check "die 1 within the memory model" [ "${p1:-99999}" -le 10850 ]
    unload "{\"name\":\"$QWEN\"}" unload_a.json

    load b.txt "{\"catalog_id\":\"$QWEN\",\"profile_id\":\"k80-pair\",\"units\":[0,1]}"
    lt_check "no choice: the recommended 32k x 1 on dies 0,1" sh -c "grep -q '\"replica\":1,\"ctx_cap\":32768,\"max_sessions\":1' $OUT/b.txt"
    lt_get "$KEY" /local/catalog > "$OUT/catalog_one_loaded.json"
    load c.txt "{\"catalog_id\":\"$QWEN\",\"profile_id\":\"k80-pair\"}"
    lt_check "another copy: replica 2 on dies 2,3" sh -c "grep -q '\"replica\":2' $OUT/c.txt && grep -q '\"units\":\[2,3\]' $OUT/c.txt"
    lt_get "$KEY" /local/hardware > "$OUT/hardware_two_copies.json"
    chat "$QWEN" copy-s1 chat_c1.json & local c2=$!
    sleep 2; chat "$QWEN" copy-s2 chat_c2.json; wait "$c2"
    lt_check "both sessions answered through one name" sh -c "grep -q choices $OUT/chat_c1.json && grep -q choices $OUT/chat_c2.json"
    lt_check "the two sessions went to different copies" sh -c \
        "grep -q \"session copy-s1 -> '$QWEN' replica on die 0\" $LT_DIR/serve.log && grep -q \"session copy-s2 -> '$QWEN' replica on die 2\" $LT_DIR/serve.log"
    chat "$QWEN" copy-s1 chat_c3.json
    lt_check "a session's next turn stays on its copy (no new binding)" \
        [ "$(grep -c "session copy-s1 ->" "$LT_DIR/serve.log")" = 1 ]

    lt_stop; start_box
    lt_wait 1200 sh -c "curl -s -H 'Authorization: Bearer $KEY' http://127.0.0.1:$LT_PORT/local/models/loaded | grep -o '\"state\":\"loaded\"' | wc -l | grep -q 2"
    loaded_has '"replica":2'
    lt_check "restart: both copies reloaded (32k x 1 each)" sh -c \
        "[ \$(grep -o '\"state\":\"loaded\"' $OUT/loaded.json | wc -l) = 2 ] && grep -q '\"replica\":2,\"ctx_cap\":32768' $OUT/loaded.json"
    lt_get "$KEY" /local/models/loaded > "$OUT/loaded_after_restart.json"

    unload "{\"name\":\"$QWEN\",\"replica\":2}" unload_one.json
    lt_check "unload copy 2: copy 1 still loaded" sh -c \
        "grep -q '\"replicas\":1' $OUT/unload_one.json && curl -s -H 'Authorization: Bearer $KEY' http://127.0.0.1:$LT_PORT/local/models/loaded | grep -q '\"replica\":1' && ! curl -s -H 'Authorization: Bearer $KEY' http://127.0.0.1:$LT_PORT/local/models/loaded | grep -q '\"replica\":2'"
    chat "$QWEN" after-one chat_d1.json
    lt_check "the name still answers" grep -q choices "$OUT/chat_d1.json"
    load e.txt "{\"catalog_id\":\"$QWEN\",\"profile_id\":\"k80-pair\"}"
    unload "{\"name\":\"$QWEN\"}" unload_all.json
    lt_check "unload all: both copies down" sh -c \
        "grep -q '\"replicas\":2' $OUT/unload_all.json && curl -s -H 'Authorization: Bearer $KEY' http://127.0.0.1:$LT_PORT/local/models/loaded | grep -q '\"models\":\[\]'"
}

run_p150() {
    lt_get "$KEY" /local/catalog > "$OUT/catalog.json"
    lt_get "$KEY" /local/hardware > "$OUT/hardware.json"
    lt_check "catalog: 8k..128k for one user, 128k recommended" sh -c \
        "grep -q '\"ctx_cap\":8192,\"max_sessions\":1' $OUT/catalog.json && grep -q '\"ctx_cap\":131072,\"max_sessions\":1,\"need_per_unit\":16771578855,\"recommended\":true' $OUT/catalog.json"
    load refuse_users.txt "{\"catalog_id\":\"$ORN\",\"profile_id\":\"p150-one\",\"max_sessions\":2}"
    lt_check "2 users refused (one user on a p150)" sh -c \
        "[ \"$(status_of "$OUT/refuse_users.txt")\" = 409 ] && grep -q 'not an offered number of users (1)' $OUT/refuse_users.txt"
    load a.txt "{\"catalog_id\":\"$ORN\",\"profile_id\":\"p150-one\",\"ctx_cap\":32768}"
    lt_check "Ornith at 32k loads" sh -c "grep -q '\"ctx_cap\":32768,\"max_sessions\":1' $OUT/a.txt"
    lt_check "card DRAM = the model's 12.245 GiB for 32k" grep -q "DRAM after arm — 12.24[45] GiB" "$LT_DIR/serve.log"
    chat "$ORN" orn-1 chat_a1.json
    lt_check "a chat answers" grep -q choices "$OUT/chat_a1.json"
    unload "{\"name\":\"$ORN\"}" unload_a.json
    lt_check "unloaded" grep -q '"unloaded"' "$OUT/unload_a.json"
}

rm -rf "$LT_DIR/state"
start_box
case "$kind" in
    k80) run_k80 ;;
    p150) run_p150 ;;
esac
lt_stop
lt_done "load_options_bank $kind"
