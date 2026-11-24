#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# @file q8_e2e_bank.sh
# @brief K80 bank end to end for catalog v3: Qwen3.6 Q8_0 on two boards, and Q3_K_M with
#        both KV choices, through `pantheon local` on a test-signed catalog.
#
# @status Run 2026-10-28 on a K80 host (dies 0-5).
# @issues none
# @todo none
#
# Usage (on the bank, from a `make USE_CUDA=1 LOCAL_TEST_CATALOG_KEY=1` checkout, the
# real box stopped, CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5):
#   tests/local/q8_e2e_bank.sh <test-signed catalog.json> <Q8_0 gguf> <hf revision>
# The Q8_0 file is installed through an HF stand-in (fixture_httpd, the real repo path
# and revision -> the local file), so the install's size + sha256 check runs; the Q3_K_M
# file must already be in LT_STORE. Steps: pair; v3 catalog; the Q8 profile fits with its
# options; install Q8; load it with no choice (recommended, 4 units = two whole boards);
# chat; unload; Q3 `k80-pair` (Q4 KV) at 16k x 2 (not its default) -> chat -> unload;
# Q3 `k80-pair-fp32` at 16k x 1 -> the serve logs fp32 KV, never "forcing Q4" -> chat ->
# unload. Leaves nothing loaded.
set -u
cd "$(dirname "$0")/../.." || exit 1
catalog=${1:?test-signed catalog.json}; q8=${2:?Q8_0 gguf}; rev=${3:?hf revision}
export LT_DIR=${LT_DIR:-build/local_q8e2e} LT_STORE=${LT_STORE:-build/local_hw/store}
. tests/local/lib.sh
OUT=$LT_DIR/out; mkdir -p "$OUT"
Q8=qwen3.6-35b-a3b-q8; Q3=qwen3.6-35b-a3b-q3km
REPO=TeamIDE/Qwen3.6-35B-A3B-Pantheon-K80

has() { grep -q -- "$2" "$1"; }
status_of() { sed -n 's/^HTTP\/1.1 \([0-9]*\).*/\1/p' "$1" | tail -1; }
load() {   # load <out> <json body>
    curl -s -i -m 1800 -X POST -H "Authorization: Bearer $KEY" \
        "http://127.0.0.1:$LT_PORT/local/models/load" -d "$2" > "$OUT/$1"
}
unload() { lt_post "$KEY" /admin/models/unload "{\"name\":\"$1\"}" > "$OUT/$2"; }
chat() {   # chat <name> <out>
    curl -s -m 900 -H "Authorization: Bearer $KEY" \
        "http://127.0.0.1:$LT_PORT/v1/chat/completions" \
        -d "{\"model\":\"$1\",\"max_tokens\":200,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Name three primary colours, one word each.\"}]}" \
        > "$OUT/$2"
}
ok200() { [ "$(status_of "$OUT/$1")" = 200 ]; }
two_boards() { grep -Eq '"units":\[(0,1,2,3|0,1,4,5|2,3,4,5)\]' "$OUT/$1"; }
installed() {   # the install finished, or the file was already in the store
    lt_get "$KEY" "/local/models/install/$Q8" | grep -q '"state":"done"' ||
        grep -q '"state":"done"' "$OUT/install.json"
}

# HF stand-in: the real repo path + revision -> the local Q8_0 file.
www=$LT_DIR/www/$REPO/resolve/$rev
mkdir -p "$www"
ln -sfn "$(readlink -f "$q8")" "$www/$(basename "$q8")"
[ -x build/fixture_httpd ] || make -s build/fixture_httpd || exit 1
setsid sh -c 'echo $$ > "$0"; exec "$@"' "$LT_DIR/hf.pid" \
    build/fixture_httpd "$LT_HF_PORT" "$LT_DIR/www" > "$LT_DIR/hf.log" 2>&1 < /dev/null &
sleep 0.5

lt_start --no-mdns --allow-http --catalog-url "file://$PWD/$catalog" \
    --hf-base-url "http://127.0.0.1:$LT_HF_PORT" || exit 1
KEY=$(lt_pair q8e2e)
lt_check "paired" test -n "$KEY"
lt_wait 60 sh -c "curl -s -H 'Authorization: Bearer $KEY' http://127.0.0.1:$LT_PORT/local/catalog | grep -q '\"version\":3'"
lt_get "$KEY" /local/catalog > "$OUT/catalog.json"
lt_check "catalog v3 accepted" has "$OUT/catalog.json" '"version":3'
lt_check "Q8 profile on two whole boards" has "$OUT/catalog.json" '"id":"k80-two-boards","rank":1,"hardware":{"unit_type":"k80-gpu","count":4,"same_card":false,"per_card":2'
lt_check "Q3 Q4 profile listed" has "$OUT/catalog.json" '"id":"k80-pair"'
lt_check "Q3 fp32 profile listed" has "$OUT/catalog.json" '"id":"k80-pair-fp32"'

lt_post "$KEY" /local/models/install "{\"catalog_id\":\"$Q8\"}" > "$OUT/install.json"
lt_check "Q8 installed (size + sha256 verified)" lt_wait 3600 installed

load q8_load.txt "{\"catalog_id\":\"$Q8\",\"profile_id\":\"k80-two-boards\"}"
cat "$OUT/q8_load.txt" | tail -1; echo
lt_check "Q8 loads" ok200 q8_load.txt
lt_check "on 4 units, two per board" two_boards q8_load.txt
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "$OUT/q8_smi_loaded.txt"
chat "$Q8" q8_chat.json
lt_check "Q8 chat answers" has "$OUT/q8_chat.json" '"choices"'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "$OUT/q8_smi_chat.txt"
unload "$Q8" q8_unload.json
lt_check "Q8 unloaded" has "$OUT/q8_unload.json" '"unloaded"'

load q3q4_load.txt "{\"catalog_id\":\"$Q3\",\"profile_id\":\"k80-pair\",\"ctx_cap\":16384,\"max_sessions\":2}"
lt_check "Q3 Q4 KV at 16k x 2 (not its default) loads" ok200 q3q4_load.txt
lt_check "the serve made a Q4 cache (the load's choice)" has "$LT_DIR/serve.log" "KV cache Q4 (the load's choice)"
chat "$Q3" q3q4_chat.json
lt_check "Q3 Q4 chat answers" has "$OUT/q3q4_chat.json" '"choices"'
unload "$Q3" q3q4_unload.json
lt_check "Q3 Q4 unloaded" has "$OUT/q3q4_unload.json" '"unloaded"'

before=$(lt_count "$LT_DIR/serve.log" "forcing Q4")
load q3fp_load.txt "{\"catalog_id\":\"$Q3\",\"profile_id\":\"k80-pair-fp32\",\"ctx_cap\":16384,\"max_sessions\":1}"
cat "$OUT/q3fp_load.txt" | tail -1; echo
lt_check "Q3 fp32 KV at 16k x 1 loads" ok200 q3fp_load.txt
lt_check "the serve made an fp32 cache (the load's choice)" has "$LT_DIR/serve.log" "KV cache fp32 (the load's choice)"
lt_check "the ladder did not switch it to Q4" test "$(lt_count "$LT_DIR/serve.log" "forcing Q4")" = "$before"
chat "$Q3" q3fp_chat.json
lt_check "Q3 fp32 chat answers" has "$OUT/q3fp_chat.json" '"choices"'
unload "$Q3" q3fp_unload.json
lt_check "Q3 fp32 unloaded" has "$OUT/q3fp_unload.json" '"unloaded"'

lt_get "$KEY" /local/models/loaded > "$OUT/loaded_end.json"
lt_check "nothing loaded at the end" has "$OUT/loaded_end.json" '"models":\[\]'
lt_stop
kill "$(cat "$LT_DIR/hf.pid" 2>/dev/null)" 2>/dev/null
lt_done q8_e2e_bank
