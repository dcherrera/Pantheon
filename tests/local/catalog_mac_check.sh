#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# tests/local/catalog_mac_check.sh — the real catalog (catalog/catalog.json) through a
# Mac `pantheon local`, no card and no bank: sign a COPY with the TEST ONLY key, start the
# box with that catalog, pair, and check /local/catalog lists both models with the right
# refusal reasons (the Mac has no K80 / p150). Install and load are not run here: they need
# the files on the HF org (H3 runs them on the cards).
#
#   make LOCAL_TEST_CATALOG_KEY=1 pantheon && tests/local/catalog_mac_check.sh
#   (then `make pantheon` again for a production binary)
#
# Writes only build/catalog_check/. Prints PASS/FAIL lines and "DONE pass=N fail=N".
set -u
cd "$(dirname "$0")/../.." || exit 1
out=$PWD/build/catalog_check
port=${CATALOG_CHECK_PORT:-18731}
rm -rf "$out"; mkdir -p "$out/state" "$out/store"
pass=0; fail=0
check() { if "${@:2}"; then echo "PASS $1"; pass=$((pass + 1)); else echo "FAIL $1"; fail=$((fail + 1)); fi; }

cp catalog/catalog.json "$out/catalog.json"
minisign -S -s tests/local/fixtures/test_catalog.key -m "$out/catalog.json" \
    -x "$out/catalog.json.minisig" -t "TEST ONLY catalog copy" > /dev/null || exit 1
./pantheon local --port "$port" --state-dir "$out/state" --store "$out/store" --no-mdns \
    --allow-http --catalog-url "file://$out/catalog.json" > "$out/box.log" 2>&1 &
box=$!
trap 'kill $box 2>/dev/null; wait $box 2>/dev/null' EXIT
for _ in $(seq 1 50); do [ -s "$out/state/pair_code" ] && break; sleep 0.2; done
code=$(./pantheon local pair-code --state-dir "$out/state" | tr -dc '0-9' | head -c 8)
key=$(curl -s -X POST "http://127.0.0.1:$port/local/pair" -d "{\"code\":\"$code\",\"label\":\"check\"}" |
      sed -n 's/.*"key":"\([^"]*\)".*/\1/p')
check "paired" [ -n "$key" ]
cat_json=""
for _ in $(seq 1 50); do
    cat_json=$(curl -s -H "Authorization: Bearer $key" "http://127.0.0.1:$port/local/catalog")
    echo "$cat_json" | grep -q '"version":1' && break; sleep 0.2
done
echo "$cat_json" > "$out/local_catalog.json"
check "catalog version 1 accepted" grep -q '"version":1' "$out/local_catalog.json"
check "two entries" [ "$(grep -o '"profile":{' "$out/local_catalog.json" | wc -l | tr -d ' ')" = 2 ]
check "qwen3.6 k80-pair listed" grep -q '"id":"qwen3.6-35b-a3b-q3km"' "$out/local_catalog.json"
check "ornith p150-one listed" grep -q '"id":"ornith-1.5-9b-q8"' "$out/local_catalog.json"
check "K80 refusal reason" grep -q 'needs 2 k80-gpu' "$out/local_catalog.json"
check "p150 refusal reason" grep -q 'needs 1 p150' "$out/local_catalog.json"
check "nothing fits on a Mac" bash -c "! grep -q '\"fits\":true' '$out/local_catalog.json'"
echo "DONE pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
