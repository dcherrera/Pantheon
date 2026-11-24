#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# @file backlog_bank.sh
# @brief K80 check of the Pantheon Local backlog (H2m, H2o, fast reads, H2n).
#   Run ON the bank from a test checkout (not the box's), with a
#   `make USE_CUDA=1 LOCAL_TEST_CATALOG_KEY=1` build, on two free GPUs of one board:
#     BACKLOG_DIES=6,7 LT_MODEL_FILE=<qwen3-0.6b-Q8_0.gguf> Q3_STORE=<store holding
#       the Q3 file> tests/local/backlog_bank.sh [small|q3|all]
#   small: fixture catalog (TEST ONLY key) on :8431 — install the small model,
#          uninstall refused while loaded, uninstall, catalog says not installed,
#          install again; /local/hardware answers in ms with a fresh snapshot age.
#   q3:    a plain `pantheon local` (no --catalog-url: the built-in catalog) on
#          :8433 over Q3_STORE (read only) — load Qwen3.6 Q3 k80-pair while
#          polling /local/models/loaded 2/s (progress lines), then unload the same way.
#   idle:  the same box with nothing loaded for 60 s: CPU seconds the background
#          device memory reads cost (utime + stime of the process).
#
# @status None
# @issues None
# @todo None

cd "$(dirname "$0")/../.." || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${BACKLOG_DIES:?free dies, e.g. 6,7}
WHAT=${1:-all}
. tests/local/lib.sh

hw_timing() {   # 10 x GET /local/hardware: wall time and the snapshot age
    for _ in $(seq 1 10); do
        curl -s -o "$LT_DIR/hw.json" -w '%{time_total}' -H "Authorization: Bearer $KEY" \
            "http://127.0.0.1:$LT_PORT/local/hardware"
        printf ' s  age %s ms\n' "$(sed -n 's/.*"mem_age_ms":\([0-9]*\),"mem_poll_ms".*/\1/p' \
            "$LT_DIR/hw.json")"
        sleep 0.3
    done
}

host_ram_ok() {   # MemAvailable >= 4.5 GB: room for a K80 upload's ~2.4 GB transient
    [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -ge 4500000 ]
}

run_small() {
    LT_PORT=8431 LT_HF_PORT=8432 LT_DIR=build/lbk_small LT_STORE=build/lbk_small/store
    echo "== small: uninstall / reinstall (H2m) + fast /local/hardware"
    rm -rf "$LT_DIR"
    # A `cpu` profile needs a box without cards (H2f): the small model goes on a K80
    # GPU, and llama-arch models need the single-user path there (spec §13).
    export QWEN35_BATCH_SLOTS=0
    lt_hf_start || return 1
    # shellcheck disable=SC2046
    lt_start --no-mdns $(lt_catalog_flags) || { lt_hf_stop; return 1; }
    KEY=$(lt_pair lbk)
    lt_check "catalog accepted" lt_wait 30 sh -c "curl -s -H 'Authorization: Bearer $KEY' \
        http://127.0.0.1:$LT_PORT/local/catalog | grep -q '\"version\":1'"
    lt_post "$KEY" /local/models/install "{\"catalog_id\":\"$LT_MODEL_ID\"}"; echo
    lt_check "installed" lt_wait 300 lt_installed
    lt_check "catalog: installed" sh -c "curl -s -H 'Authorization: Bearer $KEY' \
        http://127.0.0.1:$LT_PORT/local/catalog | grep -q '\"installed\":true'"
    ls -la "$LT_STORE"
    lt_post "$KEY" /local/models/load "{\"catalog_id\":\"$LT_MODEL_ID\",\"profile_id\":\"k80-one\"}"
    echo
    out=$(lt_post "$KEY" /local/models/uninstall "{\"catalog_id\":\"$LT_MODEL_ID\"}"); echo "$out"
    lt_check "refused while loaded" sh -c "echo '$out' | grep -q 'is loaded on'"
    lt_post "$KEY" /admin/models/unload "{\"name\":\"$LT_MODEL_ID\"}"; echo
    out=$(lt_post "$KEY" /local/models/uninstall "{\"catalog_id\":\"$LT_MODEL_ID\"}"); echo "$out"
    lt_check "uninstalled" sh -c "echo '$out' | grep -q '\"removed_files\"'"
    lt_check "store empty" [ -z "$(ls "$LT_STORE")" ]
    lt_check "catalog: not installed" sh -c "curl -s -H 'Authorization: Bearer $KEY' \
        http://127.0.0.1:$LT_PORT/local/catalog | grep -q '\"installed\":false'"
    lt_post "$KEY" /local/models/install "{\"catalog_id\":\"$LT_MODEL_ID\"}"; echo
    lt_check "installed again" lt_wait 300 lt_installed
    ls -la "$LT_STORE"
    echo "-- /local/hardware x10 (wall s, snapshot age)"
    hw_timing
    lt_stop
    lt_hf_stop
}

poll_progress() {   # poll_progress <seconds> — 2/s: state + progress of the loaded rows
    for _ in $(seq 1 $(( $1 * 2 ))); do
        printf '%s ' "$(date +%T.%N | cut -c1-11)"
        lt_get "$KEY" /local/models/loaded | python3 -c '
import json, sys
d = json.load(sys.stdin)
rows = []
for m in d["models"]:
    p = m.get("progress")
    if p:
        units = " ".join("u%d %.2f/%.2f" % (u["id"], u["bytes"] / 1e9, u["total"] / 1e9) for u in p["units"])
        rows.append("%s %s %.1f / %.1f GB layer %d/%d [%s]" % (m["state"], p["phase"], p["bytes"] / 1e9,
                    p["total"] / 1e9, p["layer"], p["n_layers"], units))
    else:
        rows.append(m["state"])
print("; ".join(rows) or "none")'
        [ -f "$LT_DIR/op.done" ] && return 0
        sleep 0.5
    done
}

run_q3() {
    LT_PORT=8433 LT_DIR=build/lbk_q3 LT_STORE=${Q3_STORE:?store with the Q3 file}
    echo "== q3: plain pantheon local (built-in catalog) + live load progress (H2n)"
    rm -rf "$LT_DIR/state" "$LT_DIR/op.done"
    unset QWEN35_BATCH_SLOTS   # qwen35moe: the box's own (batched) path
    lt_start --no-mdns || return 1
    KEY=$(lt_pair lbk)
    lt_check "built-in catalog accepted" lt_wait 60 sh -c "grep -q 'catalog version' $LT_DIR/serve.log"
    grep -m2 "catalog" "$LT_DIR/serve.log"
    echo "-- /local/hardware x10 before the load"
    hw_timing
    lt_check "host RAM room for the upload" lt_wait 600 host_ram_ok
    ( lt_post "$KEY" /local/models/load \
        '{"catalog_id":"qwen3.6-35b-a3b-q3km","profile_id":"k80-pair","units":[0,1]}' \
        > "$LT_DIR/load.json"; touch "$LT_DIR/op.done" ) &
    op=$!
    poll_progress 600
    wait "$op"   # not a bare `wait`: the box is a child of this shell too
    cat "$LT_DIR/load.json"; echo
    lt_check "loaded" grep -q '"units":\[0,1\]' "$LT_DIR/load.json"
    echo "-- /local/hardware x10 with the model loaded"
    hw_timing
    rm -f "$LT_DIR/op.done"
    ( lt_post "$KEY" /admin/models/unload '{"name":"qwen3.6-35b-a3b-q3km"}' > "$LT_DIR/unload.json"
      touch "$LT_DIR/op.done" ) &
    op=$!
    poll_progress 60
    wait "$op"
    cat "$LT_DIR/unload.json"; echo
    lt_check "nothing loaded" sh -c "curl -s -H 'Authorization: Bearer $KEY' \
        http://127.0.0.1:$LT_PORT/local/models/loaded | grep -q '\"models\":\[\]'"
    lt_stop
}

cpu_ticks() {   # utime + stime of pid $1, in clock ticks
    awk '{print $14 + $15}' "/proc/$1/stat"
}

run_idle() {
    LT_PORT=8433 LT_DIR=build/lbk_q3 LT_STORE=${Q3_STORE:?store with the Q3 file}
    echo "== idle: CPU cost of the background device memory reads"
    rm -rf "$LT_DIR/state"
    lt_start --no-mdns || return 1
    sleep 10
    local pid t0 t1
    pid=$(cat "$LT_DIR/pid")
    t0=$(cpu_ticks "$pid")
    sleep 60
    t1=$(cpu_ticks "$pid")
    echo "  $(( t1 - t0 )) ticks of CPU in 60 s (1 tick = 1/$(getconf CLK_TCK) s)"
    lt_check "under 1% of one core" [ $(( t1 - t0 )) -lt $(( $(getconf CLK_TCK) * 60 / 100 )) ]
    lt_stop
}

case "$WHAT" in
    small) run_small ;;
    q3) run_q3 ;;
    idle) run_idle ;;
    all) run_small; run_q3 ;;
esac
lt_done backlog_bank
