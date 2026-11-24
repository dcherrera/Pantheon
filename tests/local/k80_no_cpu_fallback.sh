#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# k80_no_cpu_fallback.sh — on hardware: a load the GPU refuses fails fast with
# the reason and never lands on the host CPU forward (src/core/load_guard.h).
#
#   1. plain serve, Q4 KV at 262144 + --mtp on one die     -> refused, reason
#   2. plain serve, fp32 KV at 32768 on one die             -> refused, reason
#   3. a serve holding the model on die B at 8192 (control: the GPU path still
#      loads), then POST /admin/models/load of the same model on die A at
#      262144 + mtp (the runtime / Local path)              -> HTTP error with
#      the reason, die A's memory back to idle, serve still up
#
# Each run is watched from outside: MemAvailable, swap-in/out; a run that takes
# the host below WATCH_MIN_MIB is killed and reported, so the check cannot be
# the thrash it tests for.
#
# Usage: tests/local/k80_no_cpu_fallback.sh <die A> <die B> <model.gguf>
#        (physical die numbers; CUDA_DEVICE_ORDER=PCI_BUS_ID). Run from the
#        Pantheon checkout after `make USE_CUDA=1`. Results: build/no_cpu_fallback/.

set -u
die_a=${1:?die A}; die_b=${2:?die B}; model=${3:?model}
port=${PORT:-18911}
min_mib=${WATCH_MIN_MIB:-1600}
out=build/no_cpu_fallback
mkdir -p "$out"
: > "$out/summary"

mem_line() {
    printf '%s %s %s %s\n' "$(date +%s.%N)" \
        "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" \
        "$(awk '/^pswpin/{print $2}' /proc/vmstat)" "$(awk '/^pswpout/{print $2}' /proc/vmstat)"
}

# Watch `pid` until it exits (or `until_file` holds `until_pat`); kill it if the
# host gets low. Appends samples to $out/$name.mem.
watch_pid() {
    local name=$1 pid=$2 until_file=${3:-} until_pat=${4:-}
    while kill -0 "$pid" 2>/dev/null; do
        mem_line >> "$out/$name.mem"
        local avail
        avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
        if [ "$avail" -lt "$min_mib" ]; then
            kill -9 "$pid"
            echo "$name KILLED: MemAvailable ${avail} MiB < ${min_mib}" >> "$out/summary"
            return 1
        fi
        if [ -n "$until_pat" ] && grep -q "$until_pat" "$until_file" 2>/dev/null; then
            return 0
        fi
        sleep 0.5
    done
    return 0
}

summarize() {
    awk -v n="$1" -v rc="$2" -v s="$3" 'NR==1{si=$3;so=$4;min=$2} {if($2<min)min=$2; lsi=$3; lso=$4}
        END{printf "%s rc=%s secs=%s min_avail_mib=%d swapin_pages=%d swapout_pages=%d\n",
            n, rc, s, min, lsi-si, lso-so}' "$out/$1.mem" >> "$out/summary"
}

refused_run() {
    local name=$1; shift
    local t0 t1 pid rc
    t0=$(date +%s)
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$die_a" "$@" > "$out/$name.log" 2>&1 &
    pid=$!
    mem_line > "$out/$name.mem"
    watch_pid "$name" "$pid"
    wait "$pid"; rc=$?
    t1=$(date +%s)
    summarize "$name" "$rc" "$((t1 - t0))"
    grep -m1 "REFUSED\|failed to load" "$out/$name.log" >> "$out/summary"
    grep -c "CPU forward\|CPU-resident" "$out/$name.log" | sed 's/^/  cpu-forward lines: /' >> "$out/summary"
    nvidia-smi -i "$die_a" --query-gpu=memory.used --format=csv,noheader | sed "s/^/  die $die_a after: /" >> "$out/summary"
}

refused_run q4_256k_mtp ./pantheon serve --model "$model" --name orn --dies 0 --port "$port" \
    --ctx-cap 262144 --mtp
refused_run fp32_32k env QWEN35_KV_Q4=0 ./pantheon serve --model "$model" --name orn --dies 0 \
    --port "$port" --ctx-cap 32768

# 3. Runtime path: control serve on die B (index 1), refused admin load on die A (index 0).
name=runtime
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$die_a,$die_b" ./pantheon serve \
    --model "$model" --name orn-ok --dies 1 --port "$port" --ctx-cap 8192 > "$out/$name.log" 2>&1 &
pid=$!
mem_line > "$out/$name.mem"
if watch_pid "$name" "$pid" "$out/$name.log" "serve: listening" && kill -0 "$pid" 2>/dev/null; then
    echo "control: $model loaded on die $die_b (GPU path intact)" >> "$out/summary"
    t0=$(date +%s)
    curl -s -o "$out/$name.http" -w "%{http_code}" -X POST "http://127.0.0.1:$port/admin/models/load" \
        -H 'Content-Type: application/json' \
        -d "{\"path\":\"$model\",\"name\":\"orn-big\",\"dies\":[0],\"ctx_cap\":262144,\"mtp\":true}" \
        > "$out/$name.code"
    t1=$(date +%s)
    mem_line >> "$out/$name.mem"
    echo "admin load: HTTP $(cat "$out/$name.code") in $((t1 - t0)) s: $(cat "$out/$name.http")" >> "$out/summary"
    nvidia-smi -i "$die_a" --query-gpu=memory.used --format=csv,noheader | sed "s/^/  die $die_a after: /" >> "$out/summary"
    curl -s "http://127.0.0.1:$port/v1/models" | sed 's/^/  models: /' >> "$out/summary"
    echo >> "$out/summary"
    kill -0 "$pid" 2>/dev/null && echo "  serve still up" >> "$out/summary"
    kill "$pid"
else
    echo "control: serve did not come up (see $out/$name.log)" >> "$out/summary"
    kill "$pid" 2>/dev/null
fi
wait "$pid" 2>/dev/null
summarize "$name" - -
echo CHECK_DONE >> "$out/summary"
