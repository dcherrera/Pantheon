#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# @file mem_model_measure.sh
# @brief L1: measure a catalog profile's device memory at (ctx, sessions) points.
#
# Usage (from the Pantheon checkout on the card host, box stopped):
#   tests/local/mem_model_measure.sh <k80|p150> <model.gguf> <dies> <out.csv> <ctx:sessions[:prompt_words]>...
#
# Per point: start `pantheon serve` on <dies> with the profile's flags (MTP on (k80 --mtp;
# p150 arms the head next to the weights), Q4 KV (QWEN35_KV_Q4=1, the K80 catalog default;
# KV=fp32: QWEN35_KV_Q4=0 PN_LADDER_KV_Q4=0, the fp32 profiles), run <sessions> concurrent chats (distinct session ids) with a prompt
# of <prompt_words> words (default 200), stop. Writes one CSV row per point:
#   ctx,sessions,prompt_words,unit,load_bytes,peak_bytes,kv_q4_seen,ok,host_anon_bytes
# (host_anon_bytes = the serve's RssAnon after the chats: host RAM it holds beyond the
# mmapped weights, which are page cache).
# k80: per die, nvidia-smi memory.used polled every 250 ms (load = before the chats,
# peak = max over the whole run). p150: the card allocator's "DRAM after arm" line
# (a bump allocator: used == peak), load == peak.
# Env: PN_BIN (default ./pantheon), PORT (default 8413), EXTRA_SERVE_ARGS, MTP=0 (no --mtp),
# KV=fp32 (fp32 KV instead of Q4), SMI_DIES (k80: the nvidia-smi indices of <dies>, in order,
# when CUDA_VISIBLE_DEVICES remaps them, e.g. dies 0,1,2,3 = SMI_DIES 8,9,10,11).
set -u
kind=${1:?k80|p150}; model=${2:?model.gguf}; dies=${3:?dies}; out=${4:?out.csv}; shift 4
PN_BIN=${PN_BIN:-./pantheon}; PORT=${PORT:-8413}
mtp_flag=--mtp; [ "$kind" = p150 ] && mtp_flag=   # a TT build arms the head from the store
[ "${MTP:-1}" = 0 ] && mtp_flag=                   # MTP=0: the profile runs plain decode
kv_env="QWEN35_KV_Q4=1"                              # the K80 catalog default
[ "${KV:-q4}" = fp32 ] && kv_env="QWEN35_KV_Q4=0 PN_LADDER_KV_Q4=0"
dir=$(dirname "$out"); mkdir -p "$dir"
[ -f "$out" ] || echo "ctx,sessions,prompt_words,unit,load_bytes,peak_bytes,kv_q4_seen,ok,host_anon_bytes" > "$out"

mib() { echo $(( $1 * 1048576 )); }

filler() {   # filler <words>: a prompt of that many short words
    local n=$1 i s=""
    for ((i = 0; i < n; i++)); do s+="w$((i % 97)) "; done
    printf '%s' "$s"
}

chat() {     # chat <session id> <words>
    local body
    body=$(printf '{"model":"mm","max_tokens":32,"temperature":0,"messages":[{"role":"user","content":"%s Say ok."}]}' "$(filler "$2")")
    curl -s -m 3600 -H "X-Session-Id: $1" -H 'Content-Type: application/json' \
        "http://127.0.0.1:$PORT/v1/chat/completions" -d "$body" > "$dir/chat_$1.json"
}

wait_ready() {
    local pid=$1
    for _ in $(seq 1 1200); do
        kill -0 "$pid" 2>/dev/null || return 1
        curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q '"mm"' && return 0
        sleep 0.5
    done
    return 1
}

smi_max() {  # smi_max <log> <die>: max memory.used (MiB) for that die
    awk -F', *' -v d="$2" '$1 == d && $2 + 0 > m { m = $2 + 0 } END { print m + 0 }' "$1"
}

measure_point() {
    local ctx=$1 sessions=$2 words=$3 tag log smi smi_pid pid ok=1 q4
    tag="c${ctx}_s${sessions}_w${words}"
    log="$dir/serve_$tag.log"; smi="$dir/smi_$tag.txt"
    if [ "$kind" = k80 ]; then
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 250 > "$smi" &
        smi_pid=$!
    fi
    env $kv_env setsid "$PN_BIN" serve --model "$model" --name mm --dies "$dies" $mtp_flag \
        --ctx-cap "$ctx" --max-sessions "$sessions" --reasoning-budget 2000 \
        --host 127.0.0.1 --port "$PORT" ${EXTRA_SERVE_ARGS:-} > "$log" 2>&1 < /dev/null &
    pid=$!
    if ! wait_ready "$pid"; then ok=0; fi
    sleep 2
    local load_line; load_line=$(wc -l < "${smi:-/dev/null}" 2>/dev/null || echo 0)
    if [ "$ok" = 1 ]; then
        local s chat_pids=()
        for ((s = 0; s < sessions; s++)); do chat "mm-$tag-$s" "$words" & chat_pids+=($!); done
        wait "${chat_pids[@]}"
        grep -q '"choices"' "$dir/chat_mm-$tag-0.json" || ok=0
    fi
    sleep 2
    local hwm; hwm=$(awk '/^RssAnon/ { print $2 * 1024 }' "/proc/$pid/status" 2>/dev/null)
    kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 120); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -KILL "$pid" 2>/dev/null
    q4=$(grep -c "forcing Q4\|kv_q4=1\|Q4 KV" "$log")
    if [ "$kind" = k80 ]; then
        kill "$smi_pid" 2>/dev/null; wait "$smi_pid" 2>/dev/null
        local d
        local smi_dies=${SMI_DIES:-$dies}
        for d in ${smi_dies//,/ }; do
            local load peak
            load=$(head -n "$load_line" "$smi" | smi_max /dev/stdin "$d")
            peak=$(smi_max "$smi" "$d")
            echo "$ctx,$sessions,$words,$d,$(mib "$load"),$(mib "$peak"),$q4,$ok,${hwm:-0}" >> "$out"
        done
    else
        local gib used
        gib=$(grep "DRAM after arm" "$log" | tail -1 | sed -n 's/.*— \([0-9.]*\) GiB used.*/\1/p')
        used=$(awk -v g="${gib:-0}" 'BEGIN { printf "%.0f", g * 1073741824 }')
        echo "$ctx,$sessions,$words,$dies,$used,$used,$q4,$ok,${hwm:-0}" >> "$out"
    fi
    tail -n 3 "$out"
}

for point in "$@"; do
    IFS=: read -r ctx sessions words <<< "$point"
    measure_point "$ctx" "$sessions" "${words:-200}"
done
echo "DONE"
