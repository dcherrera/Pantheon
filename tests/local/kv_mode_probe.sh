#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# @file kv_mode_probe.sh
# @brief Does a K80 qwen35 serve generate a full answer in a given KV / MTP / path mode?
#
# Usage (box checkout on the K80 bank, box stopped):
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     tests/local/kv_mode_probe.sh <model.gguf> <dies> <out dir> <label> <env...> -- <serve flags...>
# e.g. ... fp32-mtp-batched QWEN35_KV_Q4=0 PN_LADDER_KV_Q4=0 -- --mtp --ctx-cap 16384 --max-sessions 1
# One greedy 64-token answer to a short prompt; prints completion_tokens and any
# "MTP cycle failed" lines. Line "PROBE <label> completion_tokens=N mtp_failed=M".
set -u
cd "$(dirname "$0")/../.." || exit 1
model=${1:?model}; dies=${2:?dies}; out=${3:?out dir}; label=${4:?label}; shift 4
envs=(); while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done; shift
PORT=${PORT:-8415}
mkdir -p "$out"
env "${envs[@]}" setsid ./pantheon serve --model "$model" --name probe --dies "$dies" "$@" \
    --host 127.0.0.1 --port "$PORT" > "$out/$label.log" 2>&1 < /dev/null &
pid=$!
for _ in $(seq 1 900); do
    kill -0 "$pid" 2>/dev/null || break
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
    sleep 1
done
curl -s -m 600 -H "X-Session-Id: probe-1" "http://127.0.0.1:$PORT/v1/chat/completions" \
    -d '{"model":"probe","max_tokens":64,"temperature":0,"messages":[{"role":"user","content":"Count from one to twenty in words."}]}' \
    > "$out/$label.json"
kill -TERM "$pid" 2>/dev/null
for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
kill -KILL "$pid" 2>/dev/null
echo "PROBE $label completion_tokens=$(grep -o '"completion_tokens":[0-9]*' "$out/$label.json" | cut -d: -f2) mtp_failed=$(grep -c 'MTP cycle failed' "$out/$label.log")"
