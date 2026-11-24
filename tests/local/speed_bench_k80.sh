#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# @file speed_bench_k80.sh
# @brief The catalog's measured numbers for a K80 profile, taken the way Pantheon
#        Local runs it: fp32 KV (QWEN35_KV_Q4=0, PN_LADDER_KV_Q4=0), MTP on,
#        the profile's ctx-cap / sessions / reasoning budget, one board.
#
# Usage (box checkout on the K80 bank, the box stopped; never two busy dies of one
# board for timing while another bench runs):
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     tests/local/speed_bench_k80.sh <model.gguf> <dies> <ctx_cap> <sessions> <depths> <out dir>
# e.g. ... build/local_hw/store/qwen3.6-35b-a3b-q3km-17350b1355d3.gguf 0,1 16384 1 8192,14336 build/speed
# MTP=0 benches plain decode instead. Streams one greedy 256-token answer per depth through tools/batch_verify/mtp_serve_client.py
# (TTFT and decode tok/s per request in <out>/client/results.jsonl), filler = the gate's.
set -u
cd "$(dirname "$0")/../.." || exit 1
model=${1:?model.gguf}; dies=${2:?dies}; ctx=${3:?ctx_cap}; sessions=${4:?sessions}
depths=${5:?depths}; out=${6:?out dir}
PORT=${PORT:-8414}
MTP_FLAG=--mtp; [ "${MTP:-1}" = 0 ] && MTP_FLAG=   # MTP=0: plain decode
mkdir -p "$out"
git show a1d18a6:src/models/qwen35moe_forward_cuda.c > "$out/filler.txt" || exit 2
QWEN35_KV_Q4=0 PN_LADDER_KV_Q4=0 setsid ./pantheon serve --model "$model" --name bench \
    --dies "$dies" $MTP_FLAG --ctx-cap "$ctx" --max-sessions "$sessions" --reasoning-budget 2000 \
    --host 127.0.0.1 --port "$PORT" > "$out/serve.log" 2>&1 < /dev/null &
pid=$!
for _ in $(seq 1 900); do
    kill -0 "$pid" 2>/dev/null || { echo "serve exited"; tail -5 "$out/serve.log"; exit 3; }
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
    sleep 1
done
python3 tools/batch_verify/mtp_serve_client.py --port "$PORT" --model bench \
    --filler "$out/filler.txt" --prompt tools/batch_verify/mtp_serve_prompt.txt \
    --depths "$depths" --temps 0 --max-tokens 256 --sessions bench-s1 --out "$out/client" \
    > "$out/client.out" 2>&1
rc=$?
kill -TERM "$pid" 2>/dev/null
for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
kill -KILL "$pid" 2>/dev/null
grep -c "forcing Q4" "$out/serve.log" | sed 's/^/forced-q4 lines: /'
cat "$out/client/results.jsonl" 2>/dev/null
echo "SPEED_BENCH_DONE rc=$rc"
