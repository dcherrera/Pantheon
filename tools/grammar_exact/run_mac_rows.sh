#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_mac_rows.sh — real grammar-masked rows on the Mac (CPU build, no card): serve Ornith with
# PN_GRAMMAR_ROW_DUMP armed, replay captured client chat bodies one at a time (seeded), and keep
# one dump per request for grammar_row_census.
#
#   run_mac_rows.sh <bodies-dir> <replay_bodies.py> <run-name> [glob]
#
# Env: MODEL (default the Pantheon checkout's models/ornith-1.5-9b Q8_0 GGUF), FOUNDRY (the
# foundry checkout holding metal/run_guarded.sh), ROWS_PORT (18431), SEED (1234), CAP (max_tokens cap,
# 2048), CTX (ctx-cap, 32768); GUARD_* pass through to run_guarded.sh.
# Every model load goes through foundry's metal/run_guarded.sh (a bare load once panicked the M4).
# Output: tools/grammar_exact/data/<run-name>/ (gitignored): rows/<body>.grd, replay/<body>/,
# serve.log.
set -u
BODIES=${1:?bodies dir}; REPLAY=${2:?replay_bodies.py}; RUN=${3:?run name}; GLOB=${4:-req_*.json}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
MODEL=${MODEL:-$HOME/TeamIDE/david/Pantheon/models/ornith-1.5-9b/Ornith-1.5-9B-Q8_0.gguf}
FOUNDRY=${FOUNDRY:-$ROOT/../tt-grammar-foundry}
PORT=${ROWS_PORT:-18431}; SEED=${SEED:-1234}; CAP=${CAP:-2048}; CTX=${CTX:-32768}
OUT=$HERE/data/$RUN
mkdir -p "$OUT/rows" "$OUT/replay"
DUMP=$OUT/current.grd
rm -f "$DUMP"

PN_GRAMMAR_ROW_DUMP=$DUMP "$FOUNDRY/metal/run_guarded.sh" "$ROOT/pantheon" serve --model "$MODEL" \
    --name ornith --port "$PORT" --ctx-cap "$CTX" > "$OUT/serve.log" 2>&1 &
GUARD=$!
for i in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1 && break
    kill -0 "$GUARD" 2> /dev/null || { echo "serve exited (see $OUT/serve.log)"; exit 1; }
    sleep 1
done
echo "serve up $(date +%T)"
for body in $(cd "$BODIES" && ls $GLOB); do
    name=${body%.json}
    [ -s "$OUT/rows/$name.grd" ] || [ -e "$OUT/replay/$name/replay.tsv" ] && continue
    python3 "$REPLAY" "$BODIES" "$PORT" "$OUT/replay/$name" --only "$body" \
        --patch "{\"seed\": $SEED}" --cap-max-tokens "$CAP" > /dev/null 2>&1
    [ -e "$DUMP" ] && mv "$DUMP" "$OUT/rows/$name.grd"
    echo "$name $(cut -f5,6 "$OUT/replay/$name/replay.tsv" 2> /dev/null) rows=$(ls -la "$OUT/rows/$name.grd" 2> /dev/null | awk '{print $5}')"
    kill -0 "$GUARD" 2> /dev/null || { echo "serve died (see $OUT/serve.log)"; exit 1; }
done
kill "$GUARD" 2> /dev/null
pkill -f "pantheon serve --model $MODEL --name ornith --port $PORT" 2> /dev/null
wait
echo "DONE $(date +%T)"
