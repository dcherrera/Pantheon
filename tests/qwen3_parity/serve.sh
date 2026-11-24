#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# tests/qwen3_parity/serve.sh <model.gguf> <port> <out-dir> [<ref-dir>]
#
# The same fixed prompts through a plain `pantheon serve` (the path the e2e and
# the appliance use): greedy chat completions, QP_TOKENS (64) tokens, thinking
# off, one reply text per prompt in <out-dir>/s<N>.txt. With <ref-dir> (the Mac
# CPU serve's output) every reply must be byte-identical; exits 1 on any DIFF.
# Extra env for the serve process: QP_ENV (e.g. "QWEN35_BATCH_SLOTS=0").
set -u
model=${1:?model.gguf}; port=${2:?port}; out=${3:?out dir}; ref=${4:-}
tokens=${QP_TOKENS:-64}
bin=${QP_BIN:-./pantheon}   # pantheon-tt on a TT build
# A 4k KV ring, as run.sh (the 40k default is refused as memory-constrained on 8 GB).
export PN_SESSION_MAX_CTX=${PN_SESSION_MAX_CTX:-4096}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"

# shellcheck disable=SC2086
env ${QP_ENV:-} "$bin" serve --model "$model" --name qwen3 --port "$port" \
    > "$out/serve.log" 2>&1 < /dev/null &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for _ in $(seq 1 300); do
    curl -s "127.0.0.1:$port/v1/models" | grep -q qwen3 && break
    kill -0 $pid 2>/dev/null || { echo "serve exited"; tail -5 "$out/serve.log"; exit 2; }
    sleep 1
done

fail=0; n=0
while IFS='|' read -r _ prompt; do
    [ -z "$prompt" ] && continue
    n=$((n + 1))
    body=$(python3 -c 'import json,sys; print(json.dumps({"temperature":0,"max_tokens":int(sys.argv[2]),
"seed":1,"enable_thinking":False,"messages":[{"role":"user","content":sys.argv[1]}]}))' "$prompt" "$tokens")
    curl -s -m 600 "127.0.0.1:$port/v1/chat/completions" -d "$body" > "$out/s$n.json"
    python3 -c 'import json,sys; sys.stdout.write(json.load(sys.stdin)["choices"][0]["message"]["content"])' \
        < "$out/s$n.json" > "$out/s$n.txt" 2> /dev/null || echo "s$n: no reply: $(head -c 300 "$out/s$n.json")"
    if [ -n "$ref" ]; then
        if cmp -s "$out/s$n.txt" "$ref/s$n.txt"; then echo "s$n SAME"
        else echo "s$n DIFF"; fail=1; fi
    else
        echo "s$n $(wc -c < "$out/s$n.txt") bytes"
    fi
done < "$here/prompts.txt"
[ -n "$ref" ] && { [ $fail = 0 ] && echo "SERVE GATE PASS ($n prompts)" || echo "SERVE GATE FAIL"; }
exit $fail
