#!/bin/bash
# tests/local/plain_probe.sh <port> [serve flags...]   (run ON the box)
# Start a plain `pantheon serve` of the installed small model, wait, send one
# greedy chat, print the reply, stop it. Diagnoses whether a profile's flags load
# outside the appliance. Log: build/local_hw/plain_probe.log
cd "$(dirname "$0")/../.." || exit 1
port=${1:?port}; shift
f=$(ls build/local_hw/store/*.gguf | head -1)
./pantheon serve --model "$f" --name qwen3-0.6b-q8 --port "$port" "$@" > build/local_hw/plain_probe.log 2>&1 &
pid=$!
for _ in $(seq 1 120); do curl -s "127.0.0.1:$port/v1/models" | grep -q qwen3 && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
grep -n "FATAL\|carve\|refus" build/local_hw/plain_probe.log | head -5
# Prints the raw response as its last line (the caller extracts the message).
curl -s -m 300 "127.0.0.1:$port/v1/chat/completions" -d '{"temperature":0,"max_tokens":16,"seed":1,"enable_thinking":false,"messages":[{"role":"user","content":"Name three colours of the rainbow."}]}'
echo
kill $pid 2>/dev/null; wait $pid 2>/dev/null
