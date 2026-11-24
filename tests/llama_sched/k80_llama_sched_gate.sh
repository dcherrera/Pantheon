#!/bin/bash
# tests/llama_sched/k80_llama_sched_gate.sh   (run ON the bank, from the repo root, detached)
# The llama-arch batched-scheduler gate. One die, one serve at a time.
#   1. reference: QWEN35_BATCH_SLOTS=0 (single-user path), the greedy prompts one by one;
#   2. batched (the default) at each --max-sessions S in $SESSIONS: the load must carve
#      (no "cannot carve"), then S requests sent AT ONCE must all complete, their text
#      byte-identical to the reference, and the step log must show n=S (they decoded together);
#   3. refusal: S=$BIG_S at --ctx-cap $BIG_CTX must be refused at load with the sizes named.
# Env: MODEL, DIE (physical PCI-order index), CTX, SESSIONS, BIG_S, BIG_CTX, PORT, MAX_TOKENS.
# Output: build/llama_sched/gate/ (logs, replies) and gate.txt; last line "GATE_DONE fail=N".
#   setsid nohup bash tests/llama_sched/k80_llama_sched_gate.sh > /dev/null 2>&1 < /dev/null &
#
# @status complete  @issues none  @todo none
cd "$(dirname "$0")/../.." || exit 1
MODEL=${MODEL:-build/local_models/qwen3-0.6b-Q8_0.gguf}
DIE=${DIE:-6}
CTX=${CTX:-1024}
SESSIONS=${SESSIONS:-1 2 4 8}
BIG_S=${BIG_S:-8}
BIG_CTX=${BIG_CTX:-32768}
PORT=${PORT:-8461}
MAX_TOKENS=${MAX_TOKENS:-40}
out=build/llama_sched/gate
rm -rf "$out"; mkdir -p "$out"
exec > "$out/gate.txt" 2>&1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DIE QWEN35_BS_STATS=1
fail=0
PROMPTS=("Name three colours of the rainbow." "Count from one to ten in words."
         "What is the capital of France? Answer in one sentence."
         "Write a haiku about the sea.")
check() { if "${@:2}"; then echo "PASS $1"; else echo "FAIL $1"; fail=$((fail + 1)); fi; }

serve_start() {   # serve_start <log> <flags...>: 0 once /v1/models answers, 1 if it exits
    ./pantheon serve --model "$MODEL" --name fixture --port "$PORT" --dies 0 \
        --ctx-cap "${CTX_ARG:-$CTX}" "${@:2}" > "$1" 2>&1 &
    SERVE_PID=$!
    for _ in $(seq 1 180); do
        curl -s "127.0.0.1:$PORT/v1/models" | grep -q fixture && return 0
        kill -0 "$SERVE_PID" 2> /dev/null || return 1
        sleep 1
    done
    return 1
}
serve_stop() { kill "$SERVE_PID" 2> /dev/null; wait "$SERVE_PID" 2> /dev/null; }

chat() {   # chat <prompt index> <out file>: the reply's content only
    curl -s -m 600 "127.0.0.1:$PORT/v1/chat/completions" -d "{\"model\":\"fixture\",
      \"temperature\":0,\"max_tokens\":$MAX_TOKENS,\"seed\":1,\"enable_thinking\":false,
      \"messages\":[{\"role\":\"user\",\"content\":\"${PROMPTS[$1]}\"}]}" > "$2.raw"
    python3 -c "import json,sys; print(ascii(json.load(open(sys.argv[1],encoding='utf-8',errors='surrogateescape'))['choices'][0]['message'].get('content')))" \
        "$2.raw" > "$2" 2> /dev/null
}

echo "== $(git log --oneline -1)  model=$MODEL die=$DIE ctx=$CTX"
QWEN35_BATCH_SLOTS=0 serve_start "$out/ref.log" --max-sessions 1 ||
    { echo "FAIL reference serve did not start"; echo "GATE_DONE fail=1"; exit 1; }
for i in 0 1 2 3; do
    chat "$i" "$out/ref_$i.txt"; echo "ref $i: $(cat "$out/ref_$i.txt")"
    check "reference $i has a reply" [ -s "$out/ref_$i.txt" ]
done
serve_stop

for S in $SESSIONS; do
    if ! serve_start "$out/s$S.log" --max-sessions "$S" --max-concurrent "$S"; then
        echo "FAIL S=$S: serve refused the load"; fail=$((fail + 1))
        grep -m3 "carve\|FATAL" "$out/s$S.log"; serve_stop; continue
    fi
    echo "S=$S carve: $(grep -m1 'session blocks for' "$out/s$S.log")"
    check "S=$S batched scheduler on" grep -q "carved $S session blocks" "$out/s$S.log"
    for k in $(seq 0 $((S - 1))); do chat $((k % 4)) "$out/s${S}_$k.txt" & done
    wait $(jobs -p | grep -v "^$SERVE_PID\$") 2> /dev/null
    for k in $(seq 0 $((S - 1))); do
        check "S=$S request $k == single-user" cmp -s "$out/s${S}_$k.txt" "$out/ref_$((k % 4)).txt"
    done
    nmax=$(sed -n 's/.*\[bs\] step=[0-9]* n=\([0-9]*\).*/\1/p' "$out/s$S.log" | sort -n | tail -1)
    check "S=$S decoded together (max step n=$nmax)" [ "${nmax:-0}" -eq "$S" ]
    serve_stop
done

# BIG_S=0 skips the refusal arm: WITHOUT the fix this load is OOM-killed, not refused
# (the slot states are eager host allocations made before any check).
[ "$BIG_S" -gt 0 ] || { echo "GATE_DONE fail=$fail"; exit 0; }
CTX_ARG=$BIG_CTX serve_start "$out/big.log" --max-sessions "$BIG_S" --max-concurrent "$BIG_S"
rc=$?
serve_stop
echo "refusal: $(grep -m1 'REFUSED' "$out/big.log")"
check "S=$BIG_S at ctx $BIG_CTX refused at load with sizes" \
    sh -c "[ $rc -ne 0 ] && grep -q 'REFUSED: host KV ring (.* sessions x $BIG_CTX positions) needs' '$out/big.log'"
echo "GATE_DONE fail=$fail"
