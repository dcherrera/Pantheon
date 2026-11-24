#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# @file ctx_cap_refusal.sh
# @brief A CPU-resident serve refuses a prompt over --ctx-cap with a 422
#        context_over_budget whose kv_cap is the cap that refused it (H2d, H2c).
#
# @status Mac (CPU) and a p150 host (TT build, model on the CPU). Exit 0 = pass.
# @issues none
# @todo none
#
# Usage: tests/local/ctx_cap_refusal.sh [pantheon binary] [model.gguf] [port]
#   defaults: ./pantheon, models/qwen3-0.6b-Q8_0.gguf, 8425
# Prompt: ~5,000 tokens at --ctx-cap 4096; then a ~200-token prompt must still
# be answered. Log: build/ctx_cap_refusal/serve.log

cd "$(dirname "$0")/../.." || exit 1
bin=${1:-./pantheon}
model=${2:-models/qwen3-0.6b-Q8_0.gguf}
port=${3:-8425}
dir=build/ctx_cap_refusal
mkdir -p "$dir"
fail=0

# One chat with ~N filler tokens; prints "<http status> <body>".
chat() {
    python3 - "$port" "$1" <<'EOF'
import json, sys, urllib.request, urllib.error
port, n = sys.argv[1], int(sys.argv[2])
words = ("the quick brown fox jumps over the lazy dog " * (n // 9 + 1)).split()[:n]
body = json.dumps({"model": "q", "temperature": 0, "max_tokens": 8,
                   "enable_thinking": False,
                   "messages": [{"role": "user", "content": " ".join(words) + "\nWhich animal jumps?"}]})
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body.encode(),
                             {"Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=900)
    print(r.status, r.read().decode()[:400])
except urllib.error.HTTPError as e:
    print(e.code, e.read().decode()[:400])
except Exception as e:
    print(0, str(e))
EOF
}

"$bin" serve --model "$model" --name q --port "$port" --ctx-cap 4096 > "$dir/serve.log" 2>&1 &
pid=$!
for _ in $(seq 1 180); do
    curl -s "127.0.0.1:$port/v1/models" | grep -q '"q"' && break
    kill -0 $pid 2>/dev/null || break
    sleep 1
done

out=$(chat 5000)
echo "5000 tokens: $out"
if [ "${out%% *}" = 422 ] && [[ "$out" == *context_over_budget* ]] && [[ "$out" == *"kv_cap=4096"* ]]; then
    echo "PASS over-cap prompt refused, kv_cap=4096"
else
    echo "FAIL over-cap prompt was not refused with kv_cap=4096"; fail=1
fi
out=$(chat 200)
echo "200 tokens: ${out:0:160}"
case "$out" in
    "200 "*) echo "PASS short prompt answered" ;;
    *) echo "FAIL short prompt not answered"; fail=1 ;;
esac
grep -a "REJECT context_over_budget" "$dir/serve.log" | tail -1

kill $pid 2>/dev/null; wait $pid 2>/dev/null
exit $fail
