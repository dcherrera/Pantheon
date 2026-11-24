#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# tests/qwen3_parity/run.sh <model.gguf> <out-dir> [<ref-dir>]
#
# Greedy (temperature 0) generation of a fixed prompt set through the built
# ./pantheon, one file per prompt in <out-dir>. With <ref-dir> (the Mac CPU
# oracle's output dir) every file must be byte-identical: prints SAME/DIFF per
# prompt and exits 1 on any DIFF. Run from the Pantheon root, on any backend.
#
# Extra pantheon flags: QP_FLAGS (e.g. QP_FLAGS="--dies 6"). Tokens: QP_TOKENS (64).
# foundry's CUDA init prints "[fc_cuda] ..." lines on stdout; those are dropped.
set -u
model=${1:?model.gguf}; out=${2:?out dir}; ref=${3:-}
tokens=${QP_TOKENS:-64}
bin=${QP_BIN:-./pantheon}   # pantheon-tt on a TT build
# A 4k KV ring: the 40k default reserves ~4.6 GB host RSS, which an 8 GB box running
# other work OOMs on. The prompts are far shorter, so the output is the same.
export PN_SESSION_MAX_CTX=${PN_SESSION_MAX_CTX:-4096}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"

fail=0; n=0
while IFS='|' read -r mode prompt; do
    [ -z "$mode" ] && continue
    n=$((n + 1))
    raw=""; [ "$mode" = raw ] && raw="--raw"
    # shellcheck disable=SC2086
    "$bin" --model "$model" $raw --prompt "$prompt" --temperature 0 \
        --max-tokens "$tokens" ${QP_FLAGS:-} < /dev/null 2> "$out/p$n.log" \
        | grep -v '^\[fc_cuda\]' > "$out/p$n.txt"
    if [ -n "$ref" ]; then
        if cmp -s "$out/p$n.txt" "$ref/p$n.txt"; then echo "p$n SAME"
        else echo "p$n DIFF"; fail=1; fi
    else
        echo "p$n $(wc -c < "$out/p$n.txt") bytes"
    fi
done < "$here/prompts.txt"
[ -n "$ref" ] && { [ $fail = 0 ] && echo "GATE PASS ($n prompts)" || echo "GATE FAIL"; }
exit $fail
