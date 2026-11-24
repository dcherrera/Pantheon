#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_slots_fused_k80.sh — the one-slot fused slots path is engaged and
# bit-identical to the unfused slots sequence (tests/test_qwen35_slots_fused.c).
# Runs the test with the default QWEN35_FUSE (all descriptors must build) and
# with QWEN35_FUSE=0 (unfused), and requires the same logits digest.
#
# Usage: tests/run_slots_fused_k80.sh <model.gguf>
#   (dies from the env, e.g. CUDA_VISIBLE_DEVICES=8,9,10,11 QWEN35_GPU_DIES=0,1,2,3)
# Exit 0 = PASS.
set -u
cd "$(dirname "$0")/.."
M=${1:?model.gguf}
T=./tests/test_qwen35_slots_fused
[ -x $T ] || { echo "build $T first (make USE_CUDA=1 $T)"; exit 2; }
A=$(env -u QWEN35_FUSE $T "$M" 2>/dev/null) || { echo "$A"; echo "FAIL: fused run"; exit 1; }
B=$(QWEN35_FUSE=0 $T "$M" --any 2>/dev/null) || { echo "$B"; echo "FAIL: unfused run"; exit 1; }
echo "fused:   $(echo "$A" | grep SLOTSFUSED)"
echo "unfused: $(echo "$B" | grep SLOTSFUSED)"
da=$(echo "$A" | grep -o 'logits_fnv1a=[0-9a-f]*')
db=$(echo "$B" | grep -o 'logits_fnv1a=[0-9a-f]*')
ta=$(echo "$A" | grep '^TOKENS')
tb=$(echo "$B" | grep '^TOKENS')
if [ -n "$da" ] && [ "$da" = "$db" ] && [ "$ta" = "$tb" ]; then
    echo "PASS: fused one-slot path bit-identical to unfused"
    exit 0
fi
echo "FAIL: digests differ ($da vs $db)"
exit 1
