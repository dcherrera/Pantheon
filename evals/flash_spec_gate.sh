#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# flash_spec_gate.sh — regression gate for the on-die flash attention (prefill)
# path, covering the two cases the original 12-token argmax reference missed.
# Both broke output correctness silently while the 12-token gate stayed green.
#
#   Pass 1 — MULTI-TILE PREFILL (no spec). The 12-token reference is smaller
#     than one 16-key attention tile, so the cross-tile online-softmax combine
#     was never exercised. The greedy suite's prompts are 27-35 tokens (2-3
#     tiles), so QWEN35_FLASH=0 vs =1 on plain greedy decode gates the
#     multi-tile prefill: argmax-exact => 0 divergence.
#
#   Pass 2 — FLASH UNDER SPECULATIVE DECODE. spec-verify re-enters the BATCHED
#     flash path mid-sequence (pos0 > 0), reusing the GPU KV cache. A partial
#     accept leaves rejected draft rows in that GPU cache, and interleaved
#     single-token host-path decodes update only the host KV mirror; the GPU
#     mirror goes stale below the sync watermark. Flash+spec diverged 73% while
#     flash-without-spec was clean. spec is designed lossless (GEN output is
#     identical with spec on or off), so QWEN35_FLASH=0 vs =1 with
#     QWEN35_SPEC=1 must also be 0 divergence.
#
#   Pass 3 — BATCHED-PREFILL CAUSALITY. Reference-free: the first M per-position
#     argmaxes of an M-token chunked prefill must equal the first M of an N-token
#     one (N=384 > PN_KVQ4_WMAX). The q4 window ring once lapped inside a batched
#     call, so rows read K/V from LATER rows; +7.53% batched perplexity while every
#     A/B stayed green (plans/results/q4_win_ring_hazard.md). One build against
#     itself, under the deployed cache config (QWEN35_KV_Q4=1).
#     tests/test_qwen35_prefill_causality; red proof tests/causality_red/run_red.sh.
#
# NOTE: do NOT gate the flash path with a bit-exact logit memcmp
# of batched vs sequential prefill. The flash kernel reduces the softmax
# in tile order and drifts ~1e-6 vs the host's single-pass global max — it is
# argmax-exact by design, not bit-exact. The eval-harness divergence measure
# (argmax) is the correct criterion.
#
# Usage: evals/flash_spec_gate.sh [model.gguf]
#   env: QWEN35_GPU_SPLIT_LAYER / QWEN35_GPU_DIE_A / QWEN35_GPU_DIE_B
#        (defaults 20 / 0 / 1), EVAL_MODEL, EVAL_SKIP_PPL (default 0).
set -u

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
model="${EVAL_MODEL:-${1:-}}"
[ -n "$model" ] && [ -f "$model" ] || { echo "FAIL: model not found: '$model'" >&2; exit 2; }
eval_ppl="$repo_root/tests/eval_ppl"
[ -x "$eval_ppl" ] || { echo "FAIL: build tests/eval_ppl (make USE_CUDA=1 tests/eval_ppl)" >&2; exit 2; }
causality="$repo_root/tests/test_qwen35_prefill_causality"
[ -x "$causality" ] || { echo "FAIL: build tests/test_qwen35_prefill_causality (make USE_CUDA=1 tests/test_qwen35_prefill_causality)" >&2; exit 2; }

: "${QWEN35_GPU_SPLIT_LAYER:=20}"; export QWEN35_GPU_SPLIT_LAYER
: "${QWEN35_GPU_DIE_A:=0}";        export QWEN35_GPU_DIE_A
: "${QWEN35_GPU_DIE_B:=1}";        export QWEN35_GPU_DIE_B
skip_ppl="${EVAL_SKIP_PPL:-0}"

rc=0

echo "===== Pass 1: multi-tile prefill  (FLASH=0 vs =1, plain greedy) ====="
EVAL_SKIP_PPL="$skip_ppl" EVAL_OUT="$repo_root/evals/out/flash_gate_prefill" \
EVAL_ENV_A="QWEN35_FLASH=0" EVAL_ENV_B="QWEN35_FLASH=1" EVAL_MODEL="$model" \
    "$repo_root/evals/eval_quality.sh" "$eval_ppl" "$eval_ppl" || rc=1

echo "===== Pass 2: flash + speculative decode  (FLASH=0 vs =1, QWEN35_SPEC=1) ====="
QWEN35_SPEC=1 EVAL_SKIP_PPL="$skip_ppl" EVAL_OUT="$repo_root/evals/out/flash_gate_spec" \
EVAL_ENV_A="QWEN35_FLASH=0" EVAL_ENV_B="QWEN35_FLASH=1" EVAL_MODEL="$model" \
    "$repo_root/evals/eval_quality.sh" "$eval_ppl" "$eval_ppl" || rc=1

echo "===== Pass 3: batched-prefill causality  (q4 window ring, N=384 > WMAX) ====="
caus_out="$repo_root/evals/out/causality_gate.log"
mkdir -p "$(dirname "$caus_out")"
# The test SKIPs (exit 0) without a CUDA forward, so require its PASS verdict line.
if QWEN35_KV_Q4="${QWEN35_KV_Q4:-1}" PN_QWEN35_MODEL="$model" \
        "$causality" >"$caus_out" 2>&1 && grep -q "CAUSALITY path=cuda.*PASS" "$caus_out"; then
    grep -E "CAUSALITY" "$caus_out"
else
    grep -E "CAUSALITY|SKIP|pos=" "$caus_out"
    echo "GATE: batched prefill is not causal (or did not run) -- see $caus_out"
    rc=1
fi

[ "$rc" -eq 0 ] && echo "flash_spec_gate: PASS" || echo "flash_spec_gate: FAIL"
exit $rc
