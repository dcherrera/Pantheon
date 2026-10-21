#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# The RED PROOF for the batched TT prefill's byte-exactness gate.
#
# A green gate proves nothing until it has been shown to go red — four vacuous gates have
# been caught in this project, including one that was blind to every computed store. So this
# runs the same harness five times on the same card:
#
#   green32   the default 32-row batch                        must be BYTE-IDENTICAL
#   green1    PN_TT_PREFILL_ROWS=1, the sequential computation
#             reached THROUGH the batched code                must be BYTE-IDENTICAL
#   pos       every row appends at the pass's first position   must DIFFER
#   causal    the per-row loop runs in reverse                 must DIFFER
#   rowfill   q/k/v rows 1..M-1 overwritten with row 0         must DIFFER
#
# `green1` is the control that separates the two things that could be wrong: if the batching
# is broken it goes green and `green32` goes red; if the multi-row PLUMBING is broken they
# both go red.
#
# usage: run_tt_prefill_red.sh <model.gguf> <artifact.fttw> [tokens]
set -u
MODEL="${1:?model.gguf}"
ART="${2:?artifact.fttw}"
TOKENS="${3:-100}"
BIN=./tests/test_tt_prefill_parity
COMMON=(PN_TT_ARTIFACT="$ART" PN_TT_ONCARD=norm,swiglu,residual,topk PN_TT_MATMUL_CORES=4)

run_arm() {  # name, expect (identical|differ), extra env...
    local name="$1" expect="$2"; shift 2
    local out rc verdict
    out=$(env "${COMMON[@]}" "$@" "$BIN" "$MODEL" --tokens "$TOKENS" 2>&1)
    rc=$?
    verdict=$(printf '%s\n' "$out" | sed -n 's/^PARITY VERDICT //p')
    printf '%s\n' "$out" | sed -n '/^PARITY/p'
    if [ "$expect" = identical ]; then
        [ "$rc" = 0 ] && [ "$verdict" = "BYTE-IDENTICAL" ] \
            && echo "ARM $name: PASS (identical, as required)" \
            || { echo "ARM $name: FAIL — expected BYTE-IDENTICAL, got rc=$rc '$verdict'"; return 1; }
    else
        [ "$verdict" = "DIFFER" ] \
            && echo "ARM $name: PASS (went red, as required)" \
            || { echo "ARM $name: FAIL — the mutation did NOT change the answer (rc=$rc '$verdict'). The gate is vacuous."; return 1; }
    fi
}

fail=0
echo "=== link before ==="; sudo lspci -vvs 12:00.0 2>/dev/null | grep LnkSta:
run_arm green32 identical                          || fail=1
run_arm green1  identical PN_TT_PREFILL_ROWS=1     || fail=1
run_arm pos     differ    PN_TT_RED=prefill_pos    || fail=1
run_arm causal  differ    PN_TT_RED=prefill_causal || fail=1
run_arm rowfill differ    PN_TT_RED=prefill_rowfill || fail=1
echo "=== link after ==="; sudo lspci -vvs 12:00.0 2>/dev/null | grep LnkSta:
[ "$fail" = 0 ] && echo "RED PROOF: all five arms behaved" || echo "RED PROOF: FAILED"
exit "$fail"
