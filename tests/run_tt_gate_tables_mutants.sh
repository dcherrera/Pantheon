#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_tt_gate_tables_mutants.sh
#   Mutation reds for the exact card gates' host side: break one piece at a time — the tables
#   (src/models/qwen35moe_gate_tables.c), their fit pricing and bank placement
#   (qwen35moe_tt_fit_plan.c), the alpha/beta reader's tile runs (pn_tt_reader_runs.h) —
#   rebuild tests/test_tt_gate_tables, test_tt_q38_fit_banks and test_tt_reader_runs (binaries
#   deleted first) and require them RED, naming the check that caught it. Every mutated file is
#   restored byte for byte (cmp) on exit. Then the clean build must be GREEN again.
#
#   FOUNDRY=<foundry worktree> bash tests/run_tt_gate_tables_mutants.sh
#
#   Output: build/q38-exact-gates/mutants/ (per-mutant logs, summary.txt). Exit 0 = every
#   mutant red and the clean build green; 1 otherwise; 2 setup error.
#
#   Status: added with the exact card gates.
#   Issues: None
#   Todo:   None
#
set -u

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
FOUNDRY=${FOUNDRY:?set FOUNDRY to the foundry worktree}
OUT=$ROOT/build/q38-exact-gates/mutants
TESTS="tests/test_tt_gate_tables tests/test_tt_q38_fit_banks tests/test_tt_reader_runs"
SUMMARY=$OUT/summary.txt
red=0

mkdir -p "$OUT/orig"
: > "$SUMMARY"
cd "$ROOT" || exit 2

SAVED=()  # "original|copy" pairs, restored on exit whatever happens

restore_all() {
    local pair
    for pair in "${SAVED[@]+"${SAVED[@]}"}"; do
        cp -p "${pair#*|}" "${pair%%|*}"
    done
}
trap restore_all EXIT

build_and_run() {
    local t rc=0
    # shellcheck disable=SC2086
    rm -f $TESTS
    # shellcheck disable=SC2086
    make USE_BLAS=1 FOUNDRY="$FOUNDRY" $TESTS > "$1.build" 2>&1 || return 3
    : > "$1"
    for t in $TESTS; do
        "./$t" >> "$1" 2>&1 || rc=1
    done
    return "$rc"
}

# mutant <name> <file> <the check expected to catch it> <perl substitution, applied -0pi>
mutant() {
    local name=$1 file=$2 expect=$3 expr=$4 saved rc caught
    saved=$OUT/orig/$name
    cp -p "$file" "$saved"
    SAVED+=("$file|$saved")
    perl -0pi -e "$expr" "$file"
    if cmp -s "$file" "$saved"; then
        printf 'RED-SCRIPT %-24s the mutation did not apply\n' "$name" | tee -a "$SUMMARY"
        red=1
        return
    fi
    build_and_run "$OUT/$name.log"
    rc=$?
    cp -p "$saved" "$file"
    cmp -s "$file" "$saved" || { echo "restore of $file failed" >&2; exit 2; }
    caught=$(grep -cE '^ *FAIL' "$OUT/$name.log" 2>/dev/null)
    if [ "$rc" = 3 ]; then
        printf 'GREEN %-24s the mutant does not compile (%s.build)\n' "$name" "$OUT/$name.log" | tee -a "$SUMMARY"
        red=1
    elif [ "$rc" != 0 ] && grep -qF "$expect" <(grep -E '^ *FAIL' "$OUT/$name.log"); then
        printf 'RED   %-24s %s FAIL line(s), incl. "%s"\n' "$name" "$caught" "$expect" | tee -a "$SUMMARY"
    else
        printf 'GREEN %-24s rc=%s — the mutant SURVIVED (expected "%s")\n' "$name" "$rc" "$expect" | tee -a "$SUMMARY"
        red=1
    fi
}

G=src/models/qwen35moe_gate_tables.c
F=src/models/qwen35moe_tt_fit_plan.c
R=src/accel/tt/pn_tt_reader_runs.h
mutant check_always_clean "$G" "RED gate_table_corrupt" \
    's/(uint64_t pn_qwen35_gate_tables_check\([^{]*\{\n)/$1    return 0u;\n/'
mutant build_head_lane_lost "$G" "the tables build, prove and pass the lane check" \
    's/values\[\(size_t\)h \* CODES \+ code\] = float_bits\(decay\[h\]\);/values[(size_t)h * CODES + code] = float_bits(decay[0]);/'
mutant entry_offset_lost "$G" "the tables build, prove and pass" \
    's/table\[kFtgtHeaderWords \+ index\];/table[index];/'
mutant code_widen_wrong "$G" "the tables build, prove and pass" \
    's/const uint32_t bits = code << 16;/const uint32_t bits = code << 15;/'
mutant layer_stride_lost "$G" "the tables build, prove and pass the lane check" \
    's/return table_value\(tables->decay \+ \(\(size_t\)layer \* tables->heads \+ head\) \*/return table_value(tables->decay + ((size_t)0 * tables->heads + head) */'
mutant priced_beta_lost "$F" "priced at the cap" \
    's/return \(slots \+ stripes\) \* PN_QWEN35_TT_FIT_GATE_TABLE_SLOT/return slots * PN_QWEN35_TT_FIT_GATE_TABLE_SLOT/'
mutant priced_stripe_round_lost "$F" "13 heads price as 16" \
    's/\(\(heads \+ stripes - 1u\) \/ stripes\) \* stripes;/heads;/'
mutant floor_tables_dropped "$F" "the floor grows by the tables" \
    's/out->reserve \+ out->mtp \+ out->gate_tables;/out->reserve + out->mtp;/'
mutant place_keep_ignored "$F" "RED: banks 8 MiB over 128 MiB" \
    's/if \(trial\[best\] < need \+ keep_free\)/if (trial[best] < need)/'
mutant place_not_emptiest "$F" "that bank untouched" \
    's/if \(!used\[b\] && \(best == banks \|\| trial\[b\] > trial\[best\]\)\)/if (!used[b] \&\& best == banks)/'
mutant place_stripes_shared "$F" "32k dump: 8 stripes, every layer over distinct banks" \
    's/used\[best\] = beta_in\[best\] = true;/beta_in[best] = true;/'
mutant place_beta_every_stripe "$F" "each bank" \
    's/need = stripe_bytes \+ \(beta_in\[best\] \? 0u : beta_bytes\);/need = stripe_bytes + beta_bytes;/'
mutant place_refusal_dirties "$F" "free bytes untouched" \
    's/memcpy\(trial, free_bytes, banks \* sizeof\(trial\[0\]\)\);/memcpy(trial, free_bytes, banks * sizeof(trial[0]));\n        if (stripes == 1u) { free_bytes[0] = 1u; }/'
mutant runs_live_edge "$R" "the 48 live right" \
    's/live <= tile \* 32u \+ 16u;/live < tile * 32u + 16u;/'
mutant runs_lower_face_wrong "$R" "the 48 live right" \
    's/offset\[n\] = 2u \* face;/offset[n] = 3u * face;/'

restore_all
trap - EXIT
build_and_run "$OUT/clean.log"
rc=$?
printf 'clean rebuild: rc=%s, %s\n' "$rc" "$(tail -1 "$OUT/clean.log")" | tee -a "$SUMMARY"
[ "$rc" = 0 ] || red=1
[ "$red" = 0 ] && echo "ALL MUTANTS RED, CLEAN GREEN" | tee -a "$SUMMARY"
exit "$red"
