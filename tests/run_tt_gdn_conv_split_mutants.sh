#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_tt_gdn_conv_split_mutants.sh
#   Mutation reds for tests/test_tt_gdn_conv_split.c: break one piece of the CONV split at a
#   time — the production slicer and member count here, foundry's wide column encoder and the
#   kernels' decode in $FOUNDRY — rebuild the test (binary deleted first) and require it RED,
#   naming the check that caught it. Every mutated file is restored byte for byte (checked
#   with cmp) on exit, whatever happens. Then the clean build must be GREEN again.
#
#   FOUNDRY=<foundry worktree> bash tests/run_tt_gdn_conv_split_mutants.sh
#
#   Output: build/q38u-conv/mutants/ (per-mutant logs, summary.txt). Exit 0 = every mutant
#   red and the clean build green; 1 otherwise; 2 setup error.
#
#   Status: added with the wide CONV column slice (PN_TT_GDN_CONV_WIDE).
#   Issues: None
#   Todo:   None
#
set -u

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
FOUNDRY=${FOUNDRY:?set FOUNDRY to the foundry worktree}
OUT=$ROOT/build/q38u-conv/mutants
TEST=tests/test_tt_gdn_conv_split
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
    rm -f "$TEST"
    make USE_BLAS=1 FOUNDRY="$FOUNDRY" "$TEST" > "$1.build" 2>&1 || return 3
    "./$TEST" > "$1" 2>&1
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
    caught=$(grep -c '^FAIL' "$OUT/$name.log" 2>/dev/null)
    if [ "$rc" = 3 ]; then
        printf 'GREEN %-24s the mutant does not compile (%s.build)\n' "$name" "$OUT/$name.log" | tee -a "$SUMMARY"
        red=1
    elif [ "$rc" != 0 ] && grep -qF "$expect" <(grep '^FAIL' "$OUT/$name.log"); then
        printf 'RED   %-24s %s FAIL line(s), incl. "%s"\n' "$name" "$caught" "$expect" | tee -a "$SUMMARY"
    else
        printf 'GREEN %-24s rc=%s — the mutant SURVIVED (expected "%s")\n' "$name" "$rc" "$expect" | tee -a "$SUMMARY"
        red=1
    fi
}

S=src/accel/tt/pn_tt_gdn_conv_split.c
P=$FOUNDRY/tt/kernels/resident_gdn_protocol.h
E=$FOUNDRY/tt/ft_tt_op_gdn.c
mutant narrow_cap_ignored "$S" "narrow (shipped)" \
    's/per <= col_max && cols - per <= col_max/per <= 0xFFFFu && cols - per <= 0xFFFFu/'
mutant asked_ignored "$S" "PN_TT_GDN_CONV_CORES=8" \
    's/\n        members = asked;/\n        members = armed;/'
mutant slice_first_low_byte "$S" "16-way == whole" \
    's/out->col_first = index \* per;/out->col_first = (index * per) & 0xFFu;/'
mutant red_dup_inert "$S" "gdn_conv_col_dup at 16" \
    's/        out->col_first = 0u;\n        break;/        break;/'
mutant red_drop_inert "$S" "gdn_conv_col_drop at 16" \
    's/out->col_first -= depth;/out->col_first -= 0u;/'
mutant red_wrap_inert "$S" "gdn_conv_col_wrap at 16" \
    's/&= PN_TT_GDN_COL_BYTE_MAX;/&= 0xFFFFu;/g'
mutant decode_first_high_lost "$P" "16-way == whole" \
    's/shape\.col_first = shape\.head_first \|\s+\(ftg_byte\(parameters\[kGdnConvTileCols\], kGdnByteColFirstHigh\) << 8\);/shape.col_first = shape.head_first;/'
mutant decode_count_high_lost "$P" "explicit [0, 320)" \
    's/const uint32_t count = shape\.head_count \|\s+\(ftg_byte\(parameters\[kGdnConvTileCols\], kGdnByteColCountHigh\) << 8\);/const uint32_t count = shape.head_count;/'
mutant decode_cols_unmasked "$P" "16-way == whole" \
    's/parameters\[kGdnConvTileCols\] & \(uint32_t\)kGdnConvTileColsMask;/parameters[kGdnConvTileCols];/'
mutant encode_first_high_lost "$E" "16-way == whole" \
    's/\(\(\(request->col_first >> 8\) & 0xFFu\) << 16\)/0u/'
mutant encode_count_high_lost "$E" "explicit [0, 320)" \
    's/\(\(\(request->col_count >> 8\) & 0xFFu\) << 24\)/0u/'
mutant encode_narrow_moved "$E" "every 4-way member" \
    's/\(\(\(request->col_first >> 8\) & 0xFFu\) << 16\)/((request->col_first & 0xFFu) << 16)/'

mutant fits_granule_lost "$S" "18-column slices never split" \
    's/cols % members == 0u && per % granule == 0u &&/cols % members == 0u \&\& per % 2u == 0u \&\&/'
mutant encode_granule_lost "$E" "[300, 318) is refused" \
    's/request->col_count != 0u && !conv_slice_is_granular\(request\)/request->col_count != 0u \&\& !conv_slice_is_granular(request) \&\& 0/'

mutant wide_default_off "$S" "unset or empty is wide" \
    's/if \(!value \|\| !\*value\)\n        return true;/if (!value || !*value)\n        return false;/'

restore_all
trap - EXIT
build_and_run "$OUT/clean.log"
rc=$?
printf 'clean rebuild: rc=%s, %s\n' "$rc" "$(tail -1 "$OUT/clean.log")" | tee -a "$SUMMARY"
[ "$rc" = 0 ] || red=1
[ "$red" = 0 ] && echo "ALL MUTANTS RED, CLEAN GREEN" | tee -a "$SUMMARY"
exit "$red"
