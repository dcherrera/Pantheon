#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# run_red.sh — the red proof for tests/test_qwen35_prefill_causality.
#
# A green causality gate means nothing until it has been seen to go red. This builds the test
# twice from copies of this working tree under build/causality_red/ — once as is, once with a
# patch from this directory applied to the COPY (production sources are never touched) — and
# requires every arm to land where it must:
#
#   tt    green           default config (chunkwise GDN, tall 64-band passes,
#                         argmax criterion)                                     PASS
#         green-strict    STRICT = PN_TT_PREFILL_GDNC=0 PN_TT_PREFILL_ROWS=32:
#                         per-position GDN, 32-row passes, digest criterion     PASS
#         red-patch       tt_chunk_attention_causal_off.patch: the chunk
#                         attention's causal tiles are all live, so each row
#                         reads its own pass's future                           FAIL
#         red-patch-strict  the same under STRICT                               FAIL
#         red-env         STRICT + PN_TT_RED=prefill_causal on the green binary:
#                         the per-row GDN and attention loops run backwards     FAIL
#
#   STRICT needs PN_TT_PREFILL_ROWS=32: the per-position fallback (GDNC=0, prefill_causal)
#   addresses one band, so a wider pass declines — and the test fails a declined or aborted
#   prefill (pn_qwen35moe_prefill_tt_last_batched), because an aborted pass still returns
#   per-token logits. Before that check both arms ran no batched pass and scored green.
#   cuda  green           default (flash prefill)                               PASS
#         green-host      QWEN35_FLASH=0 (host-softmax batch attention)         PASS
#         red-patch       cuda_host_attention_reads_future.patch with
#                         QWEN35_FLASH=0: every row attends the whole batch     FAIL
#
# A red arm only counts when the test printed a causality FAIL verdict — a refusal, a crash
# or a load failure is not red, it is broken.
#
# usage: tests/causality_red/run_red.sh tt|cuda <model.gguf> [extra make vars...]
#   tt needs PN_TT_ARTIFACT (and whatever card env the box uses) in the environment.
set -u
backend="${1:?tt|cuda}"
model="${2:?model.gguf}"
shift 2
repo="$(cd "$(dirname "$0")/../.." && pwd)"
out="$repo/build/causality_red"
foundry=""
for candidate in "$repo/foundry" "$repo/../foundry"; do
    [ -f "$candidate/tensor/tensor.h" ] && { foundry="$(cd "$candidate" && pwd -P)"; break; }
done
[ -n "$foundry" ] || { echo "run_red: foundry not found"; exit 2; }
case "$backend" in
    tt)   flags=(USE_TT=1);   patch_file=tt_chunk_attention_causal_off.patch ;;
    cuda) flags=(USE_CUDA=1); patch_file=cuda_host_attention_reads_future.patch ;;
    *)    echo "run_red: backend must be tt or cuda"; exit 2 ;;
esac
bin=tests/test_qwen35_prefill_causality

build_copy() {  # name patch|"" [make vars...]
    local name="$1" patch_name="$2" dir="$out/$1"
    shift 2
    rm -rf "$dir" && mkdir -p "$dir" || return 1
    (cd "$repo" && git ls-files -co --exclude-standard -z | grep -zv '^build/' |
        tar --null -T - -cf -) | tar -xf - -C "$dir" || return 1
    if [ -n "$patch_name" ]; then
        patch -s -p1 -d "$dir" < "$repo/tests/causality_red/$patch_name" || return 1
    fi
    make -C "$dir" FOUNDRY="$foundry" "${flags[@]}" "$@" "$bin" > "$dir/build.log" 2>&1 ||
        { echo "run_red: build of $name failed, see $dir/build.log"; return 1; }
}

run_arm() {  # name binary expect(pass|fail) env...
    local name="$1" binary="$2" expect="$3" output rc verdict
    shift 3
    output="$(env PN_QWEN35_MODEL="$model" "$@" "$binary" 2>&1)"
    rc=$?
    printf '%s\n' "$output" > "$out/$name.log"
    verdict="$(printf '%s\n' "$output" | grep -E 'CAUSALITY path=' | tail -1)"
    echo "$name: rc=$rc ${verdict:-<no verdict line>}"
    if [ "$expect" = pass ]; then
        [ "$rc" = 0 ] && [[ "$verdict" == *PASS ]] && return 0
        echo "ARM $name: expected PASS"
    else
        [ "$rc" != 0 ] && [[ "$verdict" == *FAIL ]] && return 0
        echo "ARM $name: expected a causality FAIL — the gate is vacuous for this mutation"
    fi
    return 1
}

mkdir -p "$out"
build_copy green "" "$@" || exit 2
build_copy red "$patch_file" "$@" || exit 2
# Two arms that are the same bytes are a broken build, never a result (CLAUDE.md, A/B rule).
"$repo/tests/ab_build_arm.sh" --compare "$out/green/$bin" "$out/red/$bin" || exit 2

fail=0
if [ "$backend" = tt ]; then
    strict=(PN_TT_PREFILL_GDNC=0 PN_TT_PREFILL_ROWS=32)
    run_arm tt-green            "$out/green/$bin" pass                         || fail=1
    run_arm tt-green-strict     "$out/green/$bin" pass "${strict[@]}"          || fail=1
    run_arm tt-red-patch        "$out/red/$bin"   fail                         || fail=1
    run_arm tt-red-patch-strict "$out/red/$bin"   fail "${strict[@]}"          || fail=1
    run_arm tt-red-env          "$out/green/$bin" fail "${strict[@]}" \
        PN_TT_RED=prefill_causal                                               || fail=1
else
    run_arm cuda-green          "$out/green/$bin" pass                         || fail=1
    run_arm cuda-green-host     "$out/green/$bin" pass QWEN35_FLASH=0          || fail=1
    run_arm cuda-red-patch      "$out/red/$bin"   fail QWEN35_FLASH=0          || fail=1
fi
[ "$fail" = 0 ] && echo "RED PROOF ($backend): every arm behaved" ||
    echo "RED PROOF ($backend): FAILED"
exit "$fail"
