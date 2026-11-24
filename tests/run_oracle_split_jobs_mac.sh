#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_oracle_split_jobs_mac.sh — qwen36_oracle_dump --jobs with split-layer bands gives the same
# bytes as one run a sequence (H collection band by band).
#
#   tests/run_oracle_split_jobs_mac.sh <model.gguf> <out_dir> <tokens>...   (one file a sequence)
#
# Bands 0:SPLIT then SPLIT:LAYERS (default 4 and 8: the Qwen3.8 N8 GGUF), CPU path, every load
# through the memory guard. For each band, the sequences run once each (single runs) and then
# all together in one --jobs load, PN_SPLIT_RESID_* as `%s` templates. Checks:
#   - band-1 logits + records, band-0 records and the residual files: byte-identical;
#   - PN_ACT_STATS (imatrix + an X^T X): the --jobs file holds every sequence (counts exact);
#     its sums equal the per-sequence files merged, to rounding (the merge adds in another order);
#   - RED (harness): a job pointed at another sequence's residual differs. (Code red: a DUMP
#     built without the template expansion refuses the band-1 load, see the forward.)
# Env: GUARD (default ./foundry/metal/run_guarded.sh), DUMP, MERGE, SPLIT, LAYERS, F32 (=1:
# PN_ACT_STATS_XTX_F32). Outputs and logs stay under <out_dir>.
set -uo pipefail

MODEL=$1; OUT=$2; shift 2
TOKENS=("$@")
export GUARD=${GUARD:-./foundry/metal/run_guarded.sh}
DUMP=${DUMP:-./tools/oracle/qwen36_oracle_dump}
MERGE=${MERGE:-./tools/oracle/act_stats_merge}
SPLIT=${SPLIT:-4}; LAYERS=${LAYERS:-8}
FULL0="blk.1.ffn_gate.weight"; FULL1="blk.$((SPLIT + 1)).ffn_gate.weight"
export QWEN35_KV_Q4=0 PN_HOST_ADMIT=0
[ -n "${F32:-}" ] && export PN_ACT_STATS_XTX_F32=1
[ ${#TOKENS[@]} -ge 2 ] && [ -s "$MODEL" ] && [ -x "$DUMP" ] && [ -x "$GUARD" ] || {
    echo "usage: $0 <model.gguf> <out_dir> <tokens> <tokens>... (needs $DUMP and $GUARD)"; exit 2; }
fails=0
ok() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; fails=$((fails + 1)); fi; }

# One guarded load at a time on the whole Mac (gate-4 guard settings).
g() {
    while pgrep -f run_guarded.sh > /dev/null; do sleep 15; done
    GUARD_MIN_FREE_GB=0 GUARD_MIN_DISK_GB=0 GUARD_KILL_FREE_MB=900 GUARD_KILL_SWAP_MB=4096 \
        GUARD_TIMEOUT_SEC=${GUARD_TIMEOUT_SEC:-7200} "$GUARD" "$@"
}

seq_name() { basename "$1" .txt; }

# run <log> <env...> -- <dump args...>
run() {
    local log=$1; shift
    local envs=()
    while [ "$1" != -- ]; do envs+=("$1"); shift; done
    shift
    env "${envs[@]}" bash -c "$(declare -f g); g $DUMP $MODEL $*" > "$log" 2>&1
}

rm -rf "$OUT"
mkdir -p "$OUT"/{single,jobs}/{b0,b1,res,stats} "$OUT/logs" "$OUT/red"
: > "$OUT/jobs_b0.txt"; : > "$OUT/jobs_b1.txt"
for t in "${TOKENS[@]}"; do
    s=$(seq_name "$t"); n=$(wc -w < "$t")
    echo "$t $OUT/jobs/b0/$s $n 0 0" >> "$OUT/jobs_b0.txt"
    echo "$t $OUT/jobs/b1/$s $n 0 0" >> "$OUT/jobs_b1.txt"
done

# ---- single runs: one load a sequence a band ----
for t in "${TOKENS[@]}"; do
    s=$(seq_name "$t"); n=$(wc -w < "$t")
    run "$OUT/logs/single_b0_$s.log" PN_SPLIT_LAYERS=0:$SPLIT \
        PN_SPLIT_RESID_OUT="$OUT/single/res/$s.$SPLIT.f32" \
        PN_ACT_STATS="$OUT/single/stats/$s.b0.mxim" PN_ACT_STATS_FULL=$FULL0 -- \
        "$t" "$OUT/single/b0/$s" --count "$n" || { echo "single band 0 $s failed"; exit 1; }
    run "$OUT/logs/single_b1_$s.log" PN_SPLIT_LAYERS=$SPLIT:$LAYERS \
        PN_SPLIT_RESID_IN="$OUT/single/res/$s.$SPLIT.f32" \
        PN_ACT_STATS="$OUT/single/stats/$s.b1.mxim" PN_ACT_STATS_FULL=$FULL1 -- \
        "$t" "$OUT/single/b1/$s" --count "$n" --full-logits || { echo "single band 1 $s failed"; exit 1; }
done

# ---- --jobs: one load a band, residual files by template ----
run "$OUT/logs/jobs_b0.log" PN_SPLIT_LAYERS=0:$SPLIT \
    PN_SPLIT_RESID_OUT="$OUT/jobs/res/%s.$SPLIT.f32" \
    PN_ACT_STATS="$OUT/jobs/stats/b0.mxim" PN_ACT_STATS_FULL=$FULL0 -- \
    x x --jobs "$OUT/jobs_b0.txt" || { echo "jobs band 0 failed"; exit 1; }
run "$OUT/logs/jobs_b1.log" PN_SPLIT_LAYERS=$SPLIT:$LAYERS \
    PN_SPLIT_RESID_IN="$OUT/jobs/res/%s.$SPLIT.f32" \
    PN_ACT_STATS="$OUT/jobs/stats/b1.mxim" PN_ACT_STATS_FULL=$FULL1 -- \
    x x --jobs "$OUT/jobs_b1.txt" --full-logits || { echo "jobs band 1 failed"; exit 1; }

for t in "${TOKENS[@]}"; do
    s=$(seq_name "$t")
    ok "$s: residual file ($(wc -c < "$OUT/single/res/$s.$SPLIT.f32") B) identical" \
        "cmp -s $OUT/single/res/$s.$SPLIT.f32 $OUT/jobs/res/$s.$SPLIT.f32"
    ok "$s: band-0 records identical" "cmp -s $OUT/single/b0/$s.pos.bin $OUT/jobs/b0/$s.pos.bin"
    ok "$s: band-1 records identical" "cmp -s $OUT/single/b1/$s.pos.bin $OUT/jobs/b1/$s.pos.bin"
    ok "$s: band-1 logits ($(wc -c < "$OUT/single/b1/$s.logits.f32") B) identical" \
        "cmp -s $OUT/single/b1/$s.logits.f32 $OUT/jobs/b1/$s.logits.f32"
done

# ---- act stats: every sequence in the one --jobs file ----
for b in b0 b1; do
    singles=(); for t in "${TOKENS[@]}"; do singles+=("$OUT/single/stats/$(seq_name "$t").$b.mxim"); done
    xs=(); for f in "${singles[@]}"; do xs+=("$f.xtx"); done
    "$MERGE" "$OUT/single/stats/merged.$b.mxim" "${singles[@]}" &&
        "$MERGE" --xtx "$OUT/single/stats/merged.$b.mxim.xtx" "${xs[@]}" ||
        { echo "act_stats_merge failed"; exit 1; }
    ok "$b: --jobs act stats = the per-sequence files merged (imatrix + X^T X; counts exact, sums to 1e-12)" \
        "python3 - $OUT/jobs/stats/$b.mxim $OUT/single/stats/merged.$b.mxim $OUT/jobs/stats/$b.mxim.xtx $OUT/single/stats/merged.$b.mxim.xtx <<'PY'
import struct, sys
import numpy as np
def read(p):
    b = open(p, 'rb').read(); n, tok = struct.unpack_from('<IQ', b, 8); o, ents = 20, {}
    xtx = p.endswith('.xtx')
    for _ in range(n):
        (l,) = struct.unpack_from('<I', b, o); name = b[o + 4:o + 4 + l]; o += 4 + l
        nin, cnt = struct.unpack_from('<IQ', b, o); o += 12; k = nin * nin if xtx else nin
        ents[name] = (cnt, np.frombuffer(b, '<f8', k, o)); o += 8 * k
    return tok, ents
bad = 0
for a, b in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    ta, ea = read(a); tb, eb = read(b)
    bad += ta != tb or set(ea) != set(eb)
    for k in ea:
        ca, pa = ea[k]; cb, pb = eb[k]
        d = float(np.max(np.abs(pa - pb)) / (np.max(np.abs(pb)) or 1.0))
        bad += ca != cb or d > 1e-12
        print('  %s %s: count %d / %d, max diff %.2e of the largest entry' % (a.split('/')[-1], k.decode(), ca, cb, d), file=sys.stderr)
sys.exit(1 if bad else 0)
PY"
done

# ---- RED (harness): the last job reads the first sequence's residual ----
first=$(seq_name "${TOKENS[0]}"); last=$(seq_name "${TOKENS[${#TOKENS[@]}-1]}")
nl=$(wc -w < "${TOKENS[${#TOKENS[@]}-1]}"); nf=$(wc -w < "${TOKENS[0]}")
if [ "$nl" -le "$nf" ]; then
    echo "${TOKENS[${#TOKENS[@]}-1]} $OUT/red/$first $nl 0 0" > "$OUT/jobs_red.txt"
    run "$OUT/logs/red_wrong_resid.log" PN_SPLIT_LAYERS=$SPLIT:$LAYERS \
        PN_SPLIT_RESID_IN="$OUT/jobs/res/%s.$SPLIT.f32" -- x x --jobs "$OUT/jobs_red.txt" --full-logits
    ok "RED: $last fed $first's residual differs from $last's single run" \
        "! cmp -s $OUT/single/b1/$last.logits.f32 $OUT/red/$first.logits.f32"
else
    echo "SKIP  RED wrong residual: list the shortest sequence last"
fi
echo "$0: $fails failed"
exit $((fails > 0))
