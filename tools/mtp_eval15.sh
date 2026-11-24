#!/usr/bin/env bash
# mtp_eval15.sh <head.npz|official> <k> <tau> <out> — tools/mtp_accept.py on the four held-out
# Ornith 1.5 greedy dumps (models/dflash-dumps15), under the Mac memory guard.
set -u
M=models; H=$1; K=$2; T=$3; O=$4
HEAD=""; [ "$H" != official ] && HEAD="--head $H"
GUARD_KILL_SWAP_MB=${GUARD_KILL_SWAP_MB:-6500} GUARD_MIN_DISK_GB=8 GUARD_MIN_FREE_GB=5 GUARD_TIMEOUT_SEC=3000 \
  ${GUARD:-../tt-accept-foundry/metal/run_guarded.sh} ${PY:-python3} tools/mtp_accept.py \
  --gguf $M/ornith-1.5-9b/Ornith-1.5-9B-Q8_0.gguf $HEAD \
  --dump $M/dflash-dumps15/capital.bin --prompt-len 17 --dump $M/dflash-dumps15/math.bin --prompt-len 50 \
  --dump $M/dflash-dumps15/code.bin --prompt-len 34 --dump $M/dflash-dumps15/explain.bin --prompt-len 25 \
  --k $K --tau $T > $O 2>&1
grep -A7 "^ALL" $O
