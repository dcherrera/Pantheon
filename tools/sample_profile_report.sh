#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# sample_profile_report.sh <binary> <PN_PROFILE dump> [top N]
#   Fold a sample_profiler dump (ip tid count caller0..3) into per-thread, per-function and
#   per-call-chain totals with addr2line.
set -u
BIN=$1; DUMP=$2; TOP=${3:-60}
total=$(awk '!/^#/ {s+=$3} END {print s+0}' "$DUMP")
awk '!/^#/ {print $1; print $4; print $5; print $6; print $7}' "$DUMP" | sort -u > "$DUMP.addrs"
addr2line -f -C -e "$BIN" $(cat "$DUMP.addrs") | paste - - | awk '{print $1}' > "$DUMP.funcs"
paste "$DUMP.addrs" "$DUMP.funcs" > "$DUMP.map"
echo "== samples $total (ms of CPU), by thread"
awk '!/^#/ {t[$2]+=$3} END {for (k in t) printf "%8d  tid %s\n", t[k], k}' "$DUMP" | sort -rn
echo "== by thread + function"
awk 'NR==FNR {f[$1]=$2; next} !/^#/ {c[$2" "f[$1]]+=$3} END {for (k in c) printf "%8d  %s\n", c[k], k}' \
    "$DUMP.map" "$DUMP" | sort -rn | head -"$TOP"
echo "== by thread + call chain (leaf < caller < caller < caller)"
awk 'NR==FNR {f[$1]=$2; next} !/^#/ {k=$2" "f[$1]" < "f[$4]" < "f[$5]" < "f[$6]; c[k]+=$3}
     END {for (k in c) printf "%8d  %s\n", c[k], k}' "$DUMP.map" "$DUMP" | sort -rn | head -"$TOP"
