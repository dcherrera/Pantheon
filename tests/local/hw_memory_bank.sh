#!/bin/bash
# tests/local/hw_memory_bank.sh   (run ON the K80 bank, from the repo root, detached)
# H2n real memory on hardware: start a second `pantheon local` (all dies visible,
# nothing loaded) beside whatever already runs, read /local/hardware, and print
# each unit's report next to nvidia-smi's per-GPU use. Loads nothing, touches no
# other process. LT_PORT / LT_DIR pick the port and a private state dir.
# Writes $LT_DIR/report.txt and $LT_DIR/done.
cd "$(dirname "$0")/../.." || exit 1
. tests/local/lib.sh
rm -f "$LT_DIR/done"
mkdir -p "$LT_DIR"
out=$LT_DIR/report.txt
: > "$out"

lt_start --no-mdns || { echo "start failed" >> "$out"; echo 1 > "$LT_DIR/done"; exit 1; }
KEY=$(lt_pair)
for i in 1 2; do
    t0=$(date +%s%N)
    lt_get "$KEY" /local/hardware > "$LT_DIR/hardware_$i.json"
    t1=$(date +%s%N)
    echo "read $i: $(( (t1 - t0) / 1000000 )) ms" >> "$out"
done
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    > "$LT_DIR/smi.csv"
lt_stop

python3 - "$LT_DIR/hardware_2.json" "$LT_DIR/smi.csv" >> "$out" <<'EOF'
import json, sys
hw = json.load(open(sys.argv[1]))
smi = {int(r[0]): (int(r[1]), int(r[2])) for r in
       (l.split(',') for l in open(sys.argv[2]) if l.strip())}
mib = 1 << 20
print("unit status        live  device_MiB  ours_MiB  other_MiB  free_MiB | smi_used_MiB smi_total_MiB")
for c in hw["cards"]:
    for u in c["units"]:
        su, st = smi.get(u["id"], (-1, -1))
        print(f'{u["id"]:>4} {u["status"]:<13} {str(u["mem_live"]):<5} '
              f'{u["mem_device_total"] // mib:>10} {u["mem_used_self"] // mib:>9} '
              f'{u["mem_used_other"] // mib:>10} {u["mem_free"] // mib:>9} | {su:>12} {st:>13}')
EOF
echo 0 > "$LT_DIR/done"
