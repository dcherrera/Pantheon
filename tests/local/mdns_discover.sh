#!/bin/bash
# tests/local/mdns_discover.sh <label> [<label> ...]   (run on the Mac)
# D4 on the LAN: `dns-sd -B _pantheon._tcp` must list every given instance
# (e.g. pantheon-lab-node-1 pantheon-lab-node-2), and each <label>.local must resolve
# to an IPv4 address. Output kept in build/local_mdns/.
cd "$(dirname "$0")/../.." || exit 1
. tests/local/lib.sh
out=build/local_mdns
mkdir -p "$out"

dns-sd -B _pantheon._tcp > "$out/browse.txt" 2>&1 &
pid=$!; sleep 6; { kill "$pid"; wait "$pid"; } 2>/dev/null
cat "$out/browse.txt"
for label in "$@"; do
    lt_check "browse lists $label" grep -q " $label\$" "$out/browse.txt"
    dns-sd -G v4 "$label.local" > "$out/addr-$label.txt" 2>&1 &
    pid=$!; sleep 4; { kill "$pid"; wait "$pid"; } 2>/dev/null
    addr=$(awk -v h="$label.local." '$5 == h {print $6}' "$out/addr-$label.txt" | head -1)
    echo "  $label.local -> ${addr:-?}"
    lt_check "$label.local resolves" [ -n "$addr" ]
    dns-sd -L "$label" _pantheon._tcp > "$out/txt-$label.txt" 2>&1 &
    pid=$!; sleep 4; { kill "$pid"; wait "$pid"; } 2>/dev/null
    lt_check "$label TXT has cards=" grep -q "cards=" "$out/txt-$label.txt"
done
lt_done mdns_discover
