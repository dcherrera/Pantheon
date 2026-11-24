#!/bin/bash
# tests/local/hw_report_bank.sh <k80|p150> <expected units>
# B5 on hardware: start `pantheon local`, read /local/hardware, check the card
# tree. On the K80 bank restrict CUDA_VISIBLE_DEVICES to idle dies first; the
# expected unit count is then the number of visible dies.
cd "$(dirname "$0")/../.." || exit 1
. tests/local/lib.sh
kind=${1:?k80|p150}; units=${2:?expected unit count}
unit_type=$([ "$kind" = k80 ] && echo k80-gpu || echo p150)
cards=$([ "$kind" = k80 ] && echo $(( (units + 1) / 2 )) || echo "$units")

lt_start --no-mdns || exit 1
KEY=$(lt_pair)
lt_get "$KEY" /local/hardware > "$LT_DIR/hardware.json"
lt_stop
cat "$LT_DIR/hardware.json"; echo

f=$LT_DIR/hardware.json
lt_check "$units $unit_type units"   [ "$(lt_count "$f" "\"type\":\"$unit_type\"")" = "$units" ]
lt_check "$cards $kind cards"        [ "$(lt_count "$f" "\"type\":\"$kind\"")" = "$cards" ]
lt_check "host ram reported"         grep -q '"ram":{"total":[1-9]' "$f"
lt_check "disk free reported"        grep -q '"disk_free":[1-9]' "$f"
lt_check "pantheon backend reported" grep -q '"backend":"' "$f"
if [ "$(uname -m)" = x86_64 ]; then
    want_avx=$(grep -qw avx /proc/cpuinfo && echo true || echo false)
    lt_check "avx flag matches /proc/cpuinfo ($want_avx)" grep -q "\"avx\":$want_avx" "$f"
fi
lt_done hw_report_bank
