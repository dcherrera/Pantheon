#!/bin/bash
# tests/local/reboot_reload.sh <profile_id>   (run ON the box, after install_small.sh)
# G5 on hardware: load the installed small model with <profile_id> (k80-one on a
# K80 box, p150-one on a p150 box), restart `pantheon local` (or the systemd unit
# when PN_LOCAL_UNIT names it), and check the same model is back on the same units.
cd "$(dirname "$0")/../.." || exit 1
. tests/local/lib.sh
profile=${1:?profile id}
units_of() { lt_get "$KEY" /local/models/loaded | sed -n 's/.*"units":\[\([0-9,]*\)\].*/\1/p'; }
restart() {
    if [ -n "${PN_LOCAL_UNIT:-}" ]; then sudo systemctl restart "$PN_LOCAL_UNIT"
    else lt_stop; lt_start --no-mdns $(lt_catalog_flags); fi
}

lt_hf_start || exit 1
# shellcheck disable=SC2046
lt_start --no-mdns $(lt_catalog_flags) || exit 1
KEY=$(lt_pair)
lt_wait 30 sh -c "curl -s -H 'Authorization: Bearer $KEY' http://127.0.0.1:$LT_PORT/local/catalog | grep -q installed\\\":true"
out=$(lt_post "$KEY" /local/models/load "{\"catalog_id\":\"$LT_MODEL_ID\",\"profile_id\":\"$profile\"}")
echo "$out"
lt_check "loaded with $profile" grep -q '"units"' <<< "$out"
before=$(units_of)
echo "  units before: [$before]"
restart
lt_check "model back after restart" lt_wait 300 sh -c "curl -s -H 'Authorization: Bearer $KEY' \
    http://127.0.0.1:$LT_PORT/v1/models | grep -q '\"id\":\"$LT_MODEL_ID\"'"
after=$(units_of)
echo "  units after: [$after]"
lt_check "same units" [ "$before" = "$after" ]
lt_check "registry has it on those dies" sh -c "curl -s -H 'Authorization: Bearer $KEY' \
    http://127.0.0.1:$LT_PORT/admin/models | grep -q '\"dies\":\[$after\]'"
lt_post "$KEY" /admin/models/unload "{\"name\":\"$LT_MODEL_ID\"}"; echo
lt_stop
lt_hf_stop
lt_done reboot_reload
