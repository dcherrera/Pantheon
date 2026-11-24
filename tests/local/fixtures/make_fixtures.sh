#!/bin/bash
# tests/local/fixtures/make_fixtures.sh — re-sign the TEST ONLY manifests with the
# TEST ONLY keys in this directory (needs minisign on the Mac). The keys are not
# regenerated. catalog_v2_tampered.json gets catalog_v2.json's signature on
# purpose: it must fail to verify (red arm).
set -e
cd "$(dirname "$0")"
sign() {   # sign <key> <file> <out.minisig>
    minisign -S -s "$1" -m "$2" -x "$3" -t "pantheon local TEST ONLY $(basename "$2")" < /dev/null
}
sign test_catalog.key catalog_v1.json catalog_v1.json.minisig
sign test_catalog.key catalog_v2.json catalog_v2.json.minisig
sign test_catalog.key catalog_e2e.json catalog_e2e.json.minisig
sign test_catalog.key catalog_unit_types.json catalog_unit_types.json.minisig
sign test_catalog.key catalog_q38_tier.json catalog_q38_tier.json.minisig
sign test_catalog_backup.key catalog_v2.json catalog_v2.backup.minisig
minisign -S -l -s test_catalog.key -m catalog_v1.json -x catalog_v1.legacy.minisig \
    -t "pantheon local TEST ONLY legacy" < /dev/null
sign test_untrusted.key catalog_v1.json catalog_v1.untrusted.minisig
cp catalog_v2.json.minisig catalog_v2_tampered.json.minisig
for f in catalog_v1.json catalog_v2.json catalog_e2e.json catalog_unit_types.json catalog_q38_tier.json; do
    minisign -V -p test_catalog.pub -m "$f" -x "$f.minisig" > /dev/null
done
minisign -V -p test_catalog_backup.pub -m catalog_v2.json -x catalog_v2.backup.minisig > /dev/null
echo "fixtures signed"
