#!/bin/bash
# tests/local/install_small.sh   (run ON the box, from the repo root)
# F5 on hardware: install the small catalog model (qwen3-0.6b-Q8_0, pinned by
# sha256 in tests/local/fixtures/catalog_e2e.json) from a local HF stand-in,
# through curl, and check the verified file in the store.
#   LT_MODEL_FILE=/path/to/qwen3-0.6b-Q8_0.gguf tests/local/install_small.sh
# Needs a `make LOCAL_TEST_CATALOG_KEY=1` build (the fixture is signed with the
# TEST ONLY key).
cd "$(dirname "$0")/../.." || exit 1
. tests/local/lib.sh
rm -rf "$LT_DIR/state" "$LT_STORE"
lt_hf_start || exit 1
# shellcheck disable=SC2046
lt_start --no-mdns $(lt_catalog_flags) || { lt_hf_stop; exit 1; }
KEY=$(lt_pair)
lt_check "paired" [ -n "$KEY" ]
lt_check "catalog accepted" lt_wait 30 sh -c "curl -s -H 'Authorization: Bearer $KEY' \
    http://127.0.0.1:$LT_PORT/local/catalog | grep -q '\"version\":1'"
lt_post "$KEY" /local/models/install "{\"catalog_id\":\"$LT_MODEL_ID\"}"; echo
start=$(date +%s)
lt_check "install done" lt_wait 900 lt_installed
echo "  install took $(( $(date +%s) - start )) s"
lt_get "$KEY" "/local/models/install/$LT_MODEL_ID"; echo
file=$(ls "$LT_STORE"/$LT_MODEL_ID-*.gguf 2>/dev/null | head -1)
lt_check "file at its store name" [ -n "$file" ]
lt_check "no .part left" [ -z "$(ls "$LT_STORE"/*.part 2>/dev/null)" ]
want=$(sed -n 's/.*"sha256": "\([0-9a-f]*\)".*/\1/p' "$LT_CATALOG" | head -1)
lt_check "sha256 matches the catalog" [ "$(sha256sum "$file" | cut -d' ' -f1)" = "$want" ]
lt_stop
lt_hf_stop
lt_done install_small
