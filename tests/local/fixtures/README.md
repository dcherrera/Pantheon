# Pantheon Local test fixtures — TEST ONLY

**Every key in this directory is a TEST key. It is public, it is in git, and it
must never sign a real catalog.** Production builds do not trust any of them:
`src/local/catalog_keys.h` compiles in only David's keys, and adds
`test_catalog.pub` only when built with `make LOCAL_TEST_CATALOG_KEY=1` (the
on-hardware e2e scripts do that; nothing shipped does).

| File | What |
|---|---|
| `test_catalog.{pub,key}` | stands in for the current catalog key (no password) |
| `test_catalog_backup.{pub,key}` | stands in for the backup key |
| `test_untrusted.{pub,key}` | a key nobody trusts (wrong-key test) |
| `people_registry/keys_v1.json` | a v1 `keys.json` (admin + inference) for the v2 migration; its TEST secrets are `pnk_` + 64 × `1` (admin) and `pnk_` + 64 × `2` (inference) |
| `catalog_*.json` + `.minisig` | signed test manifests (see `make_fixtures.sh`); `catalog_e2e.json` lists the real qwen3-0.6b-Q8_0.gguf (sha256-pinned) with placeholder measured numbers, for the e2e flows; `catalog_unit_types.json` has small generated files, two tagged `"unit_types": ["p150"]`, and a second model with one weights file per unit type (`test_local_unit_types`) |
| `catalog_q38_tier.json` | staging for the Qwen3.8 256k tier (usable plan 5.1): the real `qwen3.8-27b-q8` model plus `qwen3.8-27b-tier256k` (GPTQ Bfp4-FFN trunk, profile `p150-one-256k`, ctx 262,144, K8V4, MTP k1), placeholder measured numbers; `test_tt_q38_mixw_fit` checks its memory model against the fit budget |

Regenerate the signed manifests (not the keys) with `tests/local/fixtures/make_fixtures.sh`
(needs `minisign` on the Mac). The keys are kept stable so the compiled-in test key
does not change.
