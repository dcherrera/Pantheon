# Making and signing a Pantheon Local catalog

A Pantheon Local box only auto-installs models listed in a catalog that is
**signed** with a key compiled into its binary. Anything else can still be run by
hand with `pantheon serve --model …`.

## Add a model

1. Upload the GGUF to our Hugging Face org and note the **commit** of that upload
   (the 40-hex revision, not `main`) and the file's `sha256sum` and size.
2. Measure every profile on the real hardware with the real serve: decode tok/s at
   8k / 14k / 32k, TTFT at a stated prompt size, and the effective context window.
   A profile without measured numbers is refused.
3. Check the model's licence allows us to redistribute it (David decides, per model).
4. Add the model and its profiles to `catalog.json` ([schema](catalog_schema.md))
   and **bump `version`**.
5. `make pantheon && ./pantheon local check-catalog catalog.json`.

## The real catalog

`catalog/catalog.json` is the source of truth. Until the files are uploaded, every
`revision` is a placeholder (40 zeros): it passes `check-catalog`, but
`sign_catalog.sh` refuses it. Replace each with the commit of that file's upload, then sign.
`tests/local/catalog_mac_check.sh` signs a copy with the TEST ONLY key and checks a Mac box
lists it.

## Sign and publish

```sh
tools/local/sign_catalog.sh catalog.json          # current key
tools/local/sign_catalog.sh catalog.json --backup # only if the current key is lost
```

This writes `catalog.json.minisig` (minisign, Ed25519). Publish both files side by
side on the HF org and the teamide.dev mirror. Boxes fetch them at start (and on
`POST /local/catalog/refresh`), verify the signature, refuse an older `version`,
and keep the last accepted copy for offline boots.

## Keys

- Two keys: **current** and **backup**, both compiled into the binary
  (`src/local/catalog_keys.h`). The secret keys stay offline with David; never on the
  VPS, in CI or in a repo.
- If the current key is lost or leaked, sign with the backup; the next release drops
  the old key. There is no scheduled rotation.
- Until the real public keys are in `catalog_keys.h`, a build trusts no key and
  refuses every catalog. `tests/local/fixtures/` holds **test-only** keys, trusted
  only by `make LOCAL_TEST_CATALOG_KEY=1` builds.
