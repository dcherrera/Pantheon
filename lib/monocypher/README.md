# Monocypher (vendored)

Ed25519 signature checks (RFC 8032, SHA-512) and BLAKE2b for Pantheon Local's
catalog signatures (minisign format). Only `src/local/minisign_verify.c` includes
these headers.

## Provenance

- **Upstream:** https://monocypher.org/ (https://github.com/LoupVaillant/Monocypher)
- **Version:** 4.0.3 (release tarball `monocypher-4.0.3.tar.gz`)
- **Source URL:** https://monocypher.org/download/monocypher-4.0.3.tar.gz
- **Tarball sha256:** `8cc9bc341a66249016db9bd70e9142d8d0aef9945973744b1ac05dbc55d8ee66`
- **Copied from:** GiTeam `app/lib/monocypher/` (same four files, same hashes; GiTeam
  fetched the tarball 2026-10-01 and cross-checked it against the GitHub tag `4.0.3`).
- **License:** CC0-1.0 OR BSD-2-Clause (`LICENSE`, upstream `LICENCE.md`).

| File | Upstream path | sha256 |
|---|---|---|
| `monocypher.c` | `src/monocypher.c` | `57eb914fc88136119bd41655cccb8c250048bf54d470540625186f8ab16f64be` |
| `monocypher.h` | `src/monocypher.h` | `c494da712122da7ff679fdcf318a5317e84972b6c950fe9d896212947797facd` |
| `monocypher-ed25519.c` | `src/optional/monocypher-ed25519.c` | `60fce3578fb00b00da96490653d993c4cb427b1e1be38183285c66e04d22cc18` |
| `monocypher-ed25519.h` | `src/optional/monocypher-ed25519.h` | `abc4fad381879f5c29176ebe014b9189956b3dfe0a3e36459b6990bc57212380` |

## Build

Compiled into the CLI with the serve (`MONOCYPHER_SRC` in the Makefile), with the
project's flags. Portable C99, no SIMD, so it runs on a no-AVX host.

## Updating

Do not hand-edit these files. Re-fetch the release tarball, verify it against the
GitHub tag, copy the four files flat into this directory, and update the table above.
