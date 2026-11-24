# Contributing to Pantheon

Thanks for your interest in Pantheon. This document covers the basics.

## Licensing of contributions — read this first

Pantheon ships under **AGPL-3.0** (see [`LICENSE`](./LICENSE)).

> **Contributions are made under the Contributor License Agreement in
> [`CLA.md`](./CLA.md)**, which grants me a broad copyright and patent license,
> including the right to sublicense your contribution under any terms and to
> transfer that right along with the project.

**You keep your copyright.** The CLA is a license, not an assignment — nothing
is transferred away from you. Your contribution stays yours to use, publish,
sell, or license to anyone else, anywhere, forever.

Signing takes one commit: add your entry to
[`.cla/signatures.md`](./.cla/signatures.md) in your first pull request. Nothing
to print, scan, or email. Organizations whose employees contribute on company
time sign [`CLA-CORPORATE.md`](./CLA-CORPORATE.md) by email instead.

If you would rather not sign, open an issue describing the change instead of a
pull request. An idea costs nothing to accept.

### Sign your commits (DCO)

Every commit must carry a `Signed-off-by` line certifying the
[Developer Certificate of Origin](https://developercertificate.org/) — that you
wrote the code, or otherwise have the right to submit it:

```bash
git commit -s -m "your message"
```

which appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

Use your real name and a working email address — the same address as your CLA
roster entry, since that is what CI matches on. If you are contributing work your
employer owns, confirm you actually have the right to do so before you sign off.

Both checks run in CI on every pull request, and you can run them yourself first:

```bash
./scripts/check-dco.sh     # every commit signed off, matching its author
./scripts/check-cla.sh     # every author on the CLA roster
./scripts/check-headers.sh # every source file you ADD has SPDX + the @file block
```

### License header on new files

Every source file starts with:

```c
/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
```

Add it to new files you create. Do not edit the copyright line on existing files
— you keep your copyright in what you contribute without needing a line in the
header, and a per-file list of names becomes unmaintainable fast. `git log` and
`.cla/signatures.md` are the authoritative record of who wrote what.

## Getting Started

Pantheon builds on Foundry. In the public repository it is a git submodule at
`./foundry`; clone with
`git clone --recursive https://git.teamide.dev/David/Pantheon.git`, or run
`git submodule update --init` in an existing clone. The Makefile uses, in order,
`FOUNDRY=<path>`, then `./foundry` (the submodule, or a symlink in a development
checkout), then `../foundry`
(a sibling checkout).

```bash
make USE_BLAS=1 pantheon     # build the CLI
make USE_BLAS=1 test         # build + run the test suites
make DEBUG=1 USE_BLAS=1 pantheon
make clean
```

| Flag | What | When |
|------|------|------|
| `USE_BLAS=1` | SGEMV/SGEMM via Accelerate or OpenBLAS | always |
| `USE_THREADS=1` | pthread pool for parallel ops | multi-core inference |
| `USE_METAL=1` | Metal GPU compute | macOS / Apple Silicon |
| `USE_CUDA=1` | CUDA via `dlopen` (cuBLAS + embedded PTX) | Linux + NVIDIA |

## Code Style

- **C11** (`-std=c11`); no dependencies beyond Foundry, libc, libm, pthreads and libdl, plus vendored Monocypher (`lib/monocypher/`, BSD-2-Clause, Pantheon Local catalog signatures; see NOTICE). Pantheon Local also runs the system `curl` at runtime for model downloads (not linked or bundled)
  (BLAS and accelerator frameworks are opt-in build flags)
- **`pn_` prefix** on all Pantheon symbols
- Errors: return `NULL` or an error code — no `exit()` in library code
- Opaque types in the public API (`include/pantheon.h`); concrete structs stay
  internal
- Callers free returned objects via `pn_free()`
- `snprintf`, never `sprintf`
- No warnings under `-Wall -Wextra`

Broader conventions live in `coding-best-practices.md`.

## Testing

Tests use Foundry's `lib/test.h` framework and live in `tests/`. Add a matching
target in the `Makefile`.

**If you change a forward pass or a kernel, show numerical parity.** The bar is
argmax-exact against the reference path (CPU, or an existing oracle) on a real
model — not "it produces plausible text." Include how you measured it in the PR.

## Pull Requests

Code is reviewed against
[`docs/coding-best-practices.md`](docs/coding-best-practices.md), cited by rule
number. This section covers the PR itself.

**One logical change per PR.**

**Size.** Reviewable lines = added + deleted, excluding generated files, pure
renames, and vendored imports.

| Reviewable lines | |
|---|---|
| ≤ 400 | normal |
| 400–1,000 | expect a request to split |
| > 1,000, or > 15 files | state in one line why it can't be split |

**Exempt from the size limits — but must be alone in their own PR:**

1. **Generated sources.** State the regeneration command, toolchain version, and
   target arch. Reviewed by rebuilding, not by reading.
2. **Pure moves and renames.** No content change in the same commit.
3. **Mechanical sweeps.** Include the command that reproduces the diff.
4. **Vendored or backend imports.**

**Evidence.**

- **Performance claim** — the command, the device, before/after in tok/s, and the
  noise floor. A multiplier on its own is not a result.
- **Forward pass or kernel** — parity against the reference path, argmax-exact or
  ulp-0, naming the model and quantization.
- **A test you call green** — show it can go red (rule 8). Delete the `.o` *and*
  the test binary first, or you may be running the old one.
- **Untested** — say so plainly rather than leaving it implied.

**Keep it short.** Title is `scope: imperative summary`, lowercase scope, ≤72
characters. The body says what changed and why in a few lines. Commit messages
the same: a subject line, and a body only when the reason isn't visible in the
diff. Long rationale belongs in a spec file, not a commit message. No tool,
assistant, or generator attribution anywhere — title, body, or comments.

**Review comments carry a severity prefix:**

- `blocker:` — must change before merge
- `question:` — answer before merge
- `nit:` — optional, never blocks

Unprefixed reads as `nit:`. The author does not resolve a `blocker:` thread — the
reviewer does. Approval goes stale when new commits land, unless they are a
rebase or a fixup.

**Merge bar:** builds clean with zero warnings, tests pass, DCO and CLA checks
green, and the code meets the practices doc.

## Reporting Bugs

Open an issue with:

- what you expected, and what actually happened
- minimal reproduction steps
- the model and quantization involved (many bugs are format-specific)
- your platform, compiler version, and GPU + driver if relevant

Bugs that only appear at a particular context depth, batch size, or concurrency
level are common here — please say which, since it is usually the whole clue.
