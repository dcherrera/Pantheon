## What this changes

<!-- One logical change. A few lines — what, and why. Not an essay.
If this is over 1,000 reviewable lines or 15 files, say why it can't be split. -->

## Evidence

<!-- Forward pass or kernel: parity against the reference path (CPU, or an existing
oracle) — argmax-exact, on a named model and quant. Performance: the command, the
device, before/after in tok/s, and the noise floor. Untested? Say so. -->

## Checklist

- [ ] **I license this contribution under the CLA in [`CLA.md`](../CLA.md)** — my entry
      is in [`.cla/signatures.md`](../.cla/signatures.md). I keep my copyright.
- [ ] Every commit signed off (`git commit -s`) — `./scripts/check-dco.sh`
- [ ] Builds clean: `make USE_BLAS=1 pantheon`, zero warnings under `-Wall -Wextra`
- [ ] `make USE_BLAS=1 test` passes, and new functionality has tests
- [ ] New files carry the SPDX pair and the `@file` header block
- [ ] Docs updated if the public API changed

<!-- Contributing on behalf of an employer that owns your work? Your organization
signs CLA-CORPORATE.md by email instead — see that file. -->
