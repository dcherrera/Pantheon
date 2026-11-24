#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# tt_oncard_golden.sh
#   The knob-off golden digests of tests/test_tt_oncard_dispatch.c: that same test file, built
#   against the BASE Pantheon source (default bdccd110, the S1 tip on main with split-K), prints the
#   digests of the replay programs the base code uploads. The test asserts its own knob-off
#   programs equal them, byte for byte.
#
#   The base tree is exported with `git archive` under build/s2/golden (gitignored), the test
#   file is dropped in as tests/test_tt_group_pull.c so the base Makefile's own fake-card rule
#   builds it, and the binary runs with --digest. Nothing outside build/ is touched.
#
#   FOUNDRY=<foundry checkout> [GOLDEN_MAKE_FLAGS="USE_TT=1"] bash tests/tt_oncard_golden.sh [base revision]
set -euo pipefail
REV=${1:-bdccd110}
: "${FOUNDRY:?set FOUNDRY to the foundry checkout}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
G=$ROOT/build/s2/golden
rm -rf "$G"
mkdir -p "$G"
git -C "$ROOT" archive "$REV" | tar -x -C "$G"
cp "$ROOT/tests/test_tt_oncard_dispatch.c" "$G/tests/test_tt_group_pull.c"
nice -n 10 make -C "$G" ${GOLDEN_MAKE_FLAGS:-USE_TT=1 USE_BLAS=1} FOUNDRY="$FOUNDRY" tests/test_tt_group_pull > "$G/build.log" 2>&1
echo "base $REV ($(git -C "$ROOT" rev-parse "$REV")):"
"$G/tests/test_tt_group_pull" --digest
