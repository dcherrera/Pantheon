#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# ab_build_arm.sh — build ONE arm of an A/B and prove the binary really changed.
#
# WHY THIS EXISTS
#   A kernel/engine A/B is only a measurement if the two arms are two different
#   binaries.  A build that quietly hands back the PREVIOUS binary makes the A/B
#   read 1.000x and makes a bit-exactness gate pass vacuously — both of which
#   look like an honest "no change" rather than a broken build.  That has
#   already happened once here: regenerating the generated PTX header
#   foundry/cuda/fc_cuda_ptx.h changed no .c file, so make said "is up to date"
#   and returned a byte-identical harness binary.
#
#   The Makefile now depends on headers, so that specific hole is closed.  This
#   script is the belt to that braces: make can still be defeated by an
#   mtime tie (a restored/copied file landing in the same second as the binary),
#   by a clock skew over a network mount, by a flag change make cannot see, or
#   by the next generated artefact nobody wired up.  Identical bytes out of two
#   arms is ALWAYS an error, never a result.
#
# USAGE
#   tests/ab_build_arm.sh <arm-name> <make-target> [MAKE VARS/ARGS...]
#       Deletes the target, builds it, stores it under $AB_DIR/<arm-name>/, and
#       FAILS if the result is byte-identical to any arm already stored.
#
#   tests/ab_build_arm.sh --compare <fileA> <fileB>
#       Just compares two binaries; fails if identical.
#
#   Env: AB_DIR (default build/ab) — where arms are kept.
#
# EXIT CODES
#   0  built, and this arm differs from every other stored arm
#   1  usage / build failure
#   2  IDENTICAL BINARY — the build produced no change (this is the bug)
#
# EXAMPLE (the standard shape for a kernel A/B)
#   git checkout <base>         && tests/ab_build_arm.sh base tests/test_spec_qwen35 USE_CUDA=1
#   git checkout <new>          && tests/ab_build_arm.sh new  tests/test_spec_qwen35 USE_CUDA=1
#   # then run each arm's stored binary, not whatever is in the worktree.

set -uo pipefail

AB_DIR="${AB_DIR:-build/ab}"

digest() {
    if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | awk '{print $1}'
    else md5 -q "$1"; fi
}

fail_identical() {
    echo "" >&2
    echo "FAIL: the build produced a byte-identical binary." >&2
    echo "      $1" >&2
    echo "      $2" >&2
    echo "" >&2
    echo "      This is a BROKEN BUILD, not a measurement. A change was made and" >&2
    echo "      the compiler emitted the same bytes, so any A/B on these two" >&2
    echo "      binaries reads 1.000x and any bit-exactness gate passes" >&2
    echo "      vacuously. Find out why the change did not reach the binary" >&2
    echo "      (stale generated header? mtime tie? wrong tree? wrong flags?)" >&2
    echo "      before you record a single number." >&2
    exit 2
}

if [ "${1:-}" = "--compare" ]; then
    [ $# -eq 3 ] || { echo "usage: $0 --compare <fileA> <fileB>" >&2; exit 1; }
    [ -f "$2" ] && [ -f "$3" ] || { echo "missing file" >&2; exit 1; }
    if cmp -s "$2" "$3"; then fail_identical "$2  $(digest "$2")" "$3  $(digest "$3")"; fi
    echo "OK: $2 and $3 differ ($(digest "$2") vs $(digest "$3"))"
    exit 0
fi

[ $# -ge 2 ] || { sed -n '2,40p' "$0"; exit 1; }

ARM="$1"; TARGET="$2"; shift 2

# Delete first: never let make's opinion of "up to date" decide whether this arm
# gets a fresh link.
rm -f "$TARGET"

echo "== building arm '$ARM': make $TARGET $*"
if ! make "$TARGET" "$@"; then
    echo "FAIL: build of $TARGET failed" >&2
    exit 1
fi
[ -f "$TARGET" ] || { echo "FAIL: make reported success but $TARGET does not exist" >&2; exit 1; }

mkdir -p "$AB_DIR/$ARM"
BASE="$(basename "$TARGET")"
DEST="$AB_DIR/$ARM/$BASE"
cp -p "$TARGET" "$DEST"
SUM="$(digest "$DEST")"
printf '%s  %s\n' "$SUM" "$BASE" > "$DEST.md5"

# Compare against every other arm holding a binary of the same name.
for other in "$AB_DIR"/*/"$BASE"; do
    [ -f "$other" ] || continue
    [ "$other" = "$DEST" ] && continue
    if cmp -s "$other" "$DEST"; then
        fail_identical "$other  $(digest "$other")" "$DEST  $SUM"
    fi
    echo "   differs from $other ($(digest "$other"))"
done

echo "OK: arm '$ARM' -> $DEST  md5 $SUM"
echo "   run THIS binary for the measurement, not the one in the worktree."
