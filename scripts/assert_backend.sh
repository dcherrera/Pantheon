#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# Refuse to deploy a Pantheon binary whose COMPILED backend is not the one the
# fleet expects. One line to call, exit 0 or exit 1.
#
#   scripts/assert_backend.sh <binary> <expected-backend>
#   scripts/assert_backend.sh /opt/pantheon/pantheon cuda
#
# Why this exists: `make`, `make USE_CUDA=1` and `make USE_TT=1` used to write
# the same file, so whichever ran last owned the deployed path. A CPU-only
# binary served a 35B-A3B MoE on a 2-core no-AVX Pentium with all 8 K80 dies
# idle at 0 MiB for an unknown period, and a deploy script launched it onto a GPU
# fleet manifest and reported success. The binary knew its own backend the whole
# time — nothing asked it.
#
# The binaries are now per-backend (pantheon-cuda / pantheon-tt / pantheon-cpu)
# and `pantheon` is a symlink, so this checks what the symlink RESOLVES TO by
# asking the binary itself rather than trusting the name. `--backend` prints one
# lowercase token (cuda|tt|metal|cpu) on stdout and exits 0; it loads no model
# and opens no device, so it is free to run on every deploy.
#
# A binary too old to know the flag fails CLOSED: it is by definition from
# before per-backend names existed, which is exactly the era this guards.

set -eu

BIN="${1:-}"
WANT="${2:-}"

if [ -z "$BIN" ] || [ -z "$WANT" ]; then
	echo "usage: $0 <binary> <expected-backend>   e.g. $0 ./pantheon cuda" >&2
	exit 2
fi

if [ ! -x "$BIN" ]; then
	echo "assert_backend: '$BIN' is not an executable" >&2
	exit 1
fi

GOT=$("$BIN" --backend 2>/dev/null) || GOT=""

if [ -z "$GOT" ]; then
	echo "assert_backend: '$BIN' does not support --backend — it predates" >&2
	echo "  per-backend build names and cannot be trusted to be a $WANT build." >&2
	echo "  Rebuild: make USE_CUDA=1  (produces ./pantheon-cuda)" >&2
	exit 1
fi

if [ "$GOT" != "$WANT" ]; then
	echo "assert_backend: REFUSING to deploy '$BIN' — it is a '$GOT' build, the" >&2
	echo "  fleet expects '$WANT'. A '$GOT' binary would load the models and" >&2
	echo "  answer every request on the host CPU with every die idle." >&2
	echo "  Fix: point the deploy path at the $WANT binary —" >&2
	echo "    ln -sfn pantheon-$WANT pantheon    (no rebuild needed if it exists)" >&2
	if [ "$WANT" = cpu ]; then
		echo "  or build one: make    (a plain make IS the CPU build)" >&2
	else
		# The flag is the token uppercased: cuda -> USE_CUDA=1, tt -> USE_TT=1.
		echo "  or build one: make USE_$(printf '%s' "$WANT" | tr 'a-z' 'A-Z')=1" >&2
	fi
	exit 1
fi

echo "assert_backend: '$BIN' is a '$GOT' build ($("$BIN" --version | tail -1))"
exit 0
