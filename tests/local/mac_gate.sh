#!/bin/bash
# Pantheon Local Mac gate: `make test`, then the TT and CPU binaries, each from a
# clean binary. Writes build/local_gate/result (test=/tt=/cpu= exit codes, DONE).
# Run detached; logs in build/local_gate/.
cd "$(dirname "$0")/../.." || exit 1
out=build/local_gate
mkdir -p "$out"
rm -f "$out/result"
{
  make -j4 test > "$out/test.log" 2>&1; echo "test=$?"
  rm -f pantheon-tt; make -j4 USE_TT=1 pantheon > "$out/tt.log" 2>&1; echo "tt=$?"
  rm -f pantheon-cpu; make -j4 pantheon > "$out/cpu.log" 2>&1; echo "cpu=$?"
} > "$out/result.tmp"
echo DONE >> "$out/result.tmp"
mv "$out/result.tmp" "$out/result"
