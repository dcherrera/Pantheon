#!/bin/bash
# tests/llama_sched/k80_build.sh [targets...]   (run ON the bank, from the repo root, detached)
# Builds foundry's PTX (when missing) and the CUDA serve binary + the given targets at low
# priority. Deletes every output first (make 3.81 same-second skip). Writes
# build/llama_sched/build.log; last line "BUILD_DONE rc=N".
#   setsid nohup bash tests/llama_sched/k80_build.sh > /dev/null 2>&1 < /dev/null &
#
# @status complete  @issues none  @todo none
cd "$(dirname "$0")/../.." || exit 1
out=build/llama_sched
mkdir -p "$out"
exec > "$out/build.log" 2>&1
finish() { echo "BUILD_DONE rc=$1"; exit "$1"; }
targets="pantheon $*"
echo "== Pantheon $(git log --oneline -1)"
echo "== foundry $(git -C ../foundry log --oneline -1)"
if ! ls ../foundry/cuda_kernels/*.ptx > /dev/null 2>&1; then
    (cd ../foundry/cuda_kernels && nice -n 10 make -s) || finish 3
fi
for t in $targets; do rm -f "$t"; done
nice -n 10 make -s USE_CUDA=1 $targets || finish 4
for t in $targets; do md5sum "$t"; done
finish 0
