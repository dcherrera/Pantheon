#!/bin/bash
# tests/local/bank_build.sh [make vars...]   (run ON the bank, from the repo root)
# Detached-friendly build of the appliance binary at low priority (the host is
# 2-core and shared): writes build/local_hw/build.log and build/local_hw/build.done
# (the make exit code). Example: setsid nohup tests/local/bank_build.sh USE_CUDA=1 &
cd "$(dirname "$0")/../.." || exit 1
mkdir -p build/local_hw
rm -f build/local_hw/build.done
nice -n 10 make "$@" pantheon > build/local_hw/build.log 2>&1
echo $? > build/local_hw/build.done
