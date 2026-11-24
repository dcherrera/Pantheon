#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# @file box_rebuild_restart.sh
# @brief Runs ON the box: clean-rebuild the box checkout, then restart `pantheon local`.
#
# @status Uploaded and started detached by tools/box/update_box.sh; not run by hand.
# @issues none
# @todo none
#
# Usage (from update_box.sh):
#   BOX_MAKE_FLAGS='USE_TT=1' BOX_TEST_CATALOG=1 [FT_TT_DRIVER=udrv] \
#     box_rebuild_restart.sh <box pantheon dir> <box foundry dir>
#
# Writes <pantheon>/build/box_update.status: `state=building`, then `state=done`,
# `state=built` (BOX_SKIP_RESTART=1) or `state=failed step=<step>`, each with
# `commit=<sha>`. Restarts the box exactly as tests/local/lib.sh does (lt_stop / lt_start, port 8411, state in build/local_hw).
# With BOX_TEST_CATALOG=1 it also restarts the fixture HF stand-in and starts the box
# on the fixture catalog (build needs LOCAL_TEST_CATALOG_KEY=1, which the caller adds).

PANTHEON_DIR=${1:?box pantheon dir}
FOUNDRY_DIR=${2:?box foundry dir}
STATUS_FILE=build/box_update.status
BUILD_LOG=build/box_update_build.log

cd "$PANTHEON_DIR" || exit 1
mkdir -p build
BUILT_COMMIT=$(git rev-parse --short HEAD)
FOUNDRY_COMMIT=$(git -C "$FOUNDRY_DIR" rev-parse --short HEAD)

write_box_status() {   # write_box_status <state line>
    echo "$1 commit=$BUILT_COMMIT foundry=$FOUNDRY_COMMIT" > "$STATUS_FILE"
}

fail_box_update() {    # fail_box_update <step>
    write_box_status "state=failed step=$1"
    exit 1
}

# `make clean` sweeps tests/test_* that are not .c/.h, which takes tracked files with it
# (tests/test_mtp_acceptance_gate.py). The checkout was verified clean before this ran,
# so any tracked file missing now was removed by the clean: restore it, or the next
# update would refuse a "dirty" box.
restore_files_removed_by_clean() {
    local removed
    removed=$(git ls-files --deleted)
    [ -z "$removed" ] || echo "$removed" | xargs git checkout --
}

rebuild_box_binary() {
    make clean > "$BUILD_LOG" 2>&1
    restore_files_removed_by_clean
    # shellcheck disable=SC2086  # BOX_MAKE_FLAGS is a word list on purpose
    make -j"$(nproc)" $BOX_MAKE_FLAGS FOUNDRY="$FOUNDRY_DIR" pantheon >> "$BUILD_LOG" 2>&1 ||
        return 1
    if [ "${BOX_TEST_CATALOG:-0}" = 1 ]; then
        make -s build/fixture_httpd >> "$BUILD_LOG" 2>&1 || return 1
    fi
}

restart_box_service() {
    export LT_PORT=${LT_PORT:-8411}
    export LT_MODEL_FILE=${LT_MODEL_FILE:-build/local_models/qwen3-0.6b-Q8_0.gguf}
    # shellcheck source=/dev/null
    . tests/local/lib.sh
    lt_stop
    if [ "${BOX_TEST_CATALOG:-0}" = 1 ]; then
        lt_hf_stop
        lt_hf_start || return 1
        # shellcheck disable=SC2046  # lt_catalog_flags prints a flag list
        lt_start $(lt_catalog_flags)
    else
        lt_start
    fi
}

write_box_status "state=building"
rebuild_box_binary || fail_box_update build
if [ "${BOX_SKIP_RESTART:-0}" = 1 ]; then write_box_status "state=built"; exit 0; fi
write_box_status "state=restarting"
restart_box_service || fail_box_update restart
write_box_status "state=done"
