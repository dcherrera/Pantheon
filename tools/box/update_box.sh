#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# @file update_box.sh
# @brief Run on the Mac: move a Pantheon Local box to origin/main (or BOX_PANTHEON_REF), rebuild it, restart it.
#
# @status Used for a Tenstorrent p150 box (--backend tt). The cuda profile is ready, not run yet.
# @issues none
# @todo none
#
# Usage:
#   tools/box/update_box.sh <host> --backend tt|cuda|cpu [--test-catalog] [--no-restart]
#     host            the box, as an ssh destination (user@host or a ~/.ssh/config alias)
#     --backend       tt (make USE_TT=1) | cuda (make USE_CUDA=1) | cpu (plain make)
#     --no-restart    rebuild only; leave the running box alone
#     --test-catalog  build with LOCAL_TEST_CATALOG_KEY=1 and start on the fixture
#                     catalog + HF stand-in (tests/local/lib.sh). Off by default; pass
#                     it until the real catalog is signed.
#
# The box runs from ONE place per host: ~/box/Pantheon + ~/box/foundry on the box
# (override with BOX_ROOT). Steps:
#   1. git fetch origin in the Mac Pantheon and foundry repos;
#   2. refuse if either box checkout has uncommitted tracked changes;
#   3. push origin/main (Pantheon: BOX_PANTHEON_REF) to refs/box/incoming, then `git checkout -B main` on the box;
#   4. upload box_rebuild_restart.sh and run it detached (setsid nohup), which does
#      `make clean && make <flags> FOUNDRY=<box foundry> pantheon` and restarts the box
#      the way tests/local/lib.sh lt_stop/lt_start does (port 8411, build/local_hw);
#   5. poll build/box_update.status and print the built commit.
#
# Env: FOUNDRY_REPO (Mac foundry repo; default foundry/ or ../foundry of this checkout or
#      of the main checkout when run from a worktree),
#      BOX_ROOT (absolute path on the box; default $HOME/box there), BOX_TIMEOUT_SECONDS (default 3600),
#      BOX_PANTHEON_REF (Pantheon ref to put on the box; default origin/main, e.g.
#      origin/<branch> to try an unmerged fix; foundry always gets origin/main),
#      BOX_RUN_ENV (env for `pantheon local`; tt default FT_TT_DRIVER=udrv, which
#      refuses to start while another process holds the card: BOX_RUN_ENV= runs the box
#      CPU-resident, e.g. next to another serve; on a multi-GPU CUDA box,
#      BOX_RUN_ENV='CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3' keeps the
#      box off dies another job is using).

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PANTHEON_REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel) || exit 1
BOX_TIMEOUT_SECONDS=${BOX_TIMEOUT_SECONDS:-3600}

die() { echo "update_box: $*" >&2; exit 1; }

print_usage() {
    sed -n '/^# Usage:/,/^#$/p' "$0" | sed 's/^# \{0,1\}//'
}

# Sets BOX_HOST_MAKE_FLAGS / BOX_HOST_RUN_ENV for a backend.
select_host_build_profile() {
    case "$1" in
        tt)   BOX_HOST_MAKE_FLAGS="USE_TT=1"; BOX_HOST_RUN_ENV="FT_TT_DRIVER=udrv" ;;
        cuda) BOX_HOST_MAKE_FLAGS="USE_CUDA=1"; BOX_HOST_RUN_ENV="" ;;
        cpu)  BOX_HOST_MAKE_FLAGS=""; BOX_HOST_RUN_ENV="" ;;
        *) die "unknown backend '$1' (tt | cuda | cpu)" ;;
    esac
}

find_mac_foundry_repo() {
    local candidate main_checkout
    main_checkout=$(cd "$(git -C "$PANTHEON_REPO" rev-parse --git-common-dir)/.." && pwd)
    for candidate in "${FOUNDRY_REPO:-}" "$PANTHEON_REPO/foundry" "$PANTHEON_REPO/../foundry" \
                     "$main_checkout/foundry"; do
        [ -n "$candidate" ] && [ -f "$candidate/tensor/tensor.h" ] &&
            git -C "$candidate" rev-parse --git-dir > /dev/null 2>&1 &&
            { echo "$candidate"; return 0; }
    done
    return 1
}

run_on_box() { ssh -o ConnectTimeout=20 -o ServerAliveInterval=20 "$BOX_HOST" "$@"; }

# Creates the box checkout as an empty repo if it does not exist yet, then refuses
# when it holds uncommitted tracked changes (someone's work; never overwritten).
prepare_box_checkout() {   # prepare_box_checkout <box dir>
    run_on_box "set -e
        if [ ! -d '$1/.git' ]; then mkdir -p '$1'; git -C '$1' init -q -b main; fi
        dirty=\$(git -C '$1' status --porcelain --untracked-files=no)
        if [ -n \"\$dirty\" ]; then echo \"\$dirty\"; exit 3; fi" ||
        die "$BOX_HOST:$1 has uncommitted changes (or ssh failed); not touching it"
}

push_ref_to_box() {   # push_ref_to_box <mac repo> <box dir> <ref>
    local commit
    git -C "$1" fetch -q origin || die "git fetch origin failed in $1"
    commit=$(git -C "$1" rev-parse --verify -q "$3^{commit}") || die "no ref $3 in $1"
    git -C "$1" push -q -f "$BOX_HOST:$2" "$commit:refs/box/incoming" ||
        die "push to $BOX_HOST:$2 failed"
    run_on_box "git -C '$2' checkout -q -B main refs/box/incoming" ||
        die "checkout of main failed in $BOX_HOST:$2"
}

start_detached_rebuild() {
    local make_flags=$BOX_HOST_MAKE_FLAGS test_catalog=0
    if [ "$USE_TEST_CATALOG" = 1 ]; then
        make_flags="$make_flags LOCAL_TEST_CATALOG_KEY=1"; test_catalog=1
    fi
    run_on_box "cat > '$BOX_ROOT/box_rebuild_restart.sh'" < "$SCRIPT_DIR/box_rebuild_restart.sh" ||
        die "upload of box_rebuild_restart.sh failed"
    run_on_box "mkdir -p '$BOX_PANTHEON_DIR/build' && rm -f '$BOX_STATUS_FILE' &&
        $BOX_HOST_RUN_ENV BOX_MAKE_FLAGS='$make_flags' BOX_TEST_CATALOG=$test_catalog \
        BOX_SKIP_RESTART=$SKIP_RESTART setsid nohup bash '$BOX_ROOT/box_rebuild_restart.sh' \
        '$BOX_PANTHEON_DIR' '$BOX_FOUNDRY_DIR' > '$BOX_ROOT/box_update.out' 2>&1 < /dev/null &" ||
        die "could not start the rebuild on $BOX_HOST"
}

# Polls the status file; ssh failures are retried (the link is flaky).
wait_for_rebuild() {
    local waited=0 status=""
    while [ "$waited" -lt "$BOX_TIMEOUT_SECONDS" ]; do
        sleep 15; waited=$((waited + 15))
        status=$(run_on_box "cat '$BOX_STATUS_FILE' 2>/dev/null" 2> /dev/null) || continue
        case "$status" in
            state=done*) echo "$status"; return 0 ;;
            state=built*) echo "$status"; return 0 ;;
            state=failed*)
                echo "$status" >&2
                run_on_box "tail -30 '$BOX_PANTHEON_DIR/build/box_update_build.log' \
                    '$BOX_PANTHEON_DIR/build/local_hw/serve.log' 2>/dev/null" >&2
                return 1 ;;
        esac
    done
    die "timed out after ${BOX_TIMEOUT_SECONDS}s; last status: $status"
}

BOX_HOST=""; BOX_BACKEND=""; USE_TEST_CATALOG=0; SKIP_RESTART=0
while [ $# -gt 0 ]; do
    arg=$1; shift
    case "$arg" in
        --backend) BOX_BACKEND=${1:-}; shift ;;
        --test-catalog) USE_TEST_CATALOG=1 ;;
        --no-restart) SKIP_RESTART=1 ;;
        -h|--help) print_usage; exit 0 ;;
        -*) die "unknown option $arg" ;;
        *) BOX_HOST=$arg ;;
    esac
done
[ -n "$BOX_HOST" ] && [ -n "$BOX_BACKEND" ] || { print_usage; exit 2; }
select_host_build_profile "$BOX_BACKEND"
BOX_HOST_RUN_ENV=${BOX_RUN_ENV-$BOX_HOST_RUN_ENV}
MAC_FOUNDRY_REPO=$(find_mac_foundry_repo) || die "no foundry repo found; set FOUNDRY_REPO"
BOX_ROOT=${BOX_ROOT:-$(run_on_box 'echo "$HOME/box"')} || die "ssh to $BOX_HOST failed"
BOX_PANTHEON_DIR=$BOX_ROOT/Pantheon
BOX_FOUNDRY_DIR=$BOX_ROOT/foundry
BOX_STATUS_FILE=$BOX_PANTHEON_DIR/build/box_update.status

prepare_box_checkout "$BOX_PANTHEON_DIR"
prepare_box_checkout "$BOX_FOUNDRY_DIR"
push_ref_to_box "$PANTHEON_REPO" "$BOX_PANTHEON_DIR" "${BOX_PANTHEON_REF:-origin/main}"
push_ref_to_box "$MAC_FOUNDRY_REPO" "$BOX_FOUNDRY_DIR" origin/main
start_detached_rebuild
echo "update_box: building on $BOX_HOST ($BOX_HOST_MAKE_FLAGS); polling $BOX_STATUS_FILE"
FINAL_STATUS=$(wait_for_rebuild) || exit 1
echo "update_box: $BOX_HOST: $FINAL_STATUS"
