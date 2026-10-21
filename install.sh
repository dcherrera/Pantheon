#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# Build Pantheon for a Tenstorrent Blackhole p150a and let this user drive the card,
# or (--k80) for a Tesla K80.
#
#   ./install.sh          TT: build, then set up card access (asks before using sudo)
#   ./install.sh --yes    same, without asking
#   ./install.sh --k80    K80: build pantheon-cuda (no CUDA toolkit needed; the card needs
#                         the NVIDIA 470 driver). Then ./k80_qwen36.sh or ./k80_nexn2.sh.
#
# Card access needs no vendor kernel module: Foundry's userspace driver maps the card's
# PCI BARs. The udev rule (foundry/tt/tools/60-foundry-blackhole.rules) opens those to the
# `tenstorrent` group and sets two PCI config bits that otherwise need root on every run.
# Then run ./run_ornith.sh.
set -eu
cd "$(dirname "$0")"

RULES_SRC=foundry/tt/tools/60-foundry-blackhole.rules
RULES_DST=/etc/udev/rules.d/60-foundry-blackhole.rules
YES=0
K80=0
for arg in "$@"; do
    case "$arg" in
        --yes) YES=1 ;;
        --k80) K80=1 ;;
        *) echo "install: unknown option $arg (use --yes, --k80)" >&2; exit 2 ;;
    esac
done

die() { echo "install: $*" >&2; exit 1; }

ask() {
    [ "$YES" -eq 1 ] && return 0
    printf '%s [y/N] ' "$1"
    read -r reply
    [ "$reply" = y ] || [ "$reply" = Y ]
}

check_tools() {
    local missing=() tool
    [ "$(uname -s)" = Linux ] || die "the Tenstorrent and K80 backends are Linux only"
    for tool in cc make git curl sha256sum "$@"; do
        command -v "$tool" > /dev/null || missing+=("$tool")
    done
    [ "${#missing[@]}" -eq 0 ] ||
        die "missing: ${missing[*]} (Debian/Ubuntu: sudo apt install build-essential git curl pciutils)"
}

# build_pantheon <make flag> <backend>: e.g. USE_TT=1 tt -> ./pantheon-tt
build_pantheon() {
    [ -f foundry/tensor/tensor.h ] || git submodule update --init foundry
    echo "install: building pantheon-$2 (a few minutes on a small CPU)"
    make "$1" -j"$(nproc)" "pantheon-$2"
    [ "$(./pantheon-"$2" --backend)" = "$2" ] || die "built binary is not the $2 backend"
}

# Warn (don't fail) when the K80 can't run yet: the build itself needs no driver.
check_k80_driver() {
    local driver
    if ! command -v nvidia-smi > /dev/null || ! driver=$(nvidia-smi --query-gpu=driver_version \
            --format=csv,noheader 2> /dev/null | head -1) || [ -z "$driver" ]; then
        echo "install: WARNING: no NVIDIA driver found; a K80 needs the 470 driver to run"
        return 0
    fi
    echo "install: NVIDIA driver $driver"
    case "$driver" in 470.*) ;; *) echo "install: WARNING: Kepler (K80) is supported by the 470 driver branch" ;; esac
    nvidia-smi -L | grep -q "K80" || echo "install: WARNING: no Tesla K80 listed by nvidia-smi -L"
}

# Compare rule lines only, so a reworded comment doesn't trigger sudo again.
card_access_ready() {
    [ -f "$RULES_DST" ] &&
        diff -q <(grep -v '^#' "$RULES_SRC") <(grep -v '^#' "$RULES_DST") > /dev/null &&
        id -nG | tr ' ' '\n' | grep -qx tenstorrent
}

setup_card_access() {
    if card_access_ready; then
        echo "install: card access already set up"
        return 0
    fi
    echo "install: card access needs sudo, once:"
    echo "  - create group 'tenstorrent' and add $USER to it"
    echo "  - install $RULES_DST, reload udev, re-trigger the card"
    ask "Do it now?" || { echo "install: skipped; see the header of $RULES_SRC"; return 0; }
    sudo groupadd -f tenstorrent
    sudo usermod -aG tenstorrent "$USER"
    sudo cp "$RULES_SRC" "$RULES_DST"
    sudo udevadm control --reload
    sudo udevadm trigger --subsystem-match=pci --attr-match=vendor=0x1e52
    echo "install: done. Log out and back in so the 'tenstorrent' group applies."
}

if [ "$K80" -eq 1 ]; then
    check_tools
    build_pantheon USE_CUDA=1 cuda
    check_k80_driver
    echo "install: next, ./k80_qwen36.sh or ./k80_nexn2.sh"
else
    check_tools lspci setpci
    lspci -d 1e52:b140 | grep -q . ||
        echo "install: WARNING: no Blackhole p150a (PCI 1e52:b140) found; building anyway"
    build_pantheon USE_TT=1 tt
    setup_card_access
    echo "install: next, ./run_ornith.sh"
fi
