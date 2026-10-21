#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# Run Ornith-1.5-9B on a Tenstorrent Blackhole p150a as an OpenAI-compatible server
# (model id "ornith", /v1/chat/completions). Run ./install.sh first.
#
#   ./run_ornith.sh                          serve on 127.0.0.1:8080
#   HOST=0.0.0.0 PORT=8301 ./run_ornith.sh   listen on the LAN
#   CTX_CAP=32768 ./run_ornith.sh            context window (default 131072)
#   ./run_ornith.sh --reasoning-budget 2048  extra args go to `pantheon serve`
#   ORNITH_DIR=/data/ornith ./run_ornith.sh  where the model files live
#                                            (default ~/.cache/pantheon/Ornith-1.5-9B)
#   HF_TOKEN=...                             sent to Hugging Face if set
#   ./run_ornith.sh --download-only          fetch and check the files, don't serve
#   ORNITH_MIRROR=<url> ORNITH_UPSTREAM=<url>  other download locations
#
# First run: downloads ~10 GB from Hugging Face (resumable) and checks each file's sha256;
# if our mirror is unreachable, the GGUF comes from ornith-ai and the MTP draft head is built
# from it. Pantheon then converts the GGUF to the card format on the first load and caches
# it beside the GGUF (~10 GB more). Later runs start straight away.
set -eu
cd "$(dirname "$0")"

MIRROR="${ORNITH_MIRROR:-https://huggingface.co/dcherrera/Ornith-1.5-9B-Pantheon-TT/resolve/main}"
UPSTREAM="${ORNITH_UPSTREAM:-https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF/resolve/main}"
DIR="${ORNITH_DIR:-$HOME/.cache/pantheon/Ornith-1.5-9B}"
GGUF=Ornith-1.5-9B-Q8_0.gguf
GGUF_SHA=22086870b009dbe9815ee752c48a82de930118a7c5ce5599590892ae03b8b010
MTP=Ornith-1.5-9B-Q8_0.bfp8.mtp.fttw
MTP_SHA=d9662d436973b0d28bd5ee09ab692a2050c12aaacdfa3aaf8027ec4173dc16fa

die() { echo "run_ornith: $*" >&2; exit 1; }

# The draft head is blk.32 of the GGUF; Pantheon does not build it on load, so build it here.
build_mtp() {
    local venv="$DIR/.venv" conv=foundry/tt/convert/tools/build/ft_tt_convert_gguf
    echo "run_ornith: building the MTP draft head from the GGUF"
    command -v python3 > /dev/null || return 1
    [ -x "$venv/bin/python" ] || python3 -m venv "$venv" || return 1
    "$venv/bin/pip" -q install numpy gguf || return 1
    make -C foundry/tt/convert/tools build/ft_tt_convert_gguf || return 1
    "$venv/bin/python" tools/build_mtp_gguf.py "$DIR/$GGUF" "$DIR/mtp.gguf" || return 1
    "$conv" "$DIR/mtp.gguf" "$DIR/$MTP" --ledger bfp8 || return 1
    rm -f "$DIR/mtp.gguf"
    touch "$DIR/$MTP.ok"
}

DOWNLOAD_ONLY=0
[ "${1:-}" = "--download-only" ] && { DOWNLOAD_ONLY=1; shift; }
[ "$DOWNLOAD_ONLY" -eq 1 ] || [ -x ./pantheon-tt ] || die "no ./pantheon-tt; run ./install.sh first"
. scripts/model_fetch.sh
mkdir -p "$DIR"
# Disk: ~10 GB for the GGUF if not here yet, ~10 GB for the card conversion unless
# --download-only or already converted, ~1 GB for the draft head and slack.
need_gb=1
[ -f "$DIR/$GGUF" ] || need_gb=$((need_gb + 10))
[ "$DOWNLOAD_ONLY" -eq 1 ] || [ -f "$DIR/Ornith-1.5-9B-Q8_0.bfp8.fttw" ] || need_gb=$((need_gb + 10))
[ "$need_gb" -gt 1 ] && mf_need_disk "$need_gb"
mf_fetch "$GGUF" "$GGUF_SHA" "$MIRROR" "$UPSTREAM" || die "could not download $GGUF"
mf_fetch "$MTP" "$MTP_SHA" "$MIRROR" || build_mtp ||
    echo "run_ornith: WARNING: no MTP draft head; running without speculative decoding"
[ "$DOWNLOAD_ONLY" -eq 1 ] && { echo "run_ornith: files ready in $DIR"; exit 0; }

MTP_ENV=()
mf_verified "$MTP" && MTP_ENV=(PN_TT_MTP="$DIR/$MTP")
echo "run_ornith: starting on ${HOST:-127.0.0.1}:${PORT:-8080} (first start converts for the card)"
exec env FT_TT_DRIVER="${FT_TT_DRIVER:-udrv}" ${MTP_ENV[@]+"${MTP_ENV[@]}"} ./pantheon-tt serve \
    --model "$DIR/$GGUF" --served-name ornith --ctx-cap "${CTX_CAP:-131072}" \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" "$@"
