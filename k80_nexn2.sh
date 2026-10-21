#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# Run Nex-N2-mini (35B-A3B, Q3_K_M) on one Tesla K80 as an OpenAI-compatible server with
# prefix caching (model id "nex-n2-mini"). Run ./install.sh --k80 first.
#
#   ./k80_nexn2.sh                          serve on 127.0.0.1:8082, dies 0,1
#   HOST=0.0.0.0 PORT=8203 ./k80_nexn2.sh   listen on the LAN
#   DIES=2,3 ./k80_nexn2.sh                 use the second K80 card
#   CTX_CAP=16384 MAX_SESSIONS=8 ./k80_nexn2.sh
#                                           context window (default 32000) / cached sessions (5)
#   NEXN2_DIR=/data/n2 ./k80_nexn2.sh       where the model lives
#                                           (default ~/.cache/pantheon/Nex-N2-mini)
#   ./k80_nexn2.sh --download-only          fetch and check the GGUF, don't serve
#   extra args go to `pantheon serve`
#
# First run downloads 16.2 GB (resumable) and checks its sha256. Send `session_id` with each
# request, and send each earlier assistant turn back with its `reasoning_content`, so later
# turns reuse the cached prefix (a 4k-token turn 2: ~38 s cold, 2-5 s cached). Prefill is the
# slow part on a K80.
set -eu
cd "$(dirname "$0")"

NAME=nex-n2-mini
GGUF=nex-agi_Nex-N2-mini-Q3_K_M.gguf
GGUF_SHA=8b05a0a21746793c0be111b4dce0318a98f6468f315669fb5e13651245f7ee58
GGUF_GB=16
MIRROR="${NEXN2_MIRROR:-https://huggingface.co/dcherrera/Nex-N2-mini-Pantheon-K80/resolve/main}"
UPSTREAM="${NEXN2_UPSTREAM:-https://huggingface.co/bartowski/nex-agi_Nex-N2-mini-GGUF/resolve/main}"
DIR="${NEXN2_DIR:-$HOME/.cache/pantheon/Nex-N2-mini}"
CTX_CAP="${CTX_CAP:-32000}"
MAX_SESSIONS="${MAX_SESSIONS:-5}"
REASONING_BUDGET="${REASONING_BUDGET:-2000}"
PORT="${PORT:-8082}"

. scripts/k80_serve.sh
k80_main "$@"
