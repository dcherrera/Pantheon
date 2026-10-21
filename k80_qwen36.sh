#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# Run Qwen3.6-35B-A3B (Q3_K_M) on one Tesla K80 as an OpenAI-compatible server with prefix
# caching (model id "qwen36"). Run ./install.sh --k80 first.
#
#   ./k80_qwen36.sh                         serve on 127.0.0.1:8081, dies 0,1
#   HOST=0.0.0.0 PORT=8201 ./k80_qwen36.sh  listen on the LAN
#   DIES=2,3 ./k80_qwen36.sh                use the second K80 card
#   CTX_CAP=32768 MAX_SESSIONS=4 ./k80_qwen36.sh
#                                           context window (default 131072) / cached sessions (2)
#   QWEN36_DIR=/data/qwen36 ./k80_qwen36.sh where the model lives
#                                           (default ~/.cache/pantheon/Qwen3.6-35B-A3B)
#   ./k80_qwen36.sh --download-only         fetch and check the GGUF, don't serve
#   extra args go to `pantheon serve`
#
# First run downloads 17.1 GB (resumable) and checks its sha256. Send `session_id` with each
# request, and send each earlier assistant turn back with its `reasoning_content`, so later
# turns reuse the cached prefix (a 4k-token turn 2: ~38 s cold, 2-5 s cached). Prefill is the
# slow part on a K80.
set -eu
cd "$(dirname "$0")"

NAME=qwen36
GGUF=Qwen_Qwen3.6-35B-A3B-Q3_K_M.gguf
GGUF_SHA=17350b1355d32397cffe7d4b446fd9361dbb01aa26bbf41fc950aa595033794f
GGUF_GB=17
MIRROR="${QWEN36_MIRROR:-https://huggingface.co/dcherrera/Qwen3.6-35B-A3B-Pantheon-K80/resolve/main}"
UPSTREAM="${QWEN36_UPSTREAM:-https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/resolve/main}"
DIR="${QWEN36_DIR:-$HOME/.cache/pantheon/Qwen3.6-35B-A3B}"
CTX_CAP="${CTX_CAP:-131072}"
MAX_SESSIONS="${MAX_SESSIONS:-2}"
REASONING_BUDGET="${REASONING_BUDGET:-8000}"
PORT="${PORT:-8081}"

. scripts/k80_serve.sh
k80_main "$@"
